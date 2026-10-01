"""Train a support-specific BERT head without changing the education model.

The BERT backbone is frozen. Results cover synthetic parent/template groups,
not a real customer test set. The existing 88 retrieval questions are not read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloudcare.neural import SUPPORT_LABELS, resolve_device
from cloudcare.settings import Settings


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_dataset(root: Path, seed: int, max_per_label: int) -> list[dict]:
    knowledge_path = root / "data/knowledge.jsonl"
    knowledge = [json.loads(line) for line in knowledge_path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["id"]: row for row in knowledge}
    faq_path = root / "data/faq.jsonl"
    support = []
    compressed_faq = faq_path.with_suffix(".jsonl.gz")
    if faq_path.is_file() or compressed_faq.is_file():
        faq_bytes = faq_path.read_bytes() if faq_path.is_file() else gzip.decompress(compressed_faq.read_bytes())
        for line in faq_bytes.decode("utf-8").splitlines():
            faq = json.loads(line)
            source = by_id.get(faq["source_id"])
            if source:
                support.append({"query": faq["question"], "label": "support_knowledge",
                                "group": "support:" + source.get("parent_id", source["id"]),
                                "source_id": source["id"]})
    else:
        for source in knowledge:
            for prefix in ("请说明", "我想了解", "怎么处理", "帮我查一下"):
                support.append({"query": prefix + source["title"], "label": "support_knowledge",
                                "group": "support:" + source.get("parent_id", source["id"]),
                                "source_id": source["id"]})
    topics = sorted({row["category"] for row in knowledge})
    handoff_templates = [
        "我不想看{topic}说明，请转人工客服", "{topic}问题请安排客服人员联系我",
        "我要投诉{topic}服务，找人工处理", "{topic}一直无法解决，请升级给工程师",
        "帮我为{topic}创建工单，需要人工跟进", "请不要自动回答{topic}，我要和真人聊",
        "给我接通{topic}客服专员", "{topic}需要退款审批，请人工受理",
        "我想把{topic}故障提交给支持团队", "能让工作人员帮我处理{topic}吗",
        "{topic}已经排查过了，我要升级处理", "我要反馈{topic}服务质量，需要负责人回复",
        "请为{topic}登记人工服务请求", "{topic}问题找谁投诉，帮我转接",
        "{topic}要人工介入，安排专人跟进", "请为我预约{topic}技术支持回访",
    ]
    contexts = ["", "我的组织正在使用系统，", "之前咨询过了，", "情况比较紧急，"]
    handoff = [{"query": context + template.format(topic=topic), "label": "handoff",
                "group": f"handoff:template-{index}", "source_id": None}
               for index, template in enumerate(handoff_templates) for topic in topics for context in contexts]
    external_topics = ["明天上海的天气", "番茄炒蛋的做法", "唐诗的历史", "电影推荐", "股票涨跌预测",
                       "足球比赛结果", "旅行景点推荐", "英语作文", "微积分习题", "游戏攻略",
                       "养花的方法", "儿童故事", "宇宙起源", "小说创作", "汽车购买建议", "手机评测",
                       "吉他弹奏", "猫咪品种", "马拉松训练", "绘画教程", "音乐欣赏", "古代历史",
                       "服装穿搭", "数学证明"]
    outside_templates = ["请介绍{topic}", "帮我回答一下{topic}", "我想知道{topic}", "聊一聊{topic}",
                         "给我讲讲{topic}", "能帮我查{topic}吗", "我咨询的是{topic}", "需要一份{topic}的资料"]
    outside = [{"query": context + template.format(topic=topic), "label": "out_of_scope",
                "group": f"out_of_scope:topic-{index}", "source_id": None}
               for index, topic in enumerate(external_topics) for template in outside_templates for context in contexts]
    generator = random.Random(seed)
    dataset = []
    seen_queries = set()
    for label_rows in (support, handoff, outside):
        generator.shuffle(label_rows)
        unique = []
        for row in label_rows:
            if row["query"] not in seen_queries:
                unique.append(row)
                seen_queries.add(row["query"])
        selected = unique[:max_per_label]
        groups = sorted({row["group"] for row in selected})
        generator.shuffle(groups)
        test_groups = set(groups[:max(1, round(len(groups) * .2))])
        for row in selected:
            row["split"] = "test" if row["group"] in test_groups else "train"
            row["synthetic"] = True
            dataset.append(row)
    generator.shuffle(dataset)
    return dataset


def evaluate(predictions: list[int], targets: list[int]) -> dict:
    matrix = [[0] * len(SUPPORT_LABELS) for _ in SUPPORT_LABELS]
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[target][prediction] += 1
    per_label = {}
    for index, label in enumerate(SUPPORT_LABELS):
        tp = matrix[index][index]
        precision = tp / max(1, sum(row[index] for row in matrix))
        recall = tp / max(1, sum(matrix[index]))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": sum(matrix[index])}
    return {"accuracy": sum(matrix[i][i] for i in range(3)) / max(1, len(targets)),
            "macro_f1": sum(item["f1"] for item in per_label.values()) / 3,
            "per_label": per_label, "confusion_matrix": matrix}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--max-per-label", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    settings = Settings.load()
    output = args.output or settings.bert_model_path
    output.mkdir(parents=True, exist_ok=True)
    evidence_dir = ROOT / "evaluation/support_router"
    dataset = build_dataset(ROOT, args.seed, args.max_per_label)
    dataset_path = evidence_dir / "dataset.jsonl"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in dataset), encoding="utf-8")
    train_rows = [row for row in dataset if row["split"] == "train"]
    test_rows = [row for row in dataset if row["split"] == "test"]
    train_groups = {row["group"] for row in train_rows}
    test_groups = {row["group"] for row in test_rows}
    if train_groups & test_groups:
        raise RuntimeError("Training and test groups overlap")
    report = {"task": "customer_support_routing", "labels": list(SUPPORT_LABELS),
              "created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
              "dataset_sha256": sha256(dataset_path), "knowledge_sha256": sha256(ROOT / "data/knowledge.jsonl"),
              "base_model": str(settings.bert_base_model_path),
              "training_scope": "BERT classification head fine-tuning; backbone frozen; synthetic support data",
              "evaluation_scope": "synthetic held-out parent/template/topic groups; not real customer or retrieval benchmark",
              "train_count": len(train_rows), "test_count": len(test_rows),
              "train_labels": dict(Counter(row["label"] for row in train_rows)),
              "test_labels": dict(Counter(row["label"] for row in test_rows)),
              "train_groups": len(train_groups), "test_groups": len(test_groups),
              "group_overlap": 0, "retrieval_evaluation_used_for_training": False}
    if args.build_only:
        write_json(evidence_dir / "dataset_report.json", report)
        print(json.dumps(report, ensure_ascii=False))
        return
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    base = str(settings.bert_base_model_path)
    local = Path(base).is_dir()
    tokenizer = BertTokenizer.from_pretrained(base, local_files_only=local)
    model = BertForSequenceClassification.from_pretrained(
        base, num_labels=3, id2label=dict(enumerate(SUPPORT_LABELS)),
        label2id={label: index for index, label in enumerate(SUPPORT_LABELS)},
        ignore_mismatched_sizes=True, local_files_only=local)
    # Always reset the head, including when an input checkpoint has three labels.
    model.classifier = torch.nn.Linear(model.config.hidden_size, 3)
    model.to(device)
    for parameter in model.bert.parameters():
        parameter.requires_grad = False
    model.bert.eval()
    started = time.perf_counter()
    features = []
    for offset in range(0, len(dataset), args.batch_size):
        texts = [row["query"] for row in dataset[offset:offset + args.batch_size]]
        encoding = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        encoding = {key: value.to(device) for key, value in encoding.items()}
        with torch.inference_mode():
            feature = model.bert(**encoding).pooler_output.detach().cpu()
        features.append(feature)
        if offset % (args.batch_size * 20) == 0:
            print(f"BERT feature batches: {min(offset + args.batch_size, len(dataset))}/{len(dataset)}", flush=True)
    x = torch.cat(features).to(device)
    y = torch.tensor([SUPPORT_LABELS.index(row["label"]) for row in dataset], device=device)
    train_indices = torch.tensor([i for i, row in enumerate(dataset) if row["split"] == "train"], device=device)
    test_indices = torch.tensor([i for i, row in enumerate(dataset) if row["split"] == "test"], device=device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=.003, weight_decay=.01)
    for _ in range(args.epochs):
        permutation = train_indices[torch.randperm(len(train_indices), device=device)]
        for indices in permutation.split(128):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model.classifier(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits = model.classifier(x[test_indices])
        predictions = logits.argmax(dim=-1).cpu().tolist()
        targets = y[test_indices].cpu().tolist()
        confidence = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().tolist()
    report.update(evaluate(predictions, targets))
    report.update({"device": device, "epochs": args.epochs, "backbone_frozen": True,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "seconds": time.perf_counter() - started})
    model.cpu().save_pretrained(output)
    tokenizer.save_pretrained(output)
    predictions_path = evidence_dir / "test_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row, prediction, probability in zip(test_rows, predictions, confidence, strict=True):
            handle.write(json.dumps({**row, "prediction": SUPPORT_LABELS[prediction], "confidence": probability}, ensure_ascii=False) + "\n")
    report["checkpoint_sha256"] = sha256(output / "model.safetensors")
    report["predictions_sha256"] = sha256(predictions_path)
    write_json(output / "training_report.json", report)
    write_json(evidence_dir / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
