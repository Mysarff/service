"""Reproducible fictional business corpus. No network, customer records, or LLM calls."""
import argparse
import hashlib
import gzip
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = '2026-09-25'
# Explicit module-specific business rules; these are fictional product specifications.
# code, name, route, object, role, required fields, states, quota, import cap, retention,
# common error, recovery, automation trigger
MODULES = [
 ('ACC','账号安全','个人设置/账号','账号','账号本人','工作邮箱、显示名称','待验证→正常→临时锁定',5,100,30,'验证码收不到','检查垃圾邮件和邮箱拼写，等待2分钟后重发；SSO账号由企业IT核对身份目录','登录设备变化'),
 ('ORG','组织成员','管理后台/成员','成员','组织管理员','工作邮箱、部门、角色','已邀请→已加入→已停用',200,200,90,'邀请链接过期','邀请有效期7天；撤销旧邀请后重发，核对被邀请邮箱，不要新建重复账号','成员停用'),
 ('TKT','工单管理','服务台/工单','工单','客服主管','问题摘要、问题类别、影响范围','新建→处理中→待客户反馈→已解决→已关闭',50000,1000,180,'工单没有分配','确认客服组有在线坐席，并检查分配规则的类别条件与兜底组','优先级变更'),
 ('CHAT','在线会话','服务台/在线会话','会话','坐席主管','渠道、访客标识、咨询主题','排队→接入→转接→结束',100,500,90,'访客一直排队','检查渠道营业时段、坐席在线状态和并发配额，必要时提示访客提交工单','会话转人工'),
 ('KB','知识中心','内容中心/知识库','知识文章','知识库管理员','标题、分类、正文、版本','草稿→审核中→已发布→已归档',10000,500,180,'更新后搜不到文章','确认文章已发布且访问范围正确，等待索引完成；记录文章ID供管理员检查索引任务','文章发布'),
 ('ORD','订单服务','交易中心/订单','订单','订单专员','订单号、客户标识、商品编码、数量','待支付→已支付→处理中→已完成→已取消',100000,1000,365,'重复订单','以外部订单号检查幂等键；未确认支付状态前不要重复发起扣款','订单支付成功'),
 ('SUB','订阅管理','组织设置/订阅','订阅','账单管理员','套餐、账单周期、组织标识','试用→有效→待续费→到期→已取消',10,100,365,'自动续费失败','核对付款方式有效性和支付机构限制；取消续费不会自动触发退款，退款由人工核对订单条款','续费失败'),
 ('INV','发票管理','交易中心/发票','发票申请','财务管理员','订单号、企业抬头、税号、接收邮箱','待审核→开具中→已开具→更正中',20000,500,365,'发票抬头错误','未开具时退回修改；已开具时发起更正申请，由财务核实处理，不承诺自动作废','发票开具完成'),
 ('SHIP','物流服务','履约中心/物流','物流单','履约专员','订单号、承运商、运单号','待发货→已揽收→运输中→已签收→异常',100000,1000,180,'物流轨迹停更','核对运单号及承运商接口状态，记录最后事件时间；签收争议转人工联系承运商','物流异常'),
 ('RET','退换服务','售后中心/退换','售后单','售后主管','原订单号、商品编码、售后原因','待审核→待退回→验收中→处理中→已完结',30000,500,180,'退货退款未到账','核对审核、验收和支付渠道状态；退款资格及到账时间取决于订单规则和渠道，不凭聊天直接承诺','退回商品验收'),
 ('ASSET','资产管理','运维中心/资产','资产','资产管理员','资产编号、设备类型、保管部门','在库→领用→维修→归还→报废',20000,1000,365,'资产编号重复','导入前按资产编号去重，核对历史报废记录；确需复用编号应由管理员确认关联关系','资产转移'),
 ('PRJ','项目协作','协作中心/项目','项目','项目管理员','项目名称、负责人、开始日期','规划→进行中→阻塞→完成→归档',500,200,180,'成员看不到项目','检查项目成员列表、所属组织和角色授权；外部分享由组织管理员控制','项目里程碑到期'),
 ('APR','审批流程','流程中心/审批','审批单','流程管理员','流程模板、申请人、申请说明','草稿→审批中→已通过→已拒绝→已撤回',20000,500,180,'审批停在离职人员','暂停该流程后指定合法替代审批人，保留原审批轨迹；不能绕过审批链直接标记通过','审批节点超时'),
 ('REPORT','报表分析','数据中心/报表','报表任务','数据分析员','时间范围、统计维度、指标','已提交→运行中→已完成→失败',50,200,30,'报表数据不一致','核对筛选条件、时区和统计口径，再比较相同时间窗口；延迟数据以最后刷新时间为准','报表生成完成'),
 ('API','开放接口','开发者中心/应用','API应用','集成管理员','应用名称、授权范围、回调地址','待启用→已启用→限流→已停用',20,100,90,'接口返回429','按Retry-After等待并使用指数退避，减少并发；不可通过轮换密钥规避限流','接口连续失败'),
 ('AUDIT','审计中心','安全中心/审计','审计查询','安全管理员','查询时间范围、事件类型','已提交→查询中→已完成→已过期',20,200,180,'审计记录导出失败','缩小时间范围并核对安全管理员权限，保留查询任务号；禁止向普通工单上传完整审计记录','高风险访问事件'),
]
TOPICS = [
 ('access','开通与访问',['开通','入口','在哪里','启用']),
 ('create','创建与必填字段',['新建','创建','必填','填写']),
 ('status','状态流转',['状态','流程','关闭','撤回']),
 ('permission','角色与授权',['权限','角色','授权','看不到']),
 ('import','批量导入',['导入','CSV','模板','乱码']),
 ('export','数据导出',['导出','下载','备份']),
 ('search','查询与索引',['搜索','查询','检索','筛选']),
 ('automation','通知与自动化',['通知','自动化','触发','提醒']),
 ('incident','常见故障排查',['报错','故障','失败','异常']),
 ('quota','配额与限制',['配额','限制','数量','上限']),
 ('retention','归档与保留',['保留','回收站','删除','恢复','归档']),
 ('handoff','人工升级与交接',['人工','升级','工单','交接']),
]
EXTRA = [
 ('ACC','忘记密码与重置密码','登录页点击忘记密码，输入注册工作邮箱，通过重置邮件设置新密码。链接过期后重新申请；企业SSO账号的密码由公司身份提供方管理，请联系企业IT管理员。客服不会索取密码或验证码。',['忘记密码','重置密码','找回密码']),
 ('ACC','更换手机与多因素认证','更换手机前在安全设置添加新验证器并验证成功，再移除旧设备。妥善离线保存恢复码。设备和恢复码同时丢失时，由组织管理员发起身份核验，不可通过普通客服直接关闭第二因素。',['MFA','验证器','手机','恢复码']),
 ('ORG','离职人员工作交接','组织管理员先将离职人员负责的工单、项目和知识文章交给接任人，确认无孤立资源后停用成员并撤销会话。停用不会自动删除已归属组织的资料，不能将离职账号直接借给接任人使用。',['离职','交接','停用成员']),
 ('TKT','优先级与响应时间','工单按影响范围分为单用户、部门和全组织影响。分级用于队列排序，不代表响应时间承诺。没有合同依据时不得回答多少分钟必定解决；全组织故障先关联故障公告，再由值班负责人协调。',['SLA','响应时间','优先级']),
 ('CHAT','非营业时间接待','渠道关闭时显示离线留言入口；留言保存为新工单并显示工单编号。不要展示虚构坐席在线状态，也不要承诺立即回复。营业时段由坐席主管在渠道设置维护。',['离线','非营业时间','留言']),
 ('KB','知识发布前审查','知识文章先由业务负责人核对政策和生效范围，再由知识库管理员发布。AI生成或旧版本资料不能直接作为当前政策。发布时保留版本号、生效日期和来源；撤回失效文档后重新构建检索索引。',['知识审核','发布','版本','政策']),
 ('ORD','订单取消条件','待支付订单可由订单专员核对后取消；已支付订单取消需要先确认履约状态并走售后审核。已出库订单不能仅修改数据库状态，需关联退换流程；是否退款取决于审核结果。',['取消订单','已支付','出库']),
 ('SUB','取消自动续费与退款','账单管理员在组织设置/订阅中关闭自动续费并核对生效日期。关闭续费不等于退款，当前服务是否持续到周期末以订单条款为准。退款申请需提交订单编号由账单团队审核，不能保证全额或即时到账。',['取消续费','退款','自动续费']),
 ('INV','发票申请资料','申请发票需提供订单号、企业抬头、税号和接收邮箱。先核对订单付款状态与购买方一致性，再提交财务审核。发票类型和处理时限以实际订单约定为准；不得通过普通聊天发送银行敏感信息。',['申请发票','税号','抬头']),
 ('SHIP','签收争议处理','显示已签收但客户未收货时，履约专员核对运单号、签收时间和承运商凭证，再联系收件方确认代收情况。不得在没有证据时直接认定客户已收到；无法确认时建立异常工单由人工跟进。',['签收','未收到','物流争议']),
 ('RET','退款资格与到账边界','客服不能仅凭聊天决定退款资格。售后主管需核对原订单条款、退回商品验收和支付渠道状态。此演示没有无条件退款承诺，也没有统一到账天数；需要订单核验时转人工。',['退款资格','到账','退货']),
 ('ASSET','资产领用交接','资产管理员确认设备编号、领用人和保管部门后登记领用，交接双方确认设备状态。归还时检查配件并记录损坏情况；领用记录和维修记录均关联同一资产编号，不能覆盖历史交接记录。',['领用','归还','交接']),
 ('PRJ','项目可见范围','项目管理员按成员或组授权；普通成员看不到未授权项目。组织管理员身份也不应作为公开外部分享的理由。跨组织协作需经过明确批准，离职或协作结束时及时撤回访问权限。',['看不到项目','项目权限','外部分享']),
 ('APR','审批人离职后的替换','流程管理员先暂停受影响流程，按组织制度指定替代审批人并保留变更原因与审计轨迹，再恢复审批。已完成的审批意见不能改写；客服不能绕过审批链直接批准。',['离职审批人','卡住','替换审批人']),
 ('API','API密钥泄露与轮换','怀疑API密钥泄露时立即由集成管理员吊销旧密钥，创建最小权限新密钥并更新调用方，检查异常调用日志。密钥不能放入前端、公开Git仓库或工单，客服无需知道密钥明文。',['密钥泄露','轮换','令牌']),
 ('AUDIT','审计导出与隐私范围','审计导出可能含用户标识、IP和资源名称，仅限获授权的安全调查使用。提交客服问题时只提供脱敏事件编号和时间范围；不要在普通工单或公开仓库附上原始日志。',['隐私','脱敏','审计导出']),
]

def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')

def build(tickets=6000):
    docs, faqs = [], []
    docs_dir = ROOT / 'data' / 'manuals'
    docs_dir.mkdir(parents=True, exist_ok=True)
    for code, name, route, obj, role, fields, states, quota, cap, retention, error, recovery, trigger in MODULES:
        content = [
          f'适用模块：{name}。进入「{route}」。由{role}先确认所属组织与模块授权，再为业务成员开通。首次访问请使用演示组织核对权限；找不到入口时检查当前组织和角色，不要重复注册账号。',
          f'创建{obj}：1.进入「{route}」点击新建；2.填写{fields}；3.保存前检查唯一标识与所属组织；4.提交后记录资源ID。缺少必填字段会拒绝保存；重复提交应使用原记录继续编辑，避免产生多条业务记录。',
          f'{obj}状态依次为「{states}」。操作前读取当前状态，仅对界面允许的下一状态执行变更。涉及支付、审批或停用的变更由{role}复核。已结束的记录保留审计轨迹；恢复流程应新建关联处理记录，不能覆盖历史结果。',
          f'{name}管理权限属于{role}；普通成员只能访问授权范围内的{obj}。跨组织不可见是预期行为。申请权限时注明资源ID、用途和所需时限，由组织管理员按最小权限审批。离职成员先交接后停用，禁止共享管理员账号。',
          f'{name}批量导入：从「{route}」下载当前模板，必填列为{fields}。采用UTF-8 CSV，单批最多{cap}行；先用5行确认字段映射。对唯一标识去重，导入后核对成功数与失败行原因。密码、令牌和不必要的个人信息不能写入模板。',
          f'{name}导出由{role}执行。选择组织、时间范围和字段后提交导出任务，再凭任务ID查看进度。仅导出有权限的{obj}，演示下载链接有效期24小时；到期后重新申请。下载完成核对条数，将文件存放于组织批准的位置。',
          f'{name}查询优先使用{obj}编号；也可按状态、负责人和更新时间筛选。查不到时核对组织与访问权限，再检查记录是否已提交。演示环境索引目标刷新间隔为60秒，但这不是可用性承诺；以实际索引状态为准。',
          f'{name}支持在「{trigger}」时触发通知。由{role}选择接收组和通知渠道，先发测试事件确认目标，再启用规则。同一事件按资源ID和事件ID去重；无人接收时进入待处理队列，不自动扩大数据可见范围。',
          f'{name}常见问题：{error}。处理步骤：{recovery}。排查后记录资源ID、时间和脱敏错误信息；若仍失败由{role}提交工单。不要关闭身份校验或删除历史记录来绕过错误。',
          f'{name}在此虚构标准套餐中配置{quota}个{obj}额度，单批导入上限{cap}行。这些数值是可复现的演示规则，不是实测容量。达到额度时先核对历史记录，再申请扩容；不得承诺实际服务器能支撑同样并发。',
          f'{name}归档记录的演示保留期为{retention}天；删除先进入回收站，回收站保留7天。{role}恢复前确认资源归属。清空或到期删除不能由普通客服恢复；涉及合同或审计留存时先联系组织管理员暂停删除。',
          f'{name}问题先收集组织标识、{obj}编号、发生时间、影响范围和已尝试步骤，由{role}提交人工工单。仅提交脱敏截图，不提交密码或令牌。严重影响多个成员时标明业务影响；响应时间以实际支持合同为准，本示例不虚构SLA。',
        ]
        parent = f'{code}-GUIDE'
        markdown = [f'# 云栈 CloudCare · {name}手册', f'版本：{VERSION}。本文件为合成业务资料，所有规则均为演示设定。\n']
        for i, ((topic, title, tags), text) in enumerate(zip(TOPICS, content), 1):
            ident = f'{code}-{i:02d}'
            if topic == 'incident': title = error
            question = f'{name}的{title}怎么办？'
            row = dict(id=ident, parent_id=parent, title=f'{name}｜{title}', category=name, content=text,
                       tags=tags + [name,obj,code,error if topic=='incident' else title],
                       source=f'data/manuals/{code}.md', version=VERSION, synthetic=True, topic=topic)
            docs.append(row)
            markdown.append(f'## {ident} {title}\n\n{text}\n')
            # These are paraphrase templates for FAQ coverage, not independent business facts or held-out evaluation.
            variants = [question, f'请说明{name}{title}的操作步骤', f'{name}有关{title}的规定是什么',
                        f'我想了解{name}的{title}', f'{name}使用中遇到{title}问题怎么处理',
                        f'客服您好，{name}{title}能解释一下吗', f'{name}：{title}有什么注意事项', f'{name}的{title}由谁处理']
            for v, q in enumerate(variants):
                faqs.append(dict(id=f'FAQ-{ident}-{v+1}', question=q, answer=text, source_id=ident, synthetic=True))
        (docs_dir / f'{code}.md').write_text('\n\n'.join(markdown), encoding='utf-8', newline='\n')
    extra_counts = {}
    for code, title, text, tags in EXTRA:
        extra_counts[code] = extra_counts.get(code, 12) + 1
        ident = f'{code}-{extra_counts[code]:02d}'
        name = next(m[1] for m in MODULES if m[0] == code)
        docs.append(dict(id=ident,parent_id=f'{code}-GUIDE',title=f'{name}｜{title}',category=name,content=text,
                         tags=tags,source=f'data/manuals/{code}.md',version=VERSION,synthetic=True,topic='special'))
        with (docs_dir/f'{code}.md').open('a',encoding='utf-8',newline='\n') as f: f.write(f'\n\n## {ident} {title}\n\n{text}\n')
        for i, prefix in enumerate(['请解释','如何处理','我想了解','咨询一下','客服您好，','请说明','帮我查一下','关于']):
            faqs.append(dict(id=f'FAQ-{ident}-{i+1}',question=f'{prefix}{name}{title}',answer=text,source_id=ident,synthetic=True))
    jsonl(ROOT/'data/knowledge.jsonl', docs)
    jsonl(ROOT/'data/faq.jsonl', faqs)
    rng = random.Random(20260925)
    def conversations():
        for i in range(tickets):
            faq = rng.choice(faqs)
            yield dict(id=f'SIM-{i+1:07d}', organization=f'demo-org-{rng.randrange(1,101):03d}',
                       channel=rng.choice(['web','email','helpdesk']), source_id=faq['source_id'],
                       messages=[{'role':'user','content':faq['question']}, {'role':'assistant','content':faq['answer']}],
                       synthetic=True, usage='workflow_demo_not_independent_knowledge')
    jsonl(ROOT/'data/simulated_tickets.jsonl', conversations())
    for name in ('faq.jsonl','simulated_tickets.jsonl'):
        path=ROOT/'data'/name
        path.with_suffix(path.suffix+'.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    counts=dict(source_manuals=len(MODULES), knowledge_units=len(docs), faq_variants=len(faqs), simulated_tickets=tickets)
    files={str(p.relative_to(ROOT)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in sorted((ROOT/'data').rglob('*')) if p.is_file() and p.name!='manifest.json' and p.suffix!='.gz' and 'uploads' not in p.parts}
    manifest=dict(version=VERSION, seed=20260925, synthetic=True, counts=counts, files=files,
                  note='FAQ variants and tickets repeat source facts; they are NOT additional knowledge units. No measured 800k-scale claim.')
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(counts,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--tickets',type=int,default=6000); args=parser.parse_args()
    if not 0<=args.tickets<=800000: parser.error('tickets must be between 0 and 800000')
    build(args.tickets)
