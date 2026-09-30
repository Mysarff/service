"""Execute the shipped page's ask/clear handlers against the local HTTP server."""
import shutil
import subprocess
import unittest

import engine
import test_system


@unittest.skipUnless(shutil.which('node'), 'Node.js is needed to execute the frontend JavaScript')
class FrontendHistoryTests(unittest.TestCase):
    setUp=test_system.SystemTests.setUp
    tearDown=test_system.SystemTests.tearDown

    def test_bounded_topic_survives_browser_roundtrips_and_resets(self):
        script=r"""
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[1],'utf8');
const code=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements=new Map(), requests=[], responses=[];
function element(){return {value:'',files:[],childNodes:[{textContent:''}],
    addEventListener(){},append(){},remove(){},replaceChildren(){},focus(){}}}
const context=vm.createContext({console,Option:function(){},alert(){},prompt(){return null},
    document:{querySelector(id){if(!elements.has(id))elements.set(id,element());return elements.get(id)},
              querySelectorAll(){return []},createElement:element},
    async fetch(path,options){
        if(path==='/api/chat')requests.push(JSON.parse(options.body));
        const response=await fetch(process.argv[2]+path,options);
        if(path==='/api/chat')responses.push(await response.clone().json());
        return response;
    }});
vm.runInContext(code,context);
async function ask(query){
    await vm.runInContext('ask('+JSON.stringify(query)+')',context);
    const response=responses.at(-1);
    assert.ok(response && !response.error);
    assert.ok(response.history.length<=5);
    assert.ok(response.history.every(m=>m.content.length<=2000));
    assert.ok(requests.at(-1).history.length<=5);
    return response;
}
(async()=>{
    let response=await ask('忘记密码怎么重置');
    for(let i=0;i<12;i++){
        response=await ask(['然后呢','还有呢','接着呢'][i%3]);
        assert.equal(response.sources[0]?.id,'ACC-13');
    }
    response=await ask('订单AB567891现在什么状态');
    for(let i=0;i<12;i++){
        response=await ask(['然后呢','还有呢','状态呢'][i%3]);
        assert.equal(response.mode,'insufficient_evidence');
        assert.match(response.answer,/实时订单/);
        assert.equal(response.sources.length,0);
    }
    response=await ask('忘记密码怎么重置');
    response=await ask('然后呢');
    assert.equal(response.sources[0]?.id,'ACC-13');
    elements.get('#clear').onclick();
    response=await ask('还有呢');
    assert.equal(requests.at(-1).history.length,0);
    assert.equal(response.mode,'insufficient_evidence');
})().catch(error=>{console.error(error);process.exitCode=1});
"""
        completed=subprocess.run(['node','-e',script,str(engine.ROOT/'static/index.html'),self.base],
                                 capture_output=True,text=True,timeout=30)
        self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
