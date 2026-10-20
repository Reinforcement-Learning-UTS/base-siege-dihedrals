"""Run the shipped renderer to check idle work and interactive wake-up paths."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from agents.heuristic import HeuristicCommander
from experiments.rollout import play_episode
from env.siege_env import SiegeEnv
from tests.test_browser_manual_parity import manual_payload
from ui.manual_game import ManualGame

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which('node') is None, reason='Node runs the shipped renderer')
def test_idle_boards_sleep_and_controls_resume_animation():
    manual = ManualGame(base_seed=42, record=True, record_ticks=True)
    manual.reset(seed=0)
    payload = {**manual_payload(manual), 'readonly': False, 'game_key': 'test', 'log': []}
    replay = play_episode(SiegeEnv(base_seed=42, record=True, record_ticks=True), HeuristicCommander())
    recorded = {**payload, 'readonly': True,
                'replay': {'frames': replay['visual_frames'], 'id': 'recorded', 'label': 'Test'}}
    program = r"""
const fs=require('fs'), vm=require('vm'), assert=require('assert/strict');
const data=JSON.parse(fs.readFileSync(0,'utf8')), nodes=new Map(), listeners={}, queue=[];
let paints=0, now=0;
const ctx=new Proxy({}, {get:(o,k)=>k==='clearRect'?()=>paints++:o[k]||(()=>{}),set:(o,k,v)=>(o[k]=v,true)});
function element(){return {style:{},attrs:{},children:[],value:'',checked:false,
  classList:{toggle(){}},append(...items){this.children.push(...items)},
  replaceChildren(...items){this.children=items},setAttribute(k,v){this.attrs[k]=v},
  getBoundingClientRect(){return {left:0,top:0,width:544,height:780}},getContext(){return ctx}};}
const get=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id)};
Object.assign(get('cv'),{width:544,height:544});get('speed').value='1';
get('stepmode').checked=true;get('tile').value='7';
const parent={postMessage(){}}, document={hidden:false,body:element(),getElementById:get,
  querySelector:get,createElement:element,addEventListener:(k,v)=>listeners[k]=v};
const sandbox={document,parent,window:{addEventListener:(k,v)=>listeners[k]=v},
  matchMedia:()=>({matches:false}),ResizeObserver:class {observe(){}},
  requestAnimationFrame:fn=>queue.push(fn),console};
vm.createContext(sandbox);
for(const name of ['glyphs.js','manual_engine.js','app.js'])
  vm.runInContext(fs.readFileSync(process.argv[1]+'/'+name,'utf8'),sandbox,{filename:name});
function step(){if(queue.length){now+=1000/60;queue.shift()(now)}}
function settle(){let n=0;while(queue.length&&n++<300)step();assert.equal(queue.length,0,'idle animation must stop');}
function render(payload){listeners.message({source:parent,data:{type:'streamlit:render',args:{payload}}});settle();}
render(data.manual);
assert.ok(paints<=2,'initial board should paint once, not continuously');
let count=paints;get('ranges').checked=true;get('ranges').onchange();settle();assert.equal(paints,count+1);
const buy=get('troops').children.find(x=>x.attrs['aria-label']==='Buy melee');
buy.onclick({shiftKey:false});settle();assert.equal(get('s-budget').textContent,192);
get('btn-main').onclick();get('btn-deploy').onclick({shiftKey:false});settle();
get('btn-main').onclick();settle();assert.equal(get('s-tick').textContent,290);
get('btn-main').onclick();settle();assert.ok(get('s-tick').textContent<290,'continue must restart combat');
render(data.replay);count=paints;settle();assert.equal(paints,count);
get('btn-play').onclick();for(let i=0;i<12;i++)step();assert.ok(Number(get('timeline').value)>0);
get('btn-play').onclick();settle();count=paints;settle();assert.equal(paints,count,'paused replay must sleep');
get('btn-step').onclick();settle();assert.ok(paints>count,'stepping must repaint');
get('timeline').value=1000;get('timeline').oninput();settle();assert.equal(get('playback-status').textContent,'Finished');
document.hidden=true;listeners.visibilitychange();settle();document.hidden=false;listeners.visibilitychange();settle();
assert.equal(get('playback-status').textContent,'Finished');
process.stdout.write(JSON.stringify({idle_pending_frames:queue.length,paint_count:paints}));
"""
    result = subprocess.run([shutil.which('node'), '-e', program, str(ROOT / 'ui/frontend')],
                            input=json.dumps({'manual': payload, 'replay': recorded}),
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)['idle_pending_frames'] == 0
