"""Build a self-contained offline HTML replay and evidence charts from actual JSON traces.
Does not compute or modify DeltaLM results; display mapping is fixed and external.
Run: python showcase/build_observatory.py
"""
from pathlib import Path
import json, html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'showcase'/'results';A=ROOT/'showcase'/'assets';A.mkdir(exist_ok=True)

def load(n):return json.loads((R/n).read_text())
def trace(n):return load(n+'_trace.json')['frames']

def plot(filename,lists,keys,title,ylabel):
    fig,ax=plt.subplots(figsize=(9.5,4),layout='constrained')
    for n,fr in lists:
        for key in keys:
            ax.plot([f['t'] for f in fr],[f[key] for f in fr],label=f'{n}: {key}',linewidth=1.8)
    ax.set(xlabel='Kernel steps',ylabel=ylabel,title=title)
    ax.grid(alpha=.22);ax.legend(fontsize=8,loc='best',ncol=2)
    fig.savefig(A/filename,dpi=130);plt.close(fig)

plot('05_world_replay.png',[(f'world_{i}',trace(f'05_world_seed_{i}')) for i in [0,1,2]],['relations'],'05 · Open microcosm — retained relations','Relations')
plot('07_rewrite_intervention.png',[('normal',trace('07_normal_0')),('freeze',trace('07_frozen_0'))],['contacts'],'07 · Rewrites permitted vs frozen','Contacts')
plot('09_decay.png',[('passive',trace('09_passive'))],['activity'],'09 · Long quiet history (no forced novelty)','Sum of |relation expressions|')
plot('10_habitat_schedules.png',[(n,trace('10_'+n)) for n in ['stable','light_shift','reverse_shift']],['relations'],'10 · Toy ecology histories, not biology prediction','Relations')

compact=[]
names=[('04','04_cat_example_image','Cat photograph as numeric seed'),('04','04_synthetic_checker','Checkerboard numeric seed'),
('05','05_world_seed_0','Microcosm / history A'),('05','05_world_seed_1','Microcosm / history B'),('05','05_world_seed_2','Microcosm / history C'),
('06','06_game_actions','No-end game: interaction'),('06','06_game_no_actions','No-end game: no action'),
('07','07_normal_0','Generative operators: allowed'),('07','07_frozen_0','Generative operators: frozen'),
('08','08_A_only','Only A'),('08','08_B_only','Only B'),('08','08_A_then_B','A then B'),('08','08_B_then_A','B then A'),
('09','09_passive','Long quiet history'),('09','09_recontact','Recontact history'),
('10','10_stable','Toy ecology: stable'),('10','10_light_shift','Toy ecology: changed light'),('10','10_reverse_shift','Toy ecology: reversed condition')]
for experiment,n,name in names:
    src=load(n+'_trace.json')
    compact.append({'id':n,'experiment':experiment,'title':name,
       'frames':[{k:f[k] for k in ['t','relations','contacts','births','deaths','operators','rewrites','derived','generation','activity','new_relations','removed_relations','new_contacts','removed_contacts','program_changes']} for f in src['frames']],
       'external_events':[e for e in src['events'] if e['kind']=='external_stimulation']})

payload=json.dumps(compact,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeltaLM · Open Evolution Observatory</title>
<style>
:root{color-scheme:dark;--bg:#07151d;--panel:#0c2330;--ink:#eff9fb;--muted:#9eb4bb;--line:#254453;--a:#56d6bd;--b:#e4a86a}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 90% 0%,#14414b 0%,var(--bg) 45%);color:var(--ink);font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:38px 22px 70px}header p{max-width:850px;color:var(--muted)}h1{font-size:clamp(29px,4vw,48px);line-height:1.12;letter-spacing:-.035em;margin:13px 0}h2{font-size:20px}h3{font-size:15px;margin:0}.eyebrow{letter-spacing:.22em;color:var(--a);font-size:11px;font-weight:700}.panel{background:linear-gradient(160deg,#12303c,#0b1d29);border:1px solid var(--line);border-radius:18px;padding:23px;margin:18px 0}.flex{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.grid{display:grid;grid-template-columns:2fr 1fr;gap:18px}.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.metric{padding:14px;background:#0a1c28;border:1px solid var(--line);border-radius:12px}.metric small{display:block;color:var(--muted)}.metric strong{font:700 26px/1.4 ui-monospace,monospace}.pill{display:inline-block;border:1px solid #2d5f64;color:#b7ddd8;padding:4px 10px;border-radius:20px;font-size:11px;margin-right:5px;margin-top:6px}
select,button{background:#173743;border:1px solid #466c72;color:var(--ink);padding:9px 12px;border-radius:9px;font:inherit;cursor:pointer}button:hover{border-color:var(--a)}input[type=range]{flex:1;min-width:140px;accent-color:var(--a)}.canvas{background:#091924;border-radius:11px;width:100%;height:270px}.small{color:var(--muted);font-size:12px}.note{border-left:3px solid var(--b);padding-left:13px;color:#e8d1b3;margin:16px 0}.record{padding:5px 0;border-bottom:1px solid #284554;font-size:12px}a{color:#9cebd7}img{max-width:100%;border-radius:11px;border:1px solid var(--line)}@media(max-width:750px){.grid{grid-template-columns:1fr}.panel{padding:16px}}
</style><main><header><div class="eyebrow">DeltaLM / OPEN EVOLUTION SHOWCASE</div><h1>Let histories change<br>what may happen next.</h1><p>Ten evidence-led demonstrations, not a general language model benchmark. Play back real DeltaLM kernel traces and compare interventions. A quiet world, a weak effect, or no new event is also a recorded outcome. The charts and scene mappings are external observers, not the model creating graphics.</p>
<div><span class="pill">07 · Scientific core</span><span class="pill">05 / 06 · Visual entry points</span><span class="pill">04 / 10 · Open questions</span><span class="pill">No novelty reward</span></div></header>
<section class="panel"><h2>Observed histories</h2><div class="flex"><label for="scenario">Run</label><select id="scenario"></select><button id="play">▶ Play</button><button id="back">← Step</button><button id="next">Step →</button></div>
<div class="grid" style="margin-top:17px"><div><canvas id="chart" class="canvas" width="850" height="365"></canvas><div class="flex" style="margin:10px 0"><input type="range" id="slider" min="0" value="0"><span id="timestamp" style="font-family:monospace">t=0</span></div><p class="small">Green: executable contacts · amber: live relations · purple: distinct programs written (cumulative). Stored traces are sampled, while interventions were applied at exact kernel steps.</p></div>
<div><div class="metrics"><div class="metric"><small>Relations</small><strong id="rels">—</strong></div><div class="metric"><small>Contacts</small><strong id="contacts">—</strong></div><div class="metric"><small>New relations</small><strong id="births">—</strong></div><div class="metric"><small>Removed relations</small><strong id="deaths">—</strong></div><div class="metric"><small>Program trees</small><strong id="ops">—</strong></div><div class="metric"><small>Activity</small><strong id="activity">—</strong></div></div><h3 style="margin-top:15px">This sampled instant</h3><div id="events" class="small"></div></div></div></section>
<section class="panel"><h2>Evidence, not a spectacle requirement</h2><p>07 compares the same full snapshot with future rewriting permitted or frozen. The intervention is an experimenter action; it is <b>not</b> the kernel choosing to rewrite its own primitive grammar. Contacts can change and derived relations may later participate in new contacts, but the set of basic mathematical operators remains fixed.</p><div class="note">High visual complexity does not prove a new effective generative law. Noncommutative histories do not alone prove emergence. We report small, null and negative results; no visual scene is silently invented to match a desired ending.</div><h2>Evidence plots</h2><div class="grid"><div><img src="assets/07_rewrite_intervention.png" alt="actual 07 counterfactual chart"><p class="small">07 · Changing generative conditions</p></div><div><img src="assets/09_decay.png" alt="actual quiet history chart"><p class="small">09 · Persistence and dissolution</p></div><div><img src="assets/05_world_replay.png" alt="actual microcosm chart"><p class="small">05 · Microcosm, multiple histories</p></div><div><img src="assets/10_habitat_schedules.png" alt="actual toy ecology chart"><p class="small">10 · Synthetic conditions, no real biological prediction</p></div></div></section>
<section class="panel"><h2>What 01–03 actually measure</h2><p>Text, digit image, and synthetic sound tests compare handcrafted adapter features to DeltaLM's evolved numerical representations, each read with the <b>same external nearest-neighbor classifier</b>. They do not measure standalone language, image or audio generation. See <code>results/01_03_references.json</code> and <code>run_all.py</code>.</p><h2>04 / 10 · The unknown stays open</h2><p>A scikit-image example cat photo and a synthetic checkerboard provide numerical starting structures to 04. No "new animal" was generated. The ecology-themed schedules in 10 are synthetic: no heredity, fitness or real observed population series was implemented. Its next-contact suggestion is an <b>external observer</b> conditioned on current model structure, not autonomous scientific planning.</p><h2>06 · No-end world</h2><p>The scripted action sequence is a reproducible interaction example. For a live local command-driven world, use <code>python showcase/play_world.py</code>. Neither has a winning condition.</p></section><footer class="small">Code, inputs, raw snapshots, author-declared schedules, and SHA256 manifest are shipped with this research showcase. Original core copyright Carlo Young; Apache-2.0 applies to original source files. Exact supported claims and limits: <code>showcase/RESULTS_ZH.md</code>.</footer></main>
<script id="data" type="application/json">__PAYLOAD__</script><script>
const sets=JSON.parse(document.querySelector('#data').textContent),$=s=>document.getElementById(s);let sid=0,k=0,playing=false;sets.forEach((a,i)=>{let opt=document.createElement('option');opt.value=i;opt.textContent=a.experiment+' · '+a.title;$('scenario').append(opt)});
function draw(){const d=sets[sid],f=d.frames[k],c=$('chart'),ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='#091924';ctx.fillRect(0,0,w,h);const ky=['contacts','relations','operators'],cols=['#56d6bd','#e4a86a','#bb9de6'],max=Math.max(1,...d.frames.flatMap(q=>ky.map(t=>q[t])));for(let t=0;t<5;t++){let y=22+t*(h-52)/4;ctx.strokeStyle='#264150';ctx.beginPath();ctx.moveTo(42,y);ctx.lineTo(w-12,y);ctx.stroke();ctx.fillStyle='#9aadb9';ctx.font='12px sans-serif';ctx.fillText(String(Math.round(max*(1-t/4))),6,y+3)}ky.forEach((key,j)=>{ctx.beginPath();ctx.strokeStyle=cols[j];ctx.lineWidth=2.3;d.frames.forEach((q,i)=>{let x=42+(w-58)*i/Math.max(1,d.frames.length-1),y=h-30-(h-55)*q[key]/max;if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke()});let xpos=42+(w-58)*k/Math.max(1,d.frames.length-1);ctx.strokeStyle='#f3f7fa90';ctx.beginPath();ctx.moveTo(xpos,15);ctx.lineTo(xpos,h-20);ctx.stroke();
$('timestamp').textContent='t='+f.t;$('slider').value=k;for(let [id,key] of [['rels','relations'],['contacts','contacts'],['births','births'],['deaths','deaths'],['ops','operators']])$(id).textContent=f[key];$('activity').textContent=f.activity.toFixed(3);let e=[];if(f.new_relations)e.push('New relation(s): '+f.new_relations);if(f.removed_relations)e.push('Removed relation(s): '+f.removed_relations);if(f.program_changes)e.push('Changed contact programs: '+f.program_changes);if(!e.length)e.push('No structural birth/removal in this sample; internal values may still vary.');$('events').innerHTML=e.map(t=>'<div class="record">'+t+'</div>').join('')}
function state(i){sid=i;k=0;$('slider').max=sets[sid].frames.length-1;draw()}$('scenario').onchange=e=>{playing=false;state(+e.target.value)};$('slider').oninput=e=>{playing=false;k=+e.target.value;draw()};$('next').onclick=()=>{playing=false;k=Math.min(k+1,sets[sid].frames.length-1);draw()};$('back').onclick=()=>{playing=false;k=Math.max(0,k-1);draw()};$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'❚❚ Pause':'▶ Play'};setInterval(()=>{if(playing){k=(k+1)%sets[sid].frames.length;draw()}},85);state(0);
</script></html>'''.replace('__PAYLOAD__',payload)
(ROOT/'showcase'/'observatory.html').write_text(page,encoding='utf8')
print('Created offline observatory',round(len(page)/1024),'KiB')