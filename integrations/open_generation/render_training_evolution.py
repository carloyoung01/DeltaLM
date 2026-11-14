"""Deterministic C/D output from real DeltaLM state trajectories.
Files are programmatic renderings; they are NOT generative GPT audio/video.
Requires numpy Pillow ffmpeg. Run from repository root.
"""
from __future__ import annotations
import sys,json,math,wave,subprocess,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from delta_core import DeltaKernel,Substrate
from ABCD_protocol import NODES,EDGES,EVENTS

HERE=Path(__file__).resolve().parent
OUT=HERE/'media_calibration'

def make_world(values=None):
    b=Substrate(max_relations=72,max_contacts=210)
    if values:
        b.activity_retention=float(values['activity_retention'])
        b.contact_retention=float(values['contact_retention'])
    rng=np.random.default_rng(10)
    m=DeltaKernel(b)
    for i,j in EDGES:
        m.add_relation(NODES[i],NODES[j],float(.35+.3*rng.random()),float(.045+.10*rng.random()))
    return m

def summary(m):
    s=m.state
    return {'step':s.clock,'relations':len(s.relations),'births':s.births,
            'activity':round(sum(abs(r.expression()) for r in s.relations.values()),7)}

def render(m,label):
    W,H=470,286
    im=Image.new('RGB',(W,H),(10,22,35))
    d=ImageDraw.Draw(im)
    pos={name:(170+108*math.cos(i*2*math.pi/8),139+108*math.sin(i*2*math.pi/8))
         for i,name in enumerate(NODES)}
    for r in m.state.relations.values():
        if r.source not in pos or r.target not in pos:continue
        c=min(abs(r.expression()),1.)
        color=(int(40+90*c),int(88+125*c),int(120+105*c))
        d.line([pos[r.source],pos[r.target]],fill=color,width=1)
    for x,y in pos.values():
        d.ellipse((x-5,y-5,x+5,y+5),fill=(245,183,85))
    s=summary(m)
    d.text((297,31),label,fill='white')
    d.text((297,73),f"step {s['step']}",fill=(173,220,234))
    d.text((297,100),f"links {s['relations']}",fill=(173,220,234))
    d.text((297,127),f"births {s['births']}",fill=(173,220,234))
    d.text((297,154),f"active {s['activity']:.4f}",fill=(173,220,234))
    d.text((22,270),'CORE STATE RENDER -- NOT GENERATIVE VIDEO',fill=(137,160,179))
    return im

def save_wav(events,path):
    rate=16000;length=8*rate;times=np.arange(length)/rate
    indices=np.linspace(0,len(events)-1,length)
    a=np.interp(indices,np.arange(len(events)),[e['activity'] for e in events])
    n=np.interp(indices,np.arange(len(events)),[e['relations'] for e in events])
    phase=np.cumsum(2*np.pi*(150+8*n)/rate)
    signal=.3*np.minimum(a/7.,1.)*np.sin(phase)*np.minimum(1.,times*3)*np.minimum(1.,(8-times)*3)
    with wave.open(str(path),'wb') as wf:
        wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(rate)
        wf.writeframes((np.clip(signal,-1,1)*32767).astype('<i2').tobytes())

def run():
    OUT.mkdir(exist_ok=True)
    data=json.loads((HERE/'results'/'core_calibration_2026-11-14.json').read_text())
    models={'C_default':make_world(),'D_trained':make_world(data['learned_parameters'])}
    frames=[];history={k:[] for k in models}
    for t in range(321):
        if t%8==0:
            views=[]
            for name,m in models.items():
                views.append(render(m,name));history[name].append(summary(m))
            im=Image.new('RGB',(940,286))
            im.paste(views[0],(0,0));im.paste(views[1],(470,0))
            frames.append(im)
        if t==320:break
        if t+1 in EVENTS:
            rid,imp=EVENTS[t+1]
            for m in models.values():m.stimulate('r'+str(rid),imp)
        for m in models.values():m.step()
    frames[0].save(OUT/'initial_A.png')
    frames[-1].save(OUT/'trained_comparison_CD.png')
    for name in models:
        save_wav(history[name],OUT/(name+'.wav'))
    tmp=HERE/'_temporary_frames';tmp.mkdir(exist_ok=True)
    for i,frame in enumerate(frames):frame.save(tmp/f'{i:03d}.png')
    dest=OUT/'evolution_CD.mp4'
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    proc=subprocess.run([ffmpeg,'-loglevel','error','-y','-framerate','5',
         '-i',str(tmp/'%03d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(dest)],
         capture_output=True,text=True)
    for f in tmp.glob('*.png'):f.unlink()
    tmp.rmdir()
    if proc.returncode:raise RuntimeError(proc.stderr)
    result={'status':'PROGRAMMATIC_CORE_TRACE_RENDER_NOT_GPT','frames':len(frames),
      'C_end':history['C_default'][-1],'D_end':history['D_trained'][-1],
      'files':sorted(f.name for f in OUT.iterdir())}
    (HERE/'results'/'trained_media_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
