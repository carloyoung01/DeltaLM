"""Deterministic image/audio/video readouts from actual DeltaLM trace.
Visuals and audio are EXTERNAL mappings, NOT raw neural-media generation.
The mapping is intentionally simple so claims remain falsifiable.
"""
from __future__ import annotations
import json, math, wave, shutil, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'media';OUT.mkdir(exist_ok=True)
DATA=json.loads((ROOT/'results'/'four_media_core_trace.json').read_text())
NAMES=['tide','stone','light','wind','shore','seed','shadow','echo']
NODES={n:(math.cos(-math.pi/2+2*math.pi*i/8),math.sin(-math.pi/2+2*math.pi*i/8)) for i,n in enumerate(NAMES)}
W,H=1200,676
fpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(fpath,18);big=ImageFont.truetype(fpath,26);small=ImageFont.truetype(fpath,15)

# This coordinate mapping is entirely external and pre-defined.
def nodepos(name,frame):
    if name not in NODES:return None
    x,y=NODES[name];lens=frame['node_positions'].get(name,[0.,0.])
    return (int(600+248*x+100*lens[0]),int(360+217*y+100*lens[1]))

def gradient():
    a=np.empty((H,W,3),dtype=np.uint8)
    for j in range(H):
        val=j/H
        a[j,:,0]=int(5+6*val);a[j,:,1]=int(17+8*val);a[j,:,2]=int(30+13*val)
    return Image.fromarray(a,'RGB').convert('RGBA')

def draw_scene(frame,kind='C',title='HISTORY-DEPENDENT WORLD'):
    im=gradient(); layer=Image.new('RGBA',(W,H)); d=ImageDraw.Draw(layer,'RGBA')
    for i in range(16):
        r=70+i*37
        d.ellipse((600-r,360-r,600+r,360+r),outline=(21,80,96,17),width=1)
    # Links represent actual relations; color-coded signed strength, faded when inactive.
    for link in frame['links']:
        a=nodepos(link['src'],frame);b=nodepos(link['dst'],frame)
        if not a or not b:continue
        value=link['value'];mag=abs(value)
        color=(98,224,223,min(210,max(18,int(80+mag*240)))) if value>=0 else (249,165,102,min(210,max(18,int(80+mag*240))))
        d.line((*a,*b),fill=color,width=max(1,min(5,int(1+mag*4))))
    for i,n in enumerate(NAMES):
        xy=nodepos(n,frame)
        if not xy:continue
        x,y=xy;d.ellipse((x-14,y-14,x+14,y+14), fill=(30,158,174,255),outline=(116,246,233,255),width=2)
        d.text((x+18,y-12),n,font=small,fill=(218,239,239,255))
    glow=layer.filter(ImageFilter.GaussianBlur(12));glow.putalpha(glow.getchannel('A').point(lambda a:int(a*.48)))
    im=Image.alpha_composite(im,glow);im=Image.alpha_composite(im,layer)
    dd=ImageDraw.Draw(im)
    dd.text((44,40),'DeltaLM × GENERATIVE INTERFACES',font=big,fill=(213,245,238))
    dd.text((44,82),title,font=font,fill=(132,191,203))
    dd.text((44,116),'Trace-derived structural visualization | external deterministic renderer',font=small,fill=(137,161,170))
    dd.text((44,579),f"STEP {frame['step']:03d}  •  RELATIONS {frame['relations']}  •  CONTACTS {frame['contacts']}  •  BIRTHS {frame['births']}",font=font,fill=(222,227,230))
    dd.text((44,617),f"ACTIVITY {frame['activity_abs']:.4f}  •  REWRITES {frame['contact_rewrites']}  •  {kind} not autonomous media generation",font=small,fill=(166,194,201))
    return im.convert('RGB')

if __name__=='__main__':
    frames=DATA['tide']['frames'];static=frames[0]
    draw_scene(static,'A','A · INITIAL INPUT / NO DELTALM HISTORY').save(OUT/'12_image_A_initial.png',optimize=True)
    draw_scene(frames[-1],'C','C · ACTUAL DELTALM EVOLVED STATE').save(OUT/'12_image_C_history.png',optimize=True)
    draw_scene(DATA['still']['frames'][-1],'C','C-NULL · NO EXTERNAL STIMULI').save(OUT/'12_image_C_no_stimuli.png',optimize=True)
    # C video: true structural replay, never presented as neural-generated video.
    cmd=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{W}x{H}','-r','8','-i','-','-c:v','libx264','-preset','veryfast',
         '-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'14_video_C_trace_replay.mp4')]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for f in frames:
        pic=draw_scene(f,'C','EVOLVING STATE · ACTUAL TRACES')
        p.stdin.write(pic.tobytes())
    p.stdin.close(); rc=p.wait()
    if rc:raise RuntimeError('ffmpeg exit status '+str(rc))
    staticvideo=str(OUT/'14_video_A_static.mp4')
    cmd2=['ffmpeg','-loglevel','error','-y','-loop','1','-framerate','8',
          '-i',str(OUT/'12_image_A_initial.png'),'-t',str(len(frames)/8),
          '-c:v','libx264','-preset','veryfast','-crf','23',
          '-pix_fmt','yuv420p','-movflags','+faststart',staticvideo]
    subprocess.run(cmd2,check=True)
    # Same trace maps to audio without a speech model or implicit semantics.
    sr=22050;duration=8;t=np.arange(sr*duration)/sr
    phase=2*math.pi*t
    a=.08*np.sin(220*phase)+.04*np.sin(330*phase)
    # Baseline is repeated audio pattern from initial conditions.
    for file,arr in [('13_audio_A_static.wav',a)]:
        z=np.int16(np.clip(arr,-.98,.98)*32767)
        with wave.open(str(OUT/file),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes(z.tobytes())
    signal=np.zeros_like(t)
    # Iterate fixed 8-second time bins with known trace; no stochastic flourish.
    bins=np.array_split(np.arange(t.size),len(frames))
    for i,ix in enumerate(bins):
        f=frames[i];abs_activity=f['activity_abs'];sig=np.clip(abs_activity/3.,0,.8)
        hz=180+15*(f['relations']%13)+70*math.tanh(f['mean_coupling']*8)
        local_t=t[ix]
        fade=np.sin(np.linspace(0,math.pi,len(ix)))**.45
        signal[ix]=(.075+.10*sig)*np.sin(2*math.pi*hz*local_t)*fade + (.025+.018*sig)*np.sin(2*math.pi*(hz*1.5)*local_t)*fade
    z=np.int16(np.clip(signal,-.98,.98)*32767)
    with wave.open(str(OUT/'13_audio_C_trace_sonification.wav'),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes(z.tobytes())
    print('Media files:')
    for path in sorted(OUT.iterdir()):print(path.name,path.stat().st_size)
