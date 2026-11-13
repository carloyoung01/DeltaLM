"""Domain adapters, outside DeltaLM's domain-independent core.
No image/text/audio/video label is supplied to the kernel. These adapters are
simple predetermined feature extractors; they do NOT demonstrate raw perception.
"""
from __future__ import annotations
import hashlib
import numpy as np
from delta_core import DeltaKernel, Substrate


def _resample(v: np.ndarray, n: int=16) -> np.ndarray:
    v=np.asarray(v,dtype=float).ravel()
    if not len(v):return np.zeros(n)
    return np.interp(np.linspace(0,len(v)-1,n), np.arange(len(v)),v)


def image_features(image: np.ndarray) -> np.ndarray:
    a=np.asarray(image,dtype=float)
    if a.max()>1:a=a/(a.max() or 1)
    # Marginal row and column intensities; deliberately weak spatial adapter.
    return _resample(np.r_[np.mean(a,axis=0),np.mean(a,axis=1)],16)


def text_features(message: str) -> np.ndarray:
    v=np.zeros(16)
    for token in message.lower().replace(',',' ').replace('.',' ').split():
        k=int(hashlib.blake2b(token.encode(),digest_size=4).hexdigest(),16)%16
        v[k]+=1.0
    return np.sqrt(v/(max(v.sum(),1)))


def audio_features(audio: np.ndarray, sample_rate: int=8000) -> np.ndarray:
    x=np.asarray(audio,dtype=float)
    # Four temporal windows × four spectrum bands. Preserves some time order.
    v=[]
    for frame in np.array_split(x,4):
        z=abs(np.fft.rfft(frame))**2
        freq=np.fft.rfftfreq(len(frame),1/sample_rate)
        bins=[(0,300),(300,600),(600,1100),(1100,4000)]
        v.extend([float(z[(freq>=lo)&(freq<hi)].sum()) for lo,hi in bins])
    v=np.asarray(v)
    v=np.log1p(v)
    return v/(v.max()+1e-9)


def video_features(frames: np.ndarray) -> np.ndarray:
    a=np.asarray(frames,dtype=float)
    y,x=np.mgrid[:a.shape[-2],:a.shape[-1]]
    pts=[]
    for fr in a:
        mass=fr.sum()+1e-9
        pts.extend([float((fr*x).sum()/mass/(a.shape[-1]-1)),
                    float((fr*y).sum()/mass/(a.shape[-2]-1))])
    return _resample(np.asarray(pts),16)


def encode_input(features: np.ndarray, enable_rewrites: bool=True, steps: int=7) -> dict:
    """A separate current state for each independent specimen, for diagnostics.

    Repeated user sessions instead retain the same DeltaKernel state.
    """
    x=_resample(features,16)
    x=np.clip(x,0,1)
    m=DeltaKernel(Substrate(enable_rewrites=enable_rewrites,max_relations=56,
                            max_contacts=160))
    ids=[m.add_relation(f'n{i}',f'n{i+1}',0.18+0.76*float(v),
                        pace=0.10+0.012*i) for i,v in enumerate(x)]
    m.step(steps)
    s=m.state
    # Feature vector: direct expressions plus nonlinearly produced relations,
    # local contact conditions, and evolving observation coordinates.
    primary=np.array([s.relations[rid].expression() for rid in ids])
    by_start=np.zeros(16)
    for r in s.relations.values():
        if r.origin is not None:
            k=int(r.source[1:])
            if k<16:by_start[k]+=r.expression()
    by_start=np.tanh(by_start)
    contacts=np.zeros(16)
    for c in s.contacts.values():
        r=s.relations.get(c.left)
        if r is not None:
            k=int(r.source[1:])
            if k<16: contacts[k]+=c.coupling
    coords=np.array([s.lenses[f'n{i}'][0] for i in range(16)])
    vector=np.r_[primary,by_start,contacts,coords]
    return {'vector':vector, 'original':x, 'born':s.births,
            'operators':len(s.generated_programs),
            'contacts':len(s.contacts), 'n_relations':len(s.relations)}


def make_audio(kind: int, variant: int, rng: np.random.Generator, fs=8000):
    t=np.arange(1600)/fs
    if kind==0: y=np.sin(2*np.pi*(210+variant*2)*t)
    elif kind==1: y=np.sin(2*np.pi*(480+variant*3)*t)
    elif kind==2: y=np.sin(2*np.pi*(870+variant*2)*t)
    elif kind==3: y=np.sin(2*np.pi*(160*t+650*t*t))
    elif kind==4: y=np.sin(2*np.pi*(800*t-650*t*t))
    else: y=.5*np.sin(2*np.pi*210*t)+.5*np.sin(2*np.pi*870*t)
    return y+.09*rng.standard_normal(len(y))


def make_video(kind: int, variant: int):
    frames=np.zeros((12,24,24),dtype=float)
    for t in range(12):
        if kind==0:x,y=2+int(t*1.7), 6+variant%3
        elif kind==1:x,y=5+variant%3,2+int(t*1.7)
        elif kind==2:x,y=12+int(8*np.cos(t*2*np.pi/12)),12+int(8*np.sin(t*2*np.pi/12))
        else:x,y=11+variant%3,11+variant%3
        x=int(np.clip(x,1,22));y=int(np.clip(y,1,22))
        frames[t,y-1:y+2,x-1:x+2]=1.
    return frames


def text_corpus():
    categories=[
       ['boat','fishing','ocean','island','water','vessel','harbor','sea'],
       ['plane','airport','sky','flight','cloud','stars','wing','altitude'],
       ['chicken','rice','ayam','food','meal','market','spice','kitchen'],
       ['weather','rain','wind','storm','sun','clouds','climate','temperature']]
    samples=[]
    for i,words in enumerate(categories):
        for j in range(20):
            samples.append((' '.join([words[j%8],words[(j+2)%8],
                         words[(j+5)%8],words[(j+7)%8], 'near','today']),i))
    return samples
