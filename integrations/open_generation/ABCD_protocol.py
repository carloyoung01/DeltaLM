"""Real DeltaLM A/B/C/D prompt protocol for text, image, audio, and video.
C/D are based on actual core runs. No inference is claimed without API records.
"""
from __future__ import annotations
import sys,json,argparse,os,base64,hashlib
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from delta_core import DeltaKernel,Substrate

HERE=Path(__file__).resolve().parent
NODES=('tide','stone','light','wind','shore','seed','shadow','echo')
EDGES=((0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),(0,3),(3,6),(6,1),(1,4))
INITIAL='A quiet shore, wind, stones, seeds, and an echo; no ending is prescribed.'
EVENTS={35:(0,.24),82:(4,-.18),133:(7,.27),210:(2,-.12)}
MEDIA=('text','image','audio','video')
BRANCHES=('A','B','C','D')

def trace(params=None):
    rng=np.random.default_rng(10)
    b=Substrate(max_relations=72,max_contacts=210)
    if params:
        b.activity_retention=float(params['activity_retention'])
        b.contact_retention=float(params['contact_retention'])
    m=DeltaKernel(b)
    for a,z in EDGES:
        m.add_relation(NODES[a],NODES[z],float(.35+.3*rng.random()),float(.045+.10*rng.random()))
    out=[]
    for t in range(321):
        if t in (0,35,80,135,160,210,270,320):
            s=m.state
            out.append({'step':t,'relations':len(s.relations),'contacts':len(s.contacts),
                'births':s.births,'deaths':s.deaths,
                'programs_created_diagnostic':len(s.generated_programs),
                'activity_abs':round(sum(abs(r.expression()) for r in s.relations.values()),8),
                'mean_contact_abs':round(sum(abs(c.coupling) for c in s.contacts.values())/max(1,len(s.contacts)),8)})
        if t==320:break
        if t+1 in EVENTS:
            ident,imp=EVENTS[t+1];m.stimulate('r'+str(ident),imp)
        m.step()
    return out

def generate_protocol():
    p=HERE/'results'/'core_calibration_2026-11-14.json'
    trained=json.loads(p.read_text())['learned_parameters']
    c=trace();d=trace(trained)
    result={'initial_prompt':INITIAL,'initial_nodes':8,'initial_relations':12,
       'initial_labels_are_researcher_supplied':True,
       'C_untrained_actual_core_trace':c,
       'D_trained_substrate_actual_core_trace':d,
       'GPT_multimodal_API_calls_completed':False,
       'video_OpenAI_API_discontinued_2026_09_24':True,
       'branches':{}}
    for medium in MEDIA:
        result['branches'][medium]={}
        for branch in BRANCHES:
            text='Create a '+medium+' interpretation. Common initial premise: '+INITIAL
            if branch=='A':text+='\nGenerate directly; no DeltaLM trace.'
            if branch=='B':text+='\nFree continuation with equal generation opportunities; no DeltaLM trace.'
            if branch in ('C','D'):
                t=c if branch=='C' else d
                text+='\nObserved DeltaLM numerical state history: '+json.dumps(t,separators=(',',':'))
                text+='\nThese statistics are not learned scene semantics. Respect nonchange, fading or persistence. Clearly distinguish external creative interpretation.'
                if branch=='D':text+='\nD uses synthetically calibrated core coefficients. GPT weights and semantic decoders were NOT trained.'
            result['branches'][medium][branch]={
                'prompt':text,'api_execution':'NOT_RUN',
                'uses_real_DeltaLM_trace':branch in ('C','D'),
                'trained_core_coefficients':branch=='D',
                'GPT_weights_trained':False}
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--medium',choices=MEDIA,default='text')
    ap.add_argument('--branch',choices=BRANCHES,default='C')
    ap.add_argument('--execute',action='store_true',help='Execute optional billable OpenAI API call')
    ap.add_argument('--text-model',default='gpt-6')
    ap.add_argument('--image-model',default='gpt-image-2.5-flare')
    ap.add_argument('--audio-model',default='gpt-4o-mini-tts')
    args=ap.parse_args()
    allp=generate_protocol()
    folder=HERE/'results';folder.mkdir(exist_ok=True)
    (folder/'ABCD_protocol_2026-11-14.json').write_text(json.dumps(allp,indent=2,ensure_ascii=False)+'\n')
    prompt=allp['branches'][args.medium][args.branch]['prompt']
    if not args.execute:
        print(json.dumps({'status':'dry_run_no_API_call','medium':args.medium,'branch':args.branch,'prompt':prompt},ensure_ascii=False))
        return
    if args.medium=='video':
        raise SystemExit('OpenAI Videos API shut down 2026-09-24; use deterministic trace renderers or a separately licensed provider.')
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY not configured. No API output was created.')
    from openai import OpenAI
    client=OpenAI()
    out=HERE/'api_outputs';out.mkdir(exist_ok=True)
    stem=args.medium+'_'+args.branch
    if args.medium=='text':
        model=args.text_model
        value=client.responses.create(model=model,input=prompt).output_text.encode()
        extension='.txt'
    elif args.medium=='image':
        model=args.image_model
        im=client.images.generate(model=model,prompt=prompt,size='1024x1024',n=1)
        value=base64.b64decode(im.data[0].b64_json)
        extension='.png'
    else:
        model=args.audio_model
        speech=client.audio.speech.create(model=model,voice='alloy',input=prompt[:3900],response_format='mp3')
        value=speech.content;extension='.mp3'
    (out/(stem+extension)).write_bytes(value)
    (out/(stem+'.meta.json')).write_text(json.dumps({
       'status':'completed_API_call','medium':args.medium,'branch':args.branch,
       'model':model,'sha256_prompt':hashlib.sha256(prompt.encode()).hexdigest(),
       'sha256_output':hashlib.sha256(value).hexdigest(),
       'AI_voice_disclosure_required':args.medium=='audio',
       'trained_GPT_weights':False},indent=2)+'\n')
    print('API result created: ',stem+extension)
if __name__=='__main__':main()
