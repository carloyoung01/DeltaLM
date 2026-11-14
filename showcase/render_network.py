"""Read-only deterministic relation diagram; nodes and edges are from actual snapshots.
Positions are a fixed label layout and are NOT generated morphology.
"""
import json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
R=Path(__file__).resolve().parent

def view(prefix,t,output,title):
    d=json.loads((R/'results'/f'{prefix}_trace.json').read_text())
    data=d['snapshots'][str(t)];rels=data['relations']
    names=sorted(set([r['source'] for r in rels]+[r['target'] for r in rels]))
    pos={n:(math.cos(2*math.pi*i/len(names)),math.sin(2*math.pi*i/len(names))) for i,n in enumerate(names)}
    fig,ax=plt.subplots(figsize=(7.8,6.8));fig.patch.set_facecolor('#081923');ax.set_facecolor('#081923')
    for r in rels:
        x,y=pos[r['source']];X,Y=pos[r['target']]
        power=abs(r['activity']);isnew=r['origin'] is not None
        a=FancyArrowPatch((x,y),(X,Y),arrowstyle='-|>',mutation_scale=7,
            linewidth=.35+1.2*power,color='#EFB275' if isnew else '#63CDBB',
            alpha=.15+.55*max(power,.1),connectionstyle='arc3,rad=.15')
        ax.add_patch(a)
    for name,(x,y) in pos.items():
        ax.scatter([x],[y],s=140,c='#163B4A',edgecolors='#94DDD6',linewidths=1.2,zorder=5)
        ax.text(x,y-.14,str(name),color='#F1F6F6',ha='center',fontsize=9)
    ax.set(xlim=(-1.35,1.35),ylim=(-1.3,1.3));ax.set_aspect('equal');ax.axis('off')
    ax.set_title(title,color='white',fontsize=15,pad=16)
    fig.text(.08,.075,f'Actual internal relation graph at t={t}; green=initial, amber=derived. Layout is fixed; not a biological shape.',
             color='#97B6BE',fontsize=8)
    fig.savefig(R/'assets'/output,dpi=140,facecolor=fig.get_facecolor());plt.close(fig)

view('04_cat_example_image',150,'04_cat_structure.png','04 · Cat photo features → relational structure')
view('05_world_seed_0',270,'05_final_world.png','05 · A finite world after 270 steps')
view('06_game_actions',170,'06_game_history.png','06 · A world with user interventions')