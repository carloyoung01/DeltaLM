"""Build reproducible thumbnails/audio/video preview from the test inputs."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from PIL import Image
from sklearn.datasets import load_digits
from adapters import make_audio,make_video
OUT=Path(__file__).parent/'results'
OUT.mkdir(exist_ok=True)

digits=load_digits()
fig,axs=plt.subplots(2,4,figsize=(8,4))
for i,ax in enumerate(axs.flat):
    ax.imshow(digits.images[i*4],cmap='gray_r',vmin=0,vmax=16,interpolation='nearest')
    ax.set_title(f'Label: {digits.target[i*4]}',fontsize=10)
    ax.set_xticks([]);ax.set_yticks([])
fig.tight_layout();fig.savefig(OUT/'image_examples.png',dpi=150);plt.close(fig)
rng=np.random.default_rng(123)
audio=make_audio(3,0,rng)
write(OUT/'audio_example.wav',8000,(audio/(np.max(abs(audio))+1e-8)*32767).astype('int16'))
fig,ax=plt.subplots(figsize=(9,2.5));ax.plot(np.arange(300)/8000,audio[:300]);ax.set_xlabel('Time, s');ax.set_ylabel('Amplitude')
fig.tight_layout();fig.savefig(OUT/'audio_example.png',dpi=150);plt.close(fig)
frames=make_video(2,0)
seq=[Image.fromarray((frame*255).astype('uint8')).resize((240,240),resample=Image.Resampling.NEAREST) for frame in frames]
seq[0].save(OUT/'video_example.gif',save_all=True,append_images=seq[1:],duration=150,loop=0)
fig,axs=plt.subplots(1,6,figsize=(10,2));
for idx,ax in enumerate(axs):
 ax.imshow(frames[idx*2],cmap='gray',vmin=0,vmax=1);ax.axis('off');ax.set_title(f'frame {idx*2}')
fig.tight_layout();fig.savefig(OUT/'video_frames.png',dpi=150);plt.close(fig)
