# DeltaLM × Generative Interfaces — Executable Research Addition

**2026-11-14 · Author: Carlo Young · Apache-2.0 · Research demonstration**

**Preservation:** Does not replace, rewrite, or modify the ten `showcase/` experiments or `delta_core.py`. This is an additional experimental layer, not an upgraded commercial LLM.

## What is actually implemented

1. **Actual DeltaLM runtime traces** from 8 researcher-named nodes and 12 starting relations; 320 steps in three input histories, including a no-extra-stimulus condition and a zero-state 400-step null.
2. **Text exemplar A/B/C**, authored by the GPT assistant and saved in `TEXT_EXAMPLES_ZH.md`, with a trace-grounded C and clear identification of metaphors. This is not controlled GPT API inference.
3. **Visual image A/C**, produced by a deterministic mapping from the original and evolved DeltaLM states; **not neural image generation**.
4. **Audio A/C**, original WAV sound synthesis mapping current activity, relation count and coupling into sonic parameters; **not GPT speech/music synthesis**.
5. **Video A/C**, MP4 replay of the unchanged initial state and the actual trace; **not generative video inference**.
6. **Trainable external numeric interface**, Ridge regression trained on 24 independent simulation worlds, evaluated on eight held-out worlds; **not training DeltaLM core or GPT weights**. Includes negative results.
7. **Three counterfactual forks** freezing further operation-tree rewrites from a common snapshot; the effect on numbers is reported without claiming invention of new mathematical primitives.

## Start

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install Pillow scikit-learn
python integrations/open_generation/integrated_research.py
python integrations/open_generation/render_media.py   # requires ffmpeg
python -m unittest integrations/open_generation/test_integrity.py -v
```

Open `integrations/open_generation/observatory.html` to inspect ready-made local examples. It is an **offline replay**, not a live GPT connection. For a smaller rerun the scripts use the root repository `delta_core.py` and write under `integrations/open_generation/results/` and `media/`.

## A/B/C/D research protocol

- **A**: initial GPT condition or static external-media rendering.
- **B**: GPT free continuation; provided for **text only** in this evidence package.
- **C**: actual DeltaLM trace guides the GPT text exemplar, or provides image/audio/video deterministic readout controls.
- **D**: trained *semantic/generative* interface. **Not completed**. A separate numeric Ridge probe was trained, but it is not interchangeable with D.

The package is a **partial prototype**, not a completed four-modality generative-AI API compatibility or joint-training result. This limitation is intentional and explicit; the media samples themselves are genuine files, but they are not outputs of a trained multimodal GPT endpoint.

### Grounding caveat

The researcher's labels (tide, stone, light, etc.), color/shape/sound mappings, and all metaphorical prose come from **external interfaces**, not from DeltaLM's own understanding. No ungrounded animal, biology, physical world law, general intelligence, or energy-saving capability is claimed.

### Results

See [Results & limits (Chinese)](RESULTS_ZH.md), [the three text branches](TEXT_EXAMPLES_ZH.md), [mapping and validation](METHODS.md), and `results/*.json`. New training with parameters and a holdout split is an **optional external layer**, not a mandatory event archive. Core state remains current relations / conditions / programs, not a growing log of user history.
