# DeltaLM — History-Dependent Structural Co-Evolution

**Research preview · v37 codebase · proposed public release `v0.1.0-research`**

DeltaLM is an experimental computational framework for the co-evolution of locally acting relations. It studies how historical state, local constraints, and executable operator structures affect later interactions and reachable structures. It is **not a pretrained large language model** and has no model weights or demonstrated general-purpose text/audio/image/video understanding.

The name `DeltaLM` is retained as the research project's name; `LM` does **not** mean that this implementation is necessarily a language model or a large-parameter model.

## What is implemented

- Independent relation objects with fast activity, slower traces, and history-dependent local coupling.
- A bounded expression algebra whose composite executable operator trees can be rewritten during interactions, inherited by derived relations, and evaluated in subsequent steps.
- Relations can emerge and disappear; no single reward or target output drives the core transitions.
- JSON snapshots and a local command-line interface for continuing a session.
- Domain adapters for simple numeric features extracted from image, text, audio, and video data. These adapters are separate from the core.
- Research experiments including counterfactual interventions, operator-rewrite ablations, history-order comparisons, and bounded-duration numerical runs.

## Install

Python 3.10+ is intended; the original research run and the environment required for exact reproduction must be independently verified before publishing. Dependency ranges are listed in `requirements.txt` and are not a lockfile.

```bash
python -m pip install -r requirements.txt
python experiments.py --quick
```

Minimal core usage (standard library only):

```python
from delta_core import DeltaKernel

model = DeltaKernel()
a = model.add_relation('A', 'B', 0.7)
b = model.add_relation('B', 'C', 0.6)
model.encounter(a, b)
model.step(30)
print(model.observe())
```

To use the local snapshot CLI:

```bash
python stream_cli.py --state history.json new
python stream_cli.py --state history.json add A B
python stream_cli.py --state history.json add B C
python stream_cli.py --state history.json step 25
python stream_cli.py --state history.json observe
```

## Research evidence and limitations

The v37 experiment report describes observed operator-tree diversity, an intervention changing subsequent dynamics, long-running finite numerical behavior, and four limited modality-adapter diagnostics. **Operator recombination uses a fixed primitive grammar and is not evidence of discovery of arbitrary new mathematical primitives.** Long-running bounded simulations are not proof of global stability, real-world utility, energy savings, or artificial general intelligence. Modality results do not establish end-to-end semantic understanding. See [experiment report](TEST_REPORT_ZH.md) and [mathematics/code walk-through](MATHEMATICS_AND_CODE_ZH.md).

The numerical results are provided in [`results/results.json`](results/results.json), [`results/long_run.csv`](results/long_run.csv) and [`results/modalities.csv`](results/modalities.csv). The PNG/GIF/WAV visual and audio samples are not included in the repository's initial file upload; generate them with `python experiments.py` and `python create_samples.py`. The complete reviewed archive is available separately from the author.

## Reproducibility

```bash
python experiments.py                # complete experiment suite
python create_samples.py            # regenerate visual/audio sample files
```

The accompanying results are from the original research run. A separate release-review rerun of `python experiments.py` reproduced the structured results except wall-clock timing fields; this was run in an existing prepared environment, **not** in a newly provisioned clean environment. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Scope and safety

This is research software, not an autonomous system authorized to act on the physical world. `Contact.coupling` denotes local internal coupling, not user authorization. External actuation, privacy, safety policy and permission checks belong to deployment-specific interfaces.

## Documentation

- [Chinese mathematical explanation and code guide](MATHEMATICS_AND_CODE_ZH.md)
- [Chinese test report and negative findings](TEST_REPORT_ZH.md)
- [Earlier v36 model](v36_original.py)

## License, author and citation

**Author:** Carlo Young (GitHub: `carloyoung01`).

**Copyright:** © 2026 Carlo Young.

**License:** [Apache License 2.0](LICENSE), including commercial and company use under its terms. Third-party dependencies and datasets retain their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md). No DOI, paper acceptance, or peer review is claimed.