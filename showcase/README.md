# DeltaLM — Open Evolution Showcase (v1.2 design, first evidence release)

**Author: Carlo Young** · **License: Apache-2.0** · **Core unchanged**

This is a research visualization and falsifiable experiment suite for the DeltaLM local structural co-evolution kernel. It is **not** an LLM, not an autonomous scientific discovery agent, and not a biological simulator. A system may generate relations, retain them, lose them, or remain quiet; none is treated as a forced success state.

## Run, verify, and view

From the repository root, with Python 3.11+ and the dependencies in the root `requirements.txt`:

```bash
python -m unittest showcase/test_integrity.py -v
python showcase/run_all.py
python showcase/build_observatory.py
python showcase/play_world.py
```

Open `showcase/observatory.html` in a browser to view **recorded** traces. The HTML doesn't run DeltaLM itself and its graphical mappings are external visualization rules. For a live, local, text-based world use `play_world.py` (commands: `look`, `step N`, `touch ID VALUE`, `save PATH`, `load PATH`, `quit`). There is no winning condition.

## The ten tests

| ID | Question | Observables | Qualification |
|---|---|---|---|
| 01 | Is text distinction retained? | Held-out external nearest-neighbor readout; word order collision | Not natural-language understanding |
| 02 | How is digit-image information handled? | Held-out digit classification from fixed image features | Not end-to-end vision |
| 03 | How are sounds distinguished? | Six synthetic tone categories; fixed spectral features | Not speech understanding |
| 04 | Can image-derived structure evolve without a prescribed shape? | Cat-example vs artificial checker seed histories | No generated animal morphology |
| 05 | Can a microcosm grow, persist, and dissipate? | 3 initial histories; relations, contacts, births and deaths | No simulated physics |
| 06 | Can a world continue without an ending? | A scripted interaction vs an untouched world; live CLI | No spontaneous game rule invention |
| 07 | Do historical structures affect available downstream operations? | 3 matched forks; rewrite-on versus rewrite-frozen | **Scientific center**; fixed primitive grammar |
| 08 | Does shared history differ from separate or reordered actions? | A, B, A→B, B→A, simultaneous, no input | Nonlinearity ≠ emergent law |
| 09 | Can the world quieten, dissipate and re-contact? | 1,600 passive steps, snapshots and new stimulus | UI pause ≠ autonomous stop |
| 10 | How far can evolutionary exploration be interpreted? | Synthetic habitat histories and an externally selected contact | **Not empirical biology validation** |

All actual inputs and results are reproducible and documented in `showcase/RESULTS_ZH.md`. Main JSON results are in `showcase/results/`; the optional archival research ZIP includes more snapshots and complete traces.

## Pre-registered reporting constraints

- The **07** counterfactual is the scientific center; **05 and 06** are the accessible visual interfaces; **04 and 10** probe the unknown and the implementation's limits.
- No requirement that novelty, growth, survival, or a pretty image must occur.
- Count both unchanged intervals and relation dissolution.
- Count **generated programs as syntactic trees**, not independently discovered algorithms.
- A new contact is an instantiated `Contact` object; it is not a proof that the universal reachable state-space expanded.
- All adaptation, classification, game commands, and drawing outside `delta_core.py` are declared as external interfaces.
- A nonzero L2 difference in a fork is not automatically beneficial or biologically predictive.

For detail, see [Results (Chinese)](RESULTS_ZH.md) and [Methodology](METHODS.md).