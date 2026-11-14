## 2026-11-14 — Supplement to existing DeltaLM research prerelease

Additional verified experiment: synthetic supervised calibration of actual DeltaLM Substrate coefficients, with held-out simulation worlds; baseline error 0.08705747, calibrated error approximately 1.22e-8. This is **not** DeltaLM self-originated learning or GPT weight training.

A/B/C/D prompt and provenance protocol added for text, image, speech and video. Unexecuted OpenAI API branches remain explicitly NOT_RUN, and the OpenAI Videos API is noted as shut down since 2026-09-24. Prior ten Showcase experiments and unmodified kernel are preserved.

See: integrations/open_generation/EXTENSION_RESULTS_ZH.md, core_calibration.py, ABCD_protocol.py, test_calibration_extension.py.