# Third-party code, datasets and output provenance

The DeltaLM source code is offered under Apache-2.0. **This does not relicense third-party libraries or underlying datasets.**

Dependencies used by the test scripts: NumPy, SciPy, Matplotlib, scikit-learn and Pillow (installed independently according to their own licenses). The core `delta_core.py` uses the Python standard library.

## Handwritten digit example

`create_samples.py` and `experiments.py` load `sklearn.datasets.load_digits()`. This contains a copy of the test subset of the **UCI Optical Recognition of Handwritten Digits** dataset, associated with E. Alpaydin and C. Kaynak. Dataset record: https://doi.org/10.24432/C50P49 . UCI record identifies a **CC BY 4.0** license; attribution and indication of changes are required. Dataset page: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits .

`results/image_examples.png` is a derived visualization of selected scikit-learn digits samples (8×8 samples plotted with grayscale and generated labels), not an original DeltaLM image dataset. The underlying digit data remains subject to its source terms. The GitHub repository does not include the full digits dataset.

## Other demo data

The short text corpus is generated within `adapters.py`; synthetic waveforms and moving-dot frames are produced by repository code. Charts and summary CSV/JSON are experiment outputs; no external commercial images or user records were intentionally inserted. This document does not constitute a third-party rights audit of all software packages.