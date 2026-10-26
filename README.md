# ROBT 407 Midterm: Learning, Overfitting, and Regularization with Linear Models

Fall 2026 · Nuradil Serik, Dauletkali Toktybayev

Pocket PLA, linear and logistic regression with L2 and L1 regularization, written from scratch in NumPy (`mlcore.py`).

## Setup

Python 3.10:

```
pip install -r requirements.txt
```

California housing and MNIST are downloaded by scikit-learn on first use (cached in `~/scikit_learn_data`).

## Run

```
python test_mlcore.py                                        # 52 unit tests
jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

| Notebook | Items | FAST runtime |
|---|---|---|
| `part1_pla.ipynb` | unit tests, 1.1–1.4 | 10 s |
| `part2_linreg.ipynb` | 2.2–2.7 | 16 s |
| `part3_logreg.ipynb` | 3.2–3.6 | 29 s |
| `part4_overfit.ipynb` | 4.1–4.6 | 2 min |
| `part5_mnist.ipynb` | Part 5, tracks A and B | 3 min |

Total about 6 minutes in FAST mode (the default) on an Apple M4 Pro. `FAST=0` runs at full scale and writes `cache/full/`; FAST runs plot from those caches when present (2.3 and 2.6). All randomness is seeded from 407 via `mc.rng_for`.

## Conventions

- scikit-learn only loads data or appears in cells starting with `# Library comparison` (R1).
- The bias is column 0 of every design matrix and is never penalized.
- λ follows the brief per item: ridge Ein + (λ/N)‖w‖², logistic Ein + λ‖w‖², L1 Ein + λ‖w‖₁; scikit-learn uses C = 1/(2Nλ), Lasso α = λ/2, Ridge α = λ.
- Randomized results are mean ± SE (R3). Real data are split once; preprocessing is fitted on training data and refitted in every CV fold; the test set is used once, in cells marked `# TEST` (R4).
- Resolved ambiguities in the brief are listed in [DECISIONS.md](DECISIONS.md).

## Layout

| Path | Contents |
|---|---|
| `mlcore.py`, `test_mlcore.py` | Toolkit and its tests |
| `part*.ipynb` | One notebook per part |
| `figures/` | Figures used in the report |
| `cache/full/` | Full-scale results for 2.3 and 2.6 |
| `report/main.tex` | Report source (figures from `../figures/`) |
