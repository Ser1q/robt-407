"""mlcore: NumPy toolkit for the ROBT 407 midterm (rule R1).

Design matrices carry the bias column at index 0; labels are in {-1, +1}.
"""

import itertools
import os
import re
import zlib
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.special

ROOT = Path(__file__).resolve().parent
SEED = 407
_FIG_NAME = re.compile(r"^[0-9]_[0-9a-z]+_[a-z0-9_]+$")


# ===== Infrastructure: FAST mode, seeding, figures, caches =====

def fast_mode():
    """True unless the env var FAST is set to something other than "1"."""
    return os.environ.get("FAST", "1") == "1"


def rng_for(tag):
    """Random generator for one experiment, seeded from SEED and the tag."""
    key = zlib.crc32(tag.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(key,)))


def savefig(fig, name):
    """Save fig to figures/fig_<name>.pdf."""
    if not _FIG_NAME.match(name):
        raise ValueError(f"figure name {name!r} must look like '1_2a_ein_curves'")
    path = ROOT / "figures" / f"fig_{name}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def save_cache(tag, **arrays):
    """Save arrays to cache/full/ (cache/fast/ in FAST mode)."""
    sub = "fast" if fast_mode() else "full"
    path = ROOT / "cache" / sub / f"{tag}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def load_full_cache(tag):
    """Arrays from cache/full/<tag>.npz, or None."""
    path = ROOT / "cache" / "full" / f"{tag}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# ===== Statistics =====

def mean_se(a, axis=0):
    """Mean and standard error along axis (R3)."""
    a = np.asarray(a, dtype=float)
    n = a.shape[axis]
    return a.mean(axis=axis), a.std(axis=axis, ddof=1) / np.sqrt(n)


def hoeffding_eps(n, delta=0.05, M=1):
    """Hoeffding margin sqrt(ln(2M/delta) / (2n))."""
    return float(np.sqrt(np.log(2.0 * M / delta) / (2.0 * n)))


# ===== Features and preprocessing =====

def add_bias(X):
    """Prepend a column of ones."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.hstack([np.ones((X.shape[0], 1)), X])


def legendre_features(x, Q):
    """Legendre features L0..LQ by the three-term recursion; column 0 is the bias."""
    x = np.asarray(x, dtype=float).ravel()
    L = np.empty((x.size, Q + 1))
    L[:, 0] = 1.0
    if Q >= 1:
        L[:, 1] = x
    for k in range(2, Q + 1):
        L[:, k] = ((2 * k - 1) / k) * x * L[:, k - 1] - ((k - 1) / k) * L[:, k - 2]
    return L


def poly_features(X, Q):
    """All monomials of degree 1..Q, without a bias column."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    cols = [
        np.prod(X[:, combo], axis=1)
        for deg in range(1, Q + 1)
        for combo in itertools.combinations_with_replacement(range(X.shape[1]), deg)
    ]
    return np.column_stack(cols)


def poly_feature_combos(d, Q):
    """Index tuples of the poly_features columns."""
    return [c for deg in range(1, Q + 1) for c in itertools.combinations_with_replacement(range(d), deg)]


def legendre_tensor_features(X, Q):
    """Products Li(x1) Lj(x2) with i + j <= Q; column 0 is the bias."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("legendre_tensor_features expects X of shape (N, 2)")
    L1 = legendre_features(X[:, 0], Q)
    L2 = legendre_features(X[:, 1], Q)
    cols = [L1[:, i] * L2[:, deg - i] for deg in range(Q + 1) for i in range(deg, -1, -1)]
    return np.column_stack(cols)


def intensity_symmetry(images):
    """Intensity and symmetry of square images, shape (N, 2)."""
    imgs = np.asarray(images, dtype=float)
    if imgs.ndim == 2:
        side = int(round(np.sqrt(imgs.shape[1])))
        if side * side != imgs.shape[1]:
            raise ValueError("intensity_symmetry needs square images")
        imgs = imgs.reshape(-1, side, side)
    intensity = imgs.mean(axis=(1, 2))
    symmetry = -np.abs(imgs - imgs[:, :, ::-1]).mean(axis=(1, 2))
    return np.column_stack([intensity, symmetry])


def standardize(X):
    """Z-score fitted on X; returns (Xs, params). Constant columns get std 1."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    params = {"mean": mean, "std": std}
    return apply_standardize(X, params), params


def apply_standardize(X, params):
    """Apply standardize params."""
    return (np.asarray(X, dtype=float) - params["mean"]) / params["std"]


def minmax_scale(X, lo=-1.0, hi=1.0):
    """Min-max scaling to [lo, hi] fitted on X; returns (Xs, params)."""
    X = np.asarray(X, dtype=float)
    params = {"min": X.min(axis=0), "max": X.max(axis=0), "lo": lo, "hi": hi}
    return apply_minmax(X, params), params


def apply_minmax(X, params):
    """Apply minmax_scale params."""
    span = params["max"] - params["min"]
    span = np.where(span > 0, span, 1.0)
    unit = (np.asarray(X, dtype=float) - params["min"]) / span
    return params["lo"] + unit * (params["hi"] - params["lo"])


# ===== Splits =====

def kfold_indices(N, K, rng):
    """Random K-fold (train_idx, val_idx) pairs."""
    folds = np.array_split(rng.permutation(N), K)
    out = []
    for k in range(K):
        train = np.concatenate([folds[j] for j in range(K) if j != k])
        out.append((np.sort(train), np.sort(folds[k])))
    return out


def stratified_split(y, test_frac, rng):
    """Train/test indices that keep class proportions."""
    y = np.asarray(y)
    test = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        test.append(idx[: max(1, int(round(test_frac * idx.size)))])
    test = np.sort(np.concatenate(test))
    return np.setdiff1d(np.arange(y.size), test), test


def random_split(N, test_frac, rng):
    """Random train/test indices."""
    perm = rng.permutation(N)
    n_test = int(round(test_frac * N))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


# ===== Simulated linear targets (Parts 1, 3) =====

def random_linear_target(rng, d=2):
    """Random linear target weights on [-1, 1]^d, bias first."""
    if d == 2:
        p, q = rng.uniform(-1.0, 1.0, (2, 2))
        return np.array([p[0] * q[1] - p[1] * q[0], p[1] - q[1], q[0] - p[0]])
    n = rng.standard_normal(d)
    x0 = rng.uniform(-1.0, 1.0, d)
    return np.r_[-n @ x0, n]


def linear_target_data(rng, N, w_t, flip=0.0):
    """Uniform points labelled by w_t with round(flip * N) labels flipped; returns X, y_clean, y."""
    X = add_bias(rng.uniform(-1.0, 1.0, (N, w_t.size - 1)))
    y_clean = np.where(X @ w_t > 0, 1.0, -1.0)
    idx = rng.choice(N, int(round(flip * N)), replace=False)
    y = y_clean.copy()
    y[idx] *= -1
    return X, y_clean, y


# ===== Metrics =====

def error_rate(y, yhat):
    """Fraction of wrong predictions."""
    return float(np.mean(np.asarray(y) != np.asarray(yhat)))


def accuracy(y, yhat):
    """Fraction of correct predictions."""
    return 1.0 - error_rate(y, yhat)


def confusion_matrix(y, yhat, pos=1):
    """[[TP, FN], [FP, TN]] for the class pos."""
    actual = np.asarray(y) == pos
    pred = np.asarray(yhat) == pos
    tp = np.sum(actual & pred)
    fn = np.sum(actual & ~pred)
    fp = np.sum(~actual & pred)
    tn = np.sum(~actual & ~pred)
    return np.array([[tp, fn], [fp, tn]], dtype=int)


def precision(y, yhat, pos=1):
    """TP / (TP + FP)."""
    (tp, _), (fp, _) = confusion_matrix(y, yhat, pos)
    return float(tp / (tp + fp)) if tp + fp > 0 else 0.0


def recall(y, yhat, pos=1):
    """TP / (TP + FN)."""
    (tp, fn), _ = confusion_matrix(y, yhat, pos)
    return float(tp / (tp + fn)) if tp + fn > 0 else 0.0


def f1_score(y, yhat, pos=1):
    """Harmonic mean of precision and recall."""
    p, r = precision(y, yhat, pos), recall(y, yhat, pos)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def classification_report(y, yhat, pos=1):
    """Dict of all classification metrics."""
    return {
        "error": error_rate(y, yhat),
        "accuracy": accuracy(y, yhat),
        "precision": precision(y, yhat, pos),
        "recall": recall(y, yhat, pos),
        "f1": f1_score(y, yhat, pos),
        "confusion": confusion_matrix(y, yhat, pos),
    }


def roc_curve(y, scores, pos=1):
    """ROC points (fpr, tpr, thresholds), one per distinct score."""
    y = np.asarray(y).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    is_pos = y[order] == pos
    n_pos = int(is_pos.sum())
    n_neg = is_pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_curve needs both classes in y")
    # last index of each group of tied scores
    last = np.r_[np.flatnonzero(np.diff(s_sorted)), s_sorted.size - 1]
    tp = np.cumsum(is_pos)[last]
    fp = last + 1 - tp
    fpr = np.r_[0.0, fp / n_neg]
    tpr = np.r_[0.0, tp / n_pos]
    return fpr, tpr, np.r_[np.inf, s_sorted[last]]


def auc(fpr, tpr):
    """Area under the curve by the trapezoid rule."""
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def mse(y, yhat):
    """Mean squared error."""
    r = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return float(np.mean(r**2))


def r2_score(y, yhat):
    """Coefficient of determination."""
    y = np.asarray(y, dtype=float)
    ss_res = np.sum((y - np.asarray(yhat, dtype=float)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def gradient_check(f, grad, w, h=1e-6):
    """Relative error of grad(w) against central differences of f."""
    w = np.asarray(w, dtype=float).copy()
    g = np.asarray(grad(w), dtype=float)
    g_num = np.empty_like(w)
    for i in range(w.size):
        e = np.zeros_like(w)
        e[i] = h
        g_num[i] = (f(w + e) - f(w - e)) / (2 * h)
    denom = max(np.linalg.norm(g_num) + np.linalg.norm(g), 1e-12)
    return float(np.linalg.norm(g_num - g) / denom)


# ===== Losses =====

def _pen_mask(m, penalize_bias):
    """Ones, with the bias entry 0 unless penalize_bias."""
    mask = np.ones(m)
    if not penalize_bias:
        mask[0] = 0.0
    return mask


def theta(s):
    """Logistic function, stable for any s."""
    return scipy.special.expit(s)


def ein_01(w, X, y):
    """Classification error of sign(w^T x); a margin of 0 counts as wrong."""
    return float(np.mean(y * (X @ w) <= 0))


def predict_sign(w, X):
    """sign(w^T x) with 0 mapped to -1."""
    return np.where(X @ w > 0, 1, -1)


def linreg_loss_grad(w, X, y):
    """Ein = ||Xw - y||^2 / N and its gradient."""
    r = X @ w - y
    N = y.size
    return float(r @ r / N), (2.0 / N) * (X.T @ r)


def logistic_loss_grad(w, X, y, lam=0.0, penalize_bias=False):
    """Cross-entropy + lam ||w[1:]||^2 and its gradient."""
    N = y.size
    margin = y * (X @ w)
    loss = float(np.mean(np.logaddexp(0.0, -margin)))
    grad = -(X.T @ (y * scipy.special.expit(-margin))) / N
    if lam:
        wp = w * _pen_mask(w.size, penalize_bias)
        loss += lam * float(wp @ wp)
        grad = grad + 2.0 * lam * wp
    return loss, grad


def logistic_proba(w, X):
    """P(y = +1 | x) = theta(w^T x)."""
    return scipy.special.expit(X @ w)


# ===== Linear least-squares solvers =====

def solve_normal_inv(X, y):
    """inv(X^T X) X^T y; only for the 2.3 conditioning study."""
    return np.linalg.inv(X.T @ X) @ (X.T @ y)


def solve_pinv_svd(X, y, rcond=None):
    """Minimum-norm least squares by SVD."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if rcond is None:
        rcond = max(X.shape) * np.finfo(float).eps
    keep = s > rcond * s[0]
    return Vt[keep].T @ ((U[:, keep].T @ y) / s[keep])


def solve_qr(X, y):
    """Least squares by QR."""
    Q, R = np.linalg.qr(X)
    return scipy.linalg.solve_triangular(R, Q.T @ y)


def solve_lstsq(X, y):
    """Least squares by LAPACK lstsq."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


# ===== Ridge regression (weight decay) =====

def ridge_svd(Z, y, lambdas, penalize_bias=False):
    """Weight decay Ein + (lam/N) ||w[1:]||^2 for all lambdas from one SVD; returns W, d_eff, H_diag."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    lam = np.atleast_1d(np.asarray(lambdas, dtype=float))
    N = Z.shape[0]
    if penalize_bias:
        A, yc = Z, y
    else:
        if not np.allclose(Z[:, 0], 1.0):
            raise ValueError("ridge_svd needs the bias column at Z[:,0]")
        # unpenalized bias: fit centred data, add the bias back
        mu = Z[:, 1:].mean(axis=0)
        A = Z[:, 1:] - mu
        y_mean = y.mean()
        yc = y - y_mean
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > s.max() * max(A.shape) * np.finfo(float).eps
    s2 = s**2
    # zero singular values get weight 0 (pseudo-inverse at lam = 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(keep, s / (s2 + lam[:, None]), 0.0)
        S = np.where(keep, s2 / (s2 + lam[:, None]), 0.0)
    W = (F * (U.T @ yc)) @ Vt
    d_eff = S.sum(axis=1)
    H_diag = S @ (U**2).T
    if not penalize_bias:
        W = np.column_stack([y_mean - W @ mu, W])
        d_eff = d_eff + 1.0
        H_diag = H_diag + 1.0 / N
    return {"W": W, "d_eff": d_eff, "H_diag": H_diag}


def soft_threshold(w, t):
    """Soft-thresholding, the prox of t ||.||_1."""
    return np.sign(w) * np.maximum(np.abs(w) - t, 0.0)


# ===== Legendre target experiments (Part 4) =====

def legendre_target(rng, Qf, size=None):
    """Random Legendre coefficients with E[f^2] = 1."""
    shape = (Qf + 1,) if size is None else (size, Qf + 1)
    a = rng.standard_normal(shape)
    q = np.arange(Qf + 1)
    return a / np.sqrt(np.sum(a**2 / (2 * q + 1), axis=-1, keepdims=True))


def legendre_dataset(rng, N, sigma2, Qf, size=None):
    """Noisy samples of a random Legendre target; returns x, y, a."""
    a = legendre_target(rng, Qf, size)
    shape = (N,) if size is None else (size, N)
    x = rng.uniform(-1.0, 1.0, shape)
    V = legendre_features(x.ravel(), Qf).reshape(*x.shape, Qf + 1)
    f = np.einsum("...nq,...q->...n", V, a)
    y = f + np.sqrt(sigma2) * rng.standard_normal(shape)
    return x, y, a


def eout_legendre(b, a, sigma2=0.0):
    """Analytic Eout = sum (b_q - a_q)^2 / (2q + 1) + sigma^2."""
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    L = max(b.shape[-1], a.shape[-1])

    def pad(c):
        return np.pad(c, [(0, 0)] * (c.ndim - 1) + [(0, L - c.shape[-1])])

    weights = 1.0 / (2 * np.arange(L) + 1)
    return np.sum((pad(b) - pad(a)) ** 2 * weights, axis=-1) + sigma2


def lstsq_batched(Z, y):
    """Least squares for a stack of problems by batched QR."""
    Q, R = np.linalg.qr(Z)
    return np.linalg.solve(R, np.einsum("rnm,rn->rm", Q, y)[..., None])[..., 0]


def overfit_measure(rng, N, sigma2, Qf, runs, Q_low=2, Q_high=10):
    """Eout(g_Q_high) - Eout(g_Q_low) for many runs at once."""
    x, y, a = legendre_dataset(rng, N, sigma2, Qf, size=runs)
    V = legendre_features(x.ravel(), Q_high).reshape(runs, N, Q_high + 1)
    b_high = lstsq_batched(V, y)
    b_low = lstsq_batched(V[..., : Q_low + 1], y)
    return eout_legendre(b_high, a, sigma2) - eout_legendre(b_low, a, sigma2)


# ===== Validation and cross-validation =====

def cv_curve(fit_predict, X, y, folds, loss="squared"):
    """Mean validation error per parameter over folds; loss "squared", "01" or "logistic"."""
    if loss not in ("squared", "01", "logistic"):
        raise ValueError("loss must be 'squared', '01' or 'logistic'")
    fold_err = []
    for tr, va in folds:
        pred = np.atleast_2d(fit_predict(X[tr], y[tr], X[va]))
        if loss == "squared":
            err = (pred - y[va]) ** 2
        elif loss == "01":
            err = np.where(pred > 0, 1, -1) != y[va]
        else:
            err = np.logaddexp(0.0, -y[va] * pred)
        fold_err.append(err.mean(axis=1))
    fold_err = np.array(fold_err)
    return fold_err.mean(axis=0), fold_err


def ridge_fit_predict(lambdas, penalize_bias=False):
    """cv_curve adapter for ridge_svd."""
    def fit_predict(Z_tr, y_tr, Z_val):
        return ridge_svd(Z_tr, y_tr, lambdas, penalize_bias)["W"] @ Z_val.T

    return fit_predict


def loocv_ridge(Z, y, lambdas, penalize_bias=False):
    """Closed-form leave-one-out error of weight decay for each lambda."""
    out = ridge_svd(Z, y, lambdas, penalize_bias)
    r = y - out["W"] @ Z.T
    return np.mean((r / (1.0 - out["H_diag"])) ** 2, axis=1)


# ===== Perceptron: PLA and Pocket =====

def pla(X, y, rng, T=1000, w0=None):
    """PLA on random misclassified points; returns w, Ein history, updates (None if not converged)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N = y.size
    w = np.zeros(X.shape[1]) if w0 is None else np.asarray(w0, dtype=float).copy()
    ein_hist = []
    for t in range(T + 1):
        mis = np.flatnonzero(y * (X @ w) <= 0)
        ein_hist.append(mis.size / N)
        if mis.size == 0:
            return w, np.array(ein_hist), t
        if t == T:
            break
        i = mis[rng.integers(mis.size)]
        w = w + y[i] * X[i]
    return w, np.array(ein_hist), None


def pocket(X, y, rng, T=1000, w0=None):
    """T PLA updates keeping the best weights; returns w_hat, Ein(w(t)), Ein(w_hat(t))."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N = y.size
    w = np.zeros(X.shape[1]) if w0 is None else np.asarray(w0, dtype=float).copy()
    ein_w = np.zeros(T + 1)
    ein_hat = np.zeros(T + 1)
    w_hat, best = w.copy(), np.inf
    for t in range(T + 1):
        mis = np.flatnonzero(y * (X @ w) <= 0)
        e = mis.size / N
        if e < best:
            w_hat, best = w.copy(), e
        ein_w[t], ein_hat[t] = e, best
        if mis.size == 0:
            break
        if t < T:
            i = mis[rng.integers(mis.size)]
            w = w + y[i] * X[i]
    return w_hat, ein_w, ein_hat


# ===== Optimizers =====

def lipschitz_linreg(X):
    """Lipschitz constant of the squared-error gradient."""
    return float(2.0 * np.linalg.norm(X, 2) ** 2 / X.shape[0])


def lipschitz_logistic(X, lam=0.0):
    """Lipschitz constant of the logistic gradient with L2 penalty lam."""
    return float(0.25 * np.linalg.norm(X, 2) ** 2 / X.shape[0] + 2.0 * lam)


def gradient_descent(loss_grad, w0, eta, n_iter, tol=0.0, f_stop=None):
    """Batch gradient descent; stops at ||grad|| <= tol or loss <= f_stop."""
    w = np.asarray(w0, dtype=float).copy()
    f, g = loss_grad(w)
    hist = [f]
    for _ in range(n_iter):
        if np.linalg.norm(g) <= tol or (f_stop is not None and f <= f_stop):
            break
        w = w - eta * g
        f, g = loss_grad(w)
        hist.append(f)
    return w, np.array(hist)


def sgd_logistic(X, y, rng, eta, max_epochs, tol=None, lam=0.0, penalize_bias=False, record=False, f_stop=None):
    """SGD for cross-entropy + lam ||w[1:]||^2, reshuffled each epoch; returns w, epochs, history."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1])
    pen = _pen_mask(X.shape[1], penalize_bias)
    track = record or f_stop is not None
    hist = [logistic_loss_grad(w, X, y, lam, penalize_bias)[0]] if track else None
    for epoch in range(1, max_epochs + 1):
        w_prev = w.copy()
        for n in rng.permutation(y.size):
            s = y[n] * (X[n] @ w)
            w = w - eta * (-y[n] * X[n] * scipy.special.expit(-s) + 2.0 * lam * pen * w)
        if track:
            f = logistic_loss_grad(w, X, y, lam, penalize_bias)[0]
            hist.append(f)
            if f_stop is not None and f <= f_stop:
                return w, epoch, np.array(hist)
        if tol is not None and np.linalg.norm(w - w_prev) < tol:
            return w, epoch, np.array(hist) if track else None
    return w, None, np.array(hist) if track else None


def gd_linreg_batched(X, y, eta, n_iter, w_ref=None, rtol=0.0, check_every=100):
    """Gradient descent on stacked least-squares problems; optional stop at rtol from w_ref."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    R, N, m = X.shape
    W = np.zeros((R, m))
    iters = np.full(R, -1)
    idx = np.arange(R)
    Xa, XTa, ya = X, np.ascontiguousarray(X.transpose(0, 2, 1)), y
    Wa = W.copy()
    eta_a = np.broadcast_to(np.asarray(eta, dtype=float), (R,)).copy()
    ref_norm = None if w_ref is None else np.linalg.norm(w_ref, axis=1)
    for t in range(1, n_iter + 1):
        r = (Xa @ Wa[:, :, None])[:, :, 0] - ya
        Wa = Wa - (eta_a * (2.0 / N))[:, None] * (XTa @ r[:, :, None])[:, :, 0]
        if w_ref is not None and (t % check_every == 0 or t == n_iter):
            dist = np.linalg.norm(Wa - w_ref[idx], axis=1) / ref_norm[idx]
            done = dist <= rtol
            if done.any():
                W[idx[done]] = Wa[done]
                iters[idx[done]] = t
                keep = ~done
                idx, Xa, XTa, ya, Wa, eta_a = idx[keep], Xa[keep], XTa[keep], ya[keep], Wa[keep], eta_a[keep]
                if idx.size == 0:
                    break
    W[idx] = Wa
    if w_ref is None:
        iters[:] = n_iter
    return W, iters


def prox_gradient(loss_grad, w0, eta, lam, n_iter, tol=0.0, penalize_bias=False):
    """ISTA for smooth(w) + lam ||w[1:]||_1; returns w and the objective history."""
    w = np.asarray(w0, dtype=float).copy()
    pen = _pen_mask(w.size, penalize_bias).astype(bool)
    f, g = loss_grad(w)
    hist = [f + lam * np.abs(w[pen]).sum()]
    for _ in range(n_iter):
        w_new = w - eta * g
        w_new[pen] = soft_threshold(w_new[pen], eta * lam)
        step = np.linalg.norm(w_new - w)
        w = w_new
        f, g = loss_grad(w)
        hist.append(f + lam * np.abs(w[pen]).sum())
        if step <= tol:
            break
    return w, np.array(hist)


def fista(loss_grad, w0, eta, lam, n_iter, tol=0.0, penalize_bias=False):
    """Accelerated ISTA with gradient restart; returns w and iterations used."""
    w = np.asarray(w0, dtype=float).copy()
    pen = _pen_mask(w.size, penalize_bias).astype(bool)
    v, t = w.copy(), 1.0
    for k in range(1, n_iter + 1):
        w_new = v - eta * loss_grad(v)[1]
        w_new[pen] = soft_threshold(w_new[pen], eta * lam)
        step = w_new - w
        if (v - w_new) @ step > 0:
            t = 1.0
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        v = w_new + ((t - 1.0) / t_new) * step
        w, t = w_new, t_new
        if np.linalg.norm(step) <= tol:
            return w, k
    return w, n_iter


def prox_path(loss_grad, m, eta, lambdas, n_iter, tol=0.0, penalize_bias=False, accel=False):
    """L1 path over decreasing lambdas with warm starts (accel=True uses fista)."""
    W = np.zeros((len(lambdas), m))
    iters = np.zeros(len(lambdas), dtype=int)
    w = np.zeros(m)
    for k, lam in enumerate(lambdas):
        if accel:
            w, iters[k] = fista(loss_grad, w, eta, lam, n_iter, tol, penalize_bias)
        else:
            w, hist = prox_gradient(loss_grad, w, eta, lam, n_iter, tol, penalize_bias)
            iters[k] = hist.size - 1
        W[k] = w
    return W, iters


def lbfgs(loss_grad, w0, tol=1e-12, maxiter=20000):
    """Minimize loss_grad with scipy L-BFGS-B."""
    res = scipy.optimize.minimize(
        loss_grad,
        np.asarray(w0, dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": tol, "gtol": tol, "maxiter": maxiter, "maxfun": 4 * maxiter},
    )
    return np.asarray(res.x, dtype=float), res
