"""Assert-based unit tests for mlcore (Part 0).

Run `python test_mlcore.py` from the repo root, or `%run test_mlcore.py` in a notebook.
Every test draws from mc.rng_for, so results are deterministic. The first failure stops the run.
"""

import traceback

import numpy as np
from numpy.polynomial.legendre import legval

import mlcore as mc


def _linear_data(tag, N=200, d=5, noise=0.0):
    """Well-conditioned regression data with bias column; returns X, y, w_star."""
    rng = mc.rng_for(tag)
    X = mc.add_bias(rng.standard_normal((N, d)))
    w_star = rng.standard_normal(d + 1)
    y = X @ w_star + noise * rng.standard_normal(N)
    return X, y, w_star


def _line_data(tag, N=100, flip=0.0):
    """2-D points in [-1, 1]^2 labelled by a random line, with a fraction of labels flipped."""
    rng = mc.rng_for(tag)
    X = mc.add_bias(rng.uniform(-1, 1, (N, 2)))
    p, q = rng.uniform(-1, 1, (2, 2))
    w_t = np.array([p[0] * q[1] - p[1] * q[0], p[1] - q[1], q[0] - p[0]])  # line through p and q
    y = np.where(X @ w_t > 0, 1.0, -1.0)
    flip_idx = rng.choice(N, int(round(flip * N)), replace=False)
    y[flip_idx] *= -1
    return X, y


# ===== Infrastructure and statistics =====

def test_rng_for():
    a = mc.rng_for("t").standard_normal(5)
    b = mc.rng_for("t").standard_normal(5)
    c = mc.rng_for("u").standard_normal(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_mean_se():
    m, se = mc.mean_se([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5
    assert abs(se - np.std([1, 2, 3, 4], ddof=1) / 2) < 1e-15
    assert abs(se - 0.6455) < 1e-4


def test_hoeffding():
    assert abs(mc.hoeffding_eps(110) - 0.1295) < 1e-4
    assert abs(mc.hoeffding_eps(2027) - 0.0302) < 1e-4
    assert mc.hoeffding_eps(110, M=2) > mc.hoeffding_eps(110)


def test_savefig_name():
    from matplotlib.figure import Figure   # no pyplot, so a notebook's backend is left alone

    fig = Figure()
    try:
        mc.savefig(fig, "Bad Name")
        raise AssertionError("bad figure name was accepted")
    except ValueError:
        pass
    path = mc.savefig(fig, "0_0_test_smoke")
    assert path.exists() and path.name == "fig_0_0_test_smoke.pdf"
    path.unlink()


# ===== Features, preprocessing, splits =====

def test_legendre_vs_legval():
    x = mc.rng_for("test_legendre").uniform(-1, 1, 50)
    L = mc.legendre_features(x, 10)
    assert L.shape == (50, 11)
    for q in range(11):
        assert np.max(np.abs(L[:, q] - legval(x, np.eye(11)[q]))) < 1e-12
    ends = mc.legendre_features(np.array([1.0, -1.0]), 10)
    assert np.allclose(ends[0], 1.0)
    assert np.allclose(ends[1], (-1.0) ** np.arange(11))
    assert mc.legendre_features(x, 0).shape == (50, 1)
    assert np.allclose(mc.legendre_features(x, 1), np.column_stack([np.ones(50), x]))


def test_poly_counts():
    rng = mc.rng_for("test_poly")
    assert mc.poly_features(rng.standard_normal((4, 2)), 3).shape == (4, 9)
    assert mc.poly_features(rng.standard_normal((4, 10)), 2).shape == (4, 65)
    assert mc.poly_features(rng.standard_normal((4, 30)), 2).shape == (4, 495)
    X = rng.standard_normal((6, 2))
    x1, x2 = X[:, 0], X[:, 1]
    expected = np.column_stack([x1, x2, x1**2, x1 * x2, x2**2])
    assert np.allclose(mc.poly_features(X, 2), expected)


def test_legendre_tensor():
    X = mc.rng_for("test_tensor").uniform(-1, 1, (20, 2))
    Z = mc.legendre_tensor_features(X, 3)
    assert Z.shape == (20, 10)
    assert np.allclose(Z[:, 0], 1.0)
    assert np.allclose(mc.legendre_tensor_features(X, 1), mc.add_bias(X))


def test_standardize_train_only():
    rng = mc.rng_for("test_standardize")
    X_train = rng.standard_normal((100, 3)) * [1.0, 5.0, 0.0] + [0.0, 2.0, 7.0]
    X_test = rng.standard_normal((50, 3)) + 5.0
    Xs, params = mc.standardize(X_train)
    assert np.allclose(Xs[:, :2].mean(axis=0), 0, atol=1e-12)
    assert np.allclose(Xs[:, :2].std(axis=0), 1, atol=1e-12)
    assert params["std"][2] == 1.0 and np.all(np.isfinite(Xs))
    Xt = mc.apply_standardize(X_test, params)
    assert np.allclose(Xt, (X_test - X_train.mean(axis=0)) / params["std"])
    assert not np.allclose(Xt[:, :2].mean(axis=0), 0, atol=0.5)


def test_minmax():
    rng = mc.rng_for("test_minmax")
    X_train = rng.uniform(3, 9, (40, 2))
    Xs, params = mc.minmax_scale(X_train)
    assert np.allclose(Xs.min(axis=0), -1) and np.allclose(Xs.max(axis=0), 1)
    X_test = rng.uniform(3, 9, (10, 2))
    expected = -1 + 2 * (X_test - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
    assert np.allclose(mc.apply_minmax(X_test, params), expected)
    const, _ = mc.minmax_scale(np.ones((5, 1)))
    assert np.all(np.isfinite(const))


def test_kfold():
    folds = mc.kfold_indices(23, 5, mc.rng_for("test_kfold"))
    assert len(folds) == 5
    vals = np.concatenate([v for _, v in folds])
    assert np.array_equal(np.sort(vals), np.arange(23))
    sizes = [v.size for _, v in folds]
    assert max(sizes) - min(sizes) <= 1
    for train, val in folds:
        assert np.intersect1d(train, val).size == 0
        assert np.array_equal(np.sort(np.concatenate([train, val])), np.arange(23))


def test_stratified_split():
    y = np.r_[np.ones(70), -np.ones(30)]
    train, test = mc.stratified_split(y, 0.3, mc.rng_for("test_strat"))
    assert np.sum(y[test] == 1) == 21 and np.sum(y[test] == -1) == 9
    assert np.intersect1d(train, test).size == 0
    assert np.array_equal(np.sort(np.r_[train, test]), np.arange(100))
    labels = np.array(["4"] * 10 + ["9"] * 10)
    _, test_s = mc.stratified_split(labels, 0.2, mc.rng_for("test_strat_str"))
    assert np.sum(labels[test_s] == "4") == 2


# ===== Metrics and gradient check =====

def test_confusion_f1_hand():
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1, -1, -1])
    yhat = np.array([1, 1, 1, -1, 1, 1, -1, -1, -1, -1])
    assert np.array_equal(mc.confusion_matrix(y, yhat), [[3, 1], [2, 4]])
    assert abs(mc.precision(y, yhat) - 0.6) < 1e-15
    assert abs(mc.recall(y, yhat) - 0.75) < 1e-15
    assert abs(mc.f1_score(y, yhat) - 2 / 3) < 1e-15
    assert abs(mc.accuracy(y, yhat) - 0.7) < 1e-15
    assert abs(mc.error_rate(y, yhat) - 0.3) < 1e-15
    assert mc.classification_report(y, yhat)["f1"] == mc.f1_score(y, yhat)


def test_auc_known():
    y = np.array([1, 1, -1, -1])
    fpr, tpr, _ = mc.roc_curve(y, [0.9, 0.4, 0.6, 0.1])
    assert abs(mc.auc(fpr, tpr) - 0.75) < 1e-15
    assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1
    fpr, tpr, _ = mc.roc_curve(y, [0.9, 0.8, 0.2, 0.1])
    assert mc.auc(fpr, tpr) == 1.0
    fpr, tpr, _ = mc.roc_curve(y, [0.5, 0.5, 0.5, 0.5])
    assert mc.auc(fpr, tpr) == 0.5


def test_regression_metrics():
    y = np.array([1.0, 2.0, 4.0, 7.0])
    assert mc.r2_score(y, y) == 1.0
    assert abs(mc.r2_score(y, np.full(4, y.mean()))) < 1e-15
    assert mc.mse(y, y + 1) == 1.0


def test_gradient_check_quadratic():
    w = mc.rng_for("test_gc").standard_normal(6)
    assert mc.gradient_check(lambda v: v @ v, lambda v: 2 * v, w) < 1e-8
    assert mc.gradient_check(lambda v: v @ v, lambda v: 3 * v, w) > 1e-2


# ===== Losses and solvers =====

def test_grad_linreg():
    X, y, _ = _linear_data("test_grad_lin", N=40, noise=0.5)
    w = mc.rng_for("test_grad_lin_w").standard_normal(X.shape[1])
    err = mc.gradient_check(lambda v: mc.linreg_loss_grad(v, X, y)[0],
                            lambda v: mc.linreg_loss_grad(v, X, y)[1], w)
    assert err < 1e-6, err


def test_grad_logistic():
    rng = mc.rng_for("test_grad_log")
    X = mc.add_bias(rng.standard_normal((40, 5)))
    y = np.where(rng.random(40) < 0.5, 1.0, -1.0)
    w = rng.standard_normal(6)
    for lam in (0.0, 0.1):
        for pb in (False, True):
            def f(v):
                return mc.logistic_loss_grad(v, X, y, lam, pb)[0]

            def g(v):
                return mc.logistic_loss_grad(v, X, y, lam, pb)[1]

            err = mc.gradient_check(f, g, w)
            assert err < 1e-6, (lam, pb, err)
    g0 = mc.logistic_loss_grad(w, X, y, 0.0)[1]
    g1 = mc.logistic_loss_grad(w, X, y, 0.1)[1]
    assert g0[0] == g1[0]  # bias unpenalized by default


def test_logistic_stable():
    X = np.array([[1.0], [1.0]])
    y = np.array([1.0, -1.0])
    w = np.array([1000.0])
    # no overflow, no nan (brief 3.2)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        loss, grad = mc.logistic_loss_grad(w, X, y)
        th = mc.theta(np.array([-1000.0, 1000.0]))
        p = mc.logistic_proba(w, X)
    assert abs(loss - 500.0) < 1e-9 and np.all(np.isfinite(grad))
    assert np.array_equal(th, [0.0, 1.0])
    assert np.all((p >= 0) & (p <= 1))


def test_solvers_agree():
    X, y, w_star = _linear_data("test_solvers")
    ws = [mc.solve_normal_inv(X, y), mc.solve_pinv_svd(X, y), mc.solve_qr(X, y), mc.solve_lstsq(X, y)]
    for w in ws:
        assert np.max(np.abs(w - w_star)) < 1e-10


def test_soft_threshold():
    out = mc.soft_threshold(np.array([-3.0, -0.5, 0.0, 0.5, 3.0]), 1.0)
    assert np.array_equal(out, [-2.0, 0.0, 0.0, 0.0, 2.0])


def test_ein_01():
    X, y = _line_data("test_ein01")
    assert mc.ein_01(np.zeros(3), X, y) == 1.0
    w = mc.pla(X, y, mc.rng_for("test_ein01_pla"), T=10000)[0]
    assert mc.ein_01(w, X, y) == 0.0
    assert mc.error_rate(y, mc.predict_sign(w, X)) == 0.0


# ===== Ridge =====

def _legendre_problem(tag):
    rng = mc.rng_for(tag)
    x = rng.uniform(-1, 1, 30)
    Z = mc.legendre_features(x, 10)
    y = np.sin(3 * x) + 0.3 * rng.standard_normal(30)
    return Z, y


def test_ridge_vs_direct():
    Z, y = _legendre_problem("test_ridge")
    lams = np.array([0.0, 1e-3, 1.0, 100.0])
    for pb in (False, True):
        P = np.eye(11) if pb else np.diag(np.r_[0.0, np.ones(10)])
        W = mc.ridge_svd(Z, y, lams, penalize_bias=pb)["W"]
        for k, lam in enumerate(lams):
            w_direct = np.linalg.solve(Z.T @ Z + lam * P, Z.T @ y)
            assert np.max(np.abs(W[k] - w_direct)) < 1e-8, (pb, lam)


def test_ridge_deff_limits():
    Z, y = _legendre_problem("test_ridge_deff")
    unpen = mc.ridge_svd(Z, y, [0.0, 1e12])["d_eff"]
    pen = mc.ridge_svd(Z, y, [0.0, 1e12], penalize_bias=True)["d_eff"]
    assert abs(unpen[0] - 11) < 1e-8 and abs(unpen[1] - 1) < 1e-6
    assert abs(pen[0] - 11) < 1e-8 and abs(pen[1]) < 1e-6


def test_ridge_hat_diag():
    Z, y = _legendre_problem("test_ridge_hat")
    for pb in (False, True):
        P = np.eye(11) if pb else np.diag(np.r_[0.0, np.ones(10)])
        H = Z @ np.linalg.solve(Z.T @ Z + 0.5 * P, Z.T)
        h = mc.ridge_svd(Z, y, 0.5, penalize_bias=pb)["H_diag"][0]
        assert np.max(np.abs(h - np.diag(H))) < 1e-10, pb


def test_ridge_rank_deficient():
    rng = mc.rng_for("test_ridge_rank")
    Z = mc.add_bias(rng.standard_normal((8, 20)))
    y = rng.standard_normal(8)
    out = mc.ridge_svd(Z, y, [0.0, 1.0])
    assert np.all(np.isfinite(out["W"])) and np.all(np.isfinite(out["H_diag"]))
    assert np.allclose(Z @ out["W"][0], y)  # interpolates when N < d at lam = 0


def test_ridge_requires_bias():
    Z, y = _legendre_problem("test_ridge_bias")
    try:
        mc.ridge_svd(Z[:, 1:], y, 1.0)
        raise AssertionError("missing bias column was accepted")
    except ValueError:
        pass


# ===== PLA and Pocket =====

def test_pla_separable():
    X, y = _line_data("test_pla")
    w, hist, n_updates = mc.pla(X, y, mc.rng_for("test_pla_run"), T=10000)
    assert n_updates is not None and hist[-1] == 0.0
    assert hist.size == n_updates + 1 and mc.ein_01(w, X, y) == 0.0


def test_pocket_monotone():
    X, y = _line_data("test_pocket", flip=0.1)
    w_hat, ein_w, ein_hat = mc.pocket(X, y, mc.rng_for("test_pocket_run"), T=500)
    assert ein_w.size == 501 and ein_hat.size == 501
    assert np.all(np.diff(ein_hat) <= 0)
    assert ein_hat[-1] <= ein_w.min() + 1e-12
    assert abs(mc.ein_01(w_hat, X, y) - ein_hat[-1]) < 1e-12


def test_pocket_w0():
    X, y = _line_data("test_pocket_w0", flip=0.1)
    w0 = mc.solve_lstsq(X, y)
    _, ein_w, ein_hat = mc.pocket(X, y, mc.rng_for("test_pocket_w0_run"), T=50, w0=w0)
    assert ein_w[0] == ein_hat[0] == mc.ein_01(w0, X, y)


# ===== Optimizers =====

def test_gd_linreg():
    X, y, _ = _linear_data("test_gd", noise=0.3)
    w, hist = mc.gradient_descent(lambda v: mc.linreg_loss_grad(v, X, y), np.zeros(X.shape[1]),
                                  1.0 / mc.lipschitz_linreg(X), 5000)
    assert np.max(np.abs(w - mc.solve_lstsq(X, y))) < 1e-6
    assert np.all(np.diff(hist) <= 1e-12)


def test_gd_linreg_batched_matches_loop():
    data = [_linear_data(f"test_gdb{r}", N=60, d=4, noise=0.3) for r in range(3)]
    X = np.stack([d[0] for d in data])
    y = np.stack([d[1] for d in data])
    eta = np.array([1.0 / mc.lipschitz_linreg(X[r]) for r in range(3)])
    W, iters = mc.gd_linreg_batched(X, y, eta, 300)
    assert np.all(iters == 300)
    for r in range(3):
        w_loop = mc.gradient_descent(lambda v: mc.linreg_loss_grad(v, X[r], y[r]), np.zeros(5), eta[r], 300)[0]
        assert np.max(np.abs(W[r] - w_loop)) < 1e-12


def test_gd_linreg_batched_stop():
    data = [_linear_data(f"test_gds{r}", N=60, d=4, noise=0.3) for r in range(3)]
    X = np.stack([d[0] for d in data])
    y = np.stack([d[1] for d in data])
    eta = np.array([1.0 / mc.lipschitz_linreg(X[r]) for r in range(3)])
    W_ref = np.stack([mc.solve_lstsq(X[r], y[r]) for r in range(3)])
    W, iters = mc.gd_linreg_batched(X, y, eta, 20000, w_ref=W_ref, rtol=1e-6, check_every=10)
    assert np.all((iters > 0) & (iters <= 20000))
    dist = np.linalg.norm(W - W_ref, axis=1) / np.linalg.norm(W_ref, axis=1)
    assert np.all(dist <= 1e-6)
    _, iters_capped = mc.gd_linreg_batched(X, y, eta, 5, w_ref=W_ref, rtol=1e-12)
    assert np.all(iters_capped == -1)


def test_lbfgs_linreg_logistic():
    X, y, _ = _linear_data("test_lbfgs", noise=0.3)
    w, res = mc.lbfgs(lambda v: mc.linreg_loss_grad(v, X, y), np.zeros(X.shape[1]))
    assert np.max(np.abs(w - mc.solve_lstsq(X, y))) < 1e-6 and w.dtype == np.float64
    Xc, yc = _line_data("test_lbfgs_log", flip=0.1)
    w, res = mc.lbfgs(lambda v: mc.logistic_loss_grad(v, Xc, yc, 0.01), np.zeros(3))
    assert np.linalg.norm(mc.logistic_loss_grad(w, Xc, yc, 0.01)[1]) < 1e-6


def test_prox_lam0():
    X, y, _ = _linear_data("test_prox0", noise=0.3)
    w, _ = mc.prox_gradient(lambda v: mc.linreg_loss_grad(v, X, y), np.zeros(X.shape[1]),
                            1.0 / mc.lipschitz_linreg(X), 0.0, 5000)
    assert np.max(np.abs(w - mc.solve_lstsq(X, y))) < 1e-6


def test_prox_sparsity():
    rng = mc.rng_for("test_prox_sparse")
    Xr = rng.standard_normal((100, 8))
    Xr -= Xr.mean(axis=0)
    X = mc.add_bias(Xr)
    y = Xr @ rng.standard_normal(8) + 3.0 + 0.1 * rng.standard_normal(100)
    lam_max = np.max(np.abs(2 * Xr.T @ (y - y.mean()) / 100))
    w, hist = mc.prox_gradient(lambda v: mc.linreg_loss_grad(v, X, y), np.zeros(9),
                               1.0 / mc.lipschitz_linreg(X), 10 * lam_max, 2000)
    assert np.all(w[1:] == 0.0)
    assert abs(w[0] - y.mean()) < 1e-6
    assert np.all(np.diff(hist) <= 1e-12)


# ===== Part 4 helpers =====

def test_legendre_target_norm():
    rng = mc.rng_for("test_target")
    for Qf in (0, 5, 20):
        a = mc.legendre_target(rng, Qf, size=4)
        q = np.arange(Qf + 1)
        assert np.allclose(np.sum(a**2 / (2 * q + 1), axis=1), 1.0, atol=1e-12)
        assert mc.legendre_target(rng, Qf).shape == (Qf + 1,)


def test_eout_legendre_vs_sample():
    rng = mc.rng_for("test_eout")
    a = mc.legendre_target(rng, 15)
    b = rng.standard_normal(11)
    # exact: Gauss-Legendre quadrature
    xq, wq = np.polynomial.legendre.leggauss(64)
    diff = mc.legendre_features(xq, 15) @ (np.r_[b, np.zeros(5)] - a)
    assert abs(0.5 * np.sum(wq * diff**2) - mc.eout_legendre(b, a)) < 1e-12
    # statistical: Monte Carlo, 10^6 noisy points
    x = rng.uniform(-1, 1, 10**6)
    y = mc.legendre_features(x, 15) @ a + np.sqrt(0.5) * rng.standard_normal(x.size)
    sq = (mc.legendre_features(x, 10) @ b - y) ** 2
    se = sq.std() / np.sqrt(sq.size)
    assert abs(sq.mean() - mc.eout_legendre(b, a, 0.5)) < 4 * se


def test_eout_padding_broadcast():
    rng = mc.rng_for("test_eout_pad")
    assert mc.eout_legendre(rng.standard_normal((3, 11)), rng.standard_normal(16)).shape == (3,)
    assert mc.eout_legendre(rng.standard_normal((5, 11)), rng.standard_normal((5, 21))).shape == (5,)
    b, a = rng.standard_normal(11), rng.standard_normal(16)
    assert abs(mc.eout_legendre(np.r_[b, 0.0, 0.0], a) - mc.eout_legendre(b, a)) < 1e-14


def test_lstsq_batched():
    rng = mc.rng_for("test_lstsq_batched")
    Z = rng.standard_normal((7, 30, 11))
    y = rng.standard_normal((7, 30))
    W = mc.lstsq_batched(Z, y)
    for r in range(7):
        assert np.max(np.abs(W[r] - mc.solve_lstsq(Z[r], y[r]))) < 1e-10


def test_overfit_sigma_cancels():
    rng = mc.rng_for("test_overfit")
    a, b10, b2 = rng.standard_normal(21), rng.standard_normal(11), rng.standard_normal(3)
    d0 = mc.eout_legendre(b10, a, 0.0) - mc.eout_legendre(b2, a, 0.0)
    d1 = mc.eout_legendre(b10, a, 1.7) - mc.eout_legendre(b2, a, 1.7)
    assert abs(d0 - d1) < 1e-12
    m = mc.overfit_measure(rng, 30, 0.3, 20, 50)
    assert m.shape == (50,) and np.all(np.isfinite(m))


def test_cv_curve_matches_loop():
    Z, y = _legendre_problem("test_cv_curve")
    lams = np.array([0.0, 1e-3, 0.1, 1.0, 10.0])
    folds = mc.kfold_indices(30, 5, mc.rng_for("test_cv_folds"))
    mean_err, fold_err = mc.cv_curve(mc.ridge_fit_predict(lams), Z, y, folds)
    assert fold_err.shape == (5, 5)
    P = np.diag(np.r_[0.0, np.ones(10)])
    manual = np.zeros((5, lams.size))
    for k, (tr, va) in enumerate(folds):
        for j, lam in enumerate(lams):
            w = np.linalg.solve(Z[tr].T @ Z[tr] + lam * P, Z[tr].T @ y[tr])
            manual[k, j] = np.mean((Z[va] @ w - y[va]) ** 2)
    assert np.max(np.abs(mean_err - manual.mean(axis=0))) < 1e-10
    labels = np.sign(y - y.mean())
    e01, _ = mc.cv_curve(mc.ridge_fit_predict(lams), Z, labels, folds, loss="01")
    assert np.all((e01 >= 0) & (e01 <= 1))


def test_loocv_closed_form_vs_brute():
    Z, y = _legendre_problem("test_loocv")
    lams = np.array([0.0, 1e-3, 0.1, 10.0])
    loo = mc.kfold_indices(30, 30, mc.rng_for("test_loocv_folds"))
    for pb in (False, True):
        closed = mc.loocv_ridge(Z, y, lams, penalize_bias=pb)
        brute = mc.cv_curve(mc.ridge_fit_predict(lams, pb), Z, y, loo)[0]
        assert np.max(np.abs(closed - brute)) < 1e-10, (pb, np.max(np.abs(closed - brute)))


# ===== Part 3 helpers =====

def test_linear_target_data():
    rng = mc.rng_for("test_target_data")
    w_t = mc.random_linear_target(rng)
    X, y_clean, y = mc.linear_target_data(rng, 100, w_t, flip=0.1)
    assert X.shape == (100, 3) and np.all(X[:, 0] == 1) and np.all(np.abs(X[:, 1:]) <= 1)
    assert np.array_equal(y_clean, np.where(X @ w_t > 0, 1.0, -1.0))
    assert np.sum(y != y_clean) == 10
    assert abs(w_t[0]) <= np.linalg.norm(w_t[1:]) * np.sqrt(2)   # the line meets the square
    w10 = mc.random_linear_target(rng, d=10)
    X10, _, _ = mc.linear_target_data(rng, 50, w10)
    assert X10.shape == (50, 11) and abs(w10[0]) <= np.sum(np.abs(w10[1:]))


def test_sgd_matches_full_gradient_in_expectation():
    rng = mc.rng_for("test_sgd_grad")
    X, _, y = mc.linear_target_data(rng, 80, mc.random_linear_target(rng), flip=0.1)
    eta = 1e-6
    for lam in (0.0, 0.1):
        w, _, _ = mc.sgd_logistic(X, y, mc.rng_for("test_sgd_grad_run"), eta, 1, lam=lam)
        step = -eta * X.shape[0] * mc.logistic_loss_grad(np.zeros(3), X, y, lam)[1]
        assert np.linalg.norm(w - step) / np.linalg.norm(step) < 1e-3, lam


def test_sgd_stopping_rule():
    rng = mc.rng_for("test_sgd_stop")
    X, y, _ = mc.linear_target_data(rng, 100, mc.random_linear_target(rng))
    w, n_epochs, hist = mc.sgd_logistic(X, y, mc.rng_for("test_sgd_stop_run"), 0.01, 100000, tol=0.01, record=True)
    assert n_epochs is not None and hist.size == n_epochs + 1
    # replay one epoch fewer to get the previous weights
    w_prev, _, _ = mc.sgd_logistic(X, y, mc.rng_for("test_sgd_stop_run"), 0.01, n_epochs - 1)
    assert np.linalg.norm(w - w_prev) < 0.01


def _noisy_logistic_problem(tag):
    rng = mc.rng_for(tag)
    X, _, y = mc.linear_target_data(rng, 100, mc.random_linear_target(rng), flip=0.1)
    w_opt, res = mc.lbfgs(lambda v: mc.logistic_loss_grad(v, X, y), np.zeros(3))
    return X, y, res.fun


def test_sgd_f_stop():
    X, y, f_opt = _noisy_logistic_problem("test_sgd_fstop")
    _, n_epochs, hist = mc.sgd_logistic(X, y, mc.rng_for("test_sgd_fstop_run"), 0.01, 5000, f_stop=f_opt + 1e-3)
    assert n_epochs is not None and hist[-1] <= f_opt + 1e-3 and hist.size == n_epochs + 1


def test_gd_f_stop():
    X, y, f_opt = _noisy_logistic_problem("test_gd_fstop")
    eta = 1.0 / mc.lipschitz_logistic(X)
    _, hist = mc.gradient_descent(lambda v: mc.logistic_loss_grad(v, X, y), np.zeros(3), eta, 10**5, f_stop=f_opt + 1e-6)
    assert hist[-1] <= f_opt + 1e-6 and hist.size - 1 < 10**5


# ===== Part 4 helpers (L1) =====

def test_poly_feature_combos():
    X = mc.rng_for("test_combos").standard_normal((7, 3))
    P = mc.poly_features(X, 2)
    combos = mc.poly_feature_combos(3, 2)
    assert len(combos) == P.shape[1] == 9
    for k, c in enumerate(combos):
        assert np.allclose(P[:, k], np.prod(X[:, list(c)], axis=1))


def _sparse_problem(tag):
    rng = mc.rng_for(tag)
    Xr = rng.standard_normal((80, 8))
    Xr -= Xr.mean(axis=0)
    w = np.r_[2.0, -1.5, np.zeros(6)]
    y = Xr @ w + 1.0 + 0.3 * rng.standard_normal(80)
    return mc.add_bias(Xr), y


def test_prox_path_matches_single():
    X, y = _sparse_problem("test_prox_path")
    lg = lambda v: mc.linreg_loss_grad(v, X, y)  # noqa: E731
    eta = 1.0 / mc.lipschitz_linreg(X)
    lams = np.array([2.0, 0.5, 0.05])
    W, iters = mc.prox_path(lg, 9, eta, lams, 50000, tol=1e-13)
    assert np.all(iters < 50000)
    for k, lam in enumerate(lams):
        w_cold, _ = mc.prox_gradient(lg, np.zeros(9), eta, lam, 50000, tol=1e-13)
        assert np.max(np.abs(W[k] - w_cold)) < 1e-6
    nnz = (W[:, 1:] != 0).sum(axis=1)
    assert nnz[0] <= nnz[-1]


# ===== Part 1, 3 and 5 helpers =====

def test_intensity_symmetry():
    img = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    F = mc.intensity_symmetry(np.stack([img, np.ones((3, 3))]))
    assert np.allclose(F[0], [2 / 9, -4 / 9])        # |img - mirror| is 1 at four corners
    assert np.allclose(F[1], [1.0, 0.0])             # constant image is symmetric
    assert np.allclose(mc.intensity_symmetry(img.reshape(1, 9)), F[:1])


def test_fista_matches_ista():
    X, y = _sparse_problem("test_fista")
    lg = lambda v: mc.linreg_loss_grad(v, X, y)  # noqa: E731
    eta = 1.0 / mc.lipschitz_linreg(X)
    for lam in (0.5, 0.05):
        w_f, it_f = mc.fista(lg, np.zeros(9), eta, lam, 100000, tol=1e-14)
        w_i, h_i = mc.prox_gradient(lg, np.zeros(9), eta, lam, 100000, tol=1e-14)
        assert np.max(np.abs(w_f - w_i)) < 1e-8, lam
        assert it_f <= h_i.size - 1
    lams = np.array([2.0, 0.5, 0.05])
    W_a, _ = mc.prox_path(lg, 9, eta, lams, 100000, tol=1e-14, accel=True)
    W_i, _ = mc.prox_path(lg, 9, eta, lams, 100000, tol=1e-14)
    assert np.max(np.abs(W_a - W_i)) < 1e-8


def test_cv_curve_logistic_loss():
    X, y = _line_data("test_cv_logistic", N=60, flip=0.1)
    folds = mc.kfold_indices(60, 5, mc.rng_for("test_cv_logistic_folds"))
    w = np.array([0.1, 1.0, -1.0])
    mean_err, fold_err = mc.cv_curve(lambda Xt, yt, Xv: Xv @ w, X, y, folds, loss="logistic")
    for k, (_, va) in enumerate(folds):
        assert np.isclose(fold_err[k, 0], mc.logistic_loss_grad(w, X[va], y[va])[0])
    assert np.isclose(mean_err[0], fold_err[:, 0].mean())


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            raise
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
