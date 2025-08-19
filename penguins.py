import re
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Tuple

def load_penguins_txt(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses the provided messy 'penguins.txt' format.
    Returns:
        X_raw: float array of shape (n, 2) for [flipper_length_mm, body_mass_g]
        y: int labels (0=FEMALE, 1=MALE)
        label_names: np.array(["FEMALE","MALE"])
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            line = ln.strip()
            if not line:
                continue

            if "culmen" in line.lower():
                continue

            if "NA" in line:
                continue

            m = re.match(
                r"^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)([A-Za-z]+)?\s*$",
                line
            )
            if not m:

                continue
            a, b, flipper, mass, sex = m.groups()

            try:
                a = float(a); b = float(b); flipper = float(flipper); mass = float(mass)
            except ValueError:
                continue


            if not (0 < flipper < 400 and 500 < mass < 10000 and 20 < a < 70 and 10 < b < 40):
                continue

            sex = (sex or "").upper()
            if sex not in ("MALE", "FEMALE"):
                continue

            rows.append((flipper, mass, sex))

    if not rows:
        raise ValueError("No valid rows parsed. Check file path/format.")


    flipper = np.array([r[0] for r in rows], dtype=float)
    mass = np.array([r[1] for r in rows], dtype=float)
    sex_str = np.array([r[2] for r in rows], dtype=object)


    label_names, y = np.unique(sex_str, return_inverse=True)  # ['FEMALE','MALE']
    X_raw = np.column_stack([flipper, mass])
    return X_raw, y, label_names


def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def zscore_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma

def zscore_transform(X, mu, sigma):
    return (X - mu) / sigma


class KNearestNeighbors:
    def __init__(self, k: int = 5):
        self.k = int(k)
        self.X_train = None
        self.y_train = None

    def train(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

    def predict(self, X_test: np.ndarray, num_loops: int = 0) -> np.ndarray:
        if num_loops == 0:
            D = self._dist_vectorized(X_test)
        elif num_loops == 1:
            D = self._dist_one_loop(X_test)
        else:
            D = self._dist_two_loops(X_test)
        return self._predict_from_distances(D)

    def _dist_two_loops(self, X_test):
        n_test = X_test.shape[0]
        n_train = self.X_train.shape[0]
        D = np.zeros((n_test, n_train), dtype=float)
        for i in range(n_test):
            for j in range(n_train):
                diff = X_test[i] - self.X_train[j]
                D[i, j] = math.sqrt(np.dot(diff, diff))
        return D

    def _dist_one_loop(self, X_test):
        n_test = X_test.shape[0]
        D = np.zeros((n_test, self.X_train.shape[0]), dtype=float)
        for i in range(n_test):
            diff = self.X_train - X_test[i]
            D[i, :] = np.sqrt(np.sum(diff * diff, axis=1))
        return D

    def _dist_vectorized(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        t2 = np.sum(X_test**2, axis=1, keepdims=True)            # (n_test, 1)
        r2 = np.sum(self.X_train**2, axis=1, keepdims=True).T    # (1, n_train)
        cross = X_test @ self.X_train.T                          # (n_test, n_train)
        d2 = np.maximum(t2 + r2 - 2.0 * cross, 0.0)
        return np.sqrt(d2)

    def _predict_from_distances(self, D: np.ndarray) -> np.ndarray:
        n_test = D.shape[0]
        y_pred = np.empty(n_test, dtype=self.y_train.dtype)
        for i in range(n_test):
            nn_idx = np.argpartition(D[i], self.k)[:self.k]
            nn_labels = self.y_train[nn_idx]
            vals, counts = np.unique(nn_labels, return_counts=True)
            y_pred[i] = vals[np.argmax(counts)]
        return y_pred


def kfold_indices(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds

def cross_val_score_knn(X, y, k_values=(1,3,5,7,9,11,13,15), n_folds=5, seed=0):
    folds = kfold_indices(len(X), k=n_folds, seed=seed)
    scores = {k: [] for k in k_values}
    for k in k_values:
        clf = KNearestNeighbors(k=k)
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            clf.train(X[train_idx], y[train_idx])
            y_hat = clf.predict(X[val_idx])
            scores[k].append((y_hat == y[val_idx]).mean())
    # return mean CV accuracy per k
    return {k: float(np.mean(v)) for k, v in scores.items()}


def binom_p_value_greater(x, n, p0=0.5):
    """
    P(X >= x) for X~Binom(n,p0). Exact using DP cumulative (no scipy needed).
    """
    # compute pmf via DP
    from math import comb
    p = 0.0
    for k in range(x, n+1):
        p += comb(n, k) * (p0**k) * ((1-p0)**(n-k))
    return p



def main():
    X_raw, y, label_names = load_penguins_txt("penguins.txt")

    mu, sigma = zscore_fit(X_raw)
    X = zscore_transform(X_raw, mu, sigma)


    colors = np.array(["tab:orange", "tab:blue"])  # FEMALE, MALE
    plt.figure()
    for cls, name in enumerate(label_names):
        m = (y == cls)
        plt.scatter(X_raw[m, 0], X_raw[m, 1], label=name, alpha=0.7)
    plt.xlabel("Flipper length (mm)")
    plt.ylabel("Body mass (g)")
    plt.title("Penguins: Flipper vs Body Mass by Sex")
    plt.legend()
    plt.tight_layout()
    plt.show()


    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, seed=42)

    # 4) Cross-validate to pick k
    k_grid = (1,3,5,7,9,11,13,15,21,31)
    cv_scores = cross_val_score_knn(Xtr, ytr, k_values=k_grid, n_folds=5, seed=123)
    best_k = max(cv_scores.items(), key=lambda kv: kv[1])[0]
    print("CV accuracy by k:")
    for k in sorted(cv_scores):
        print(f"  k={k:<2d}  mean_acc={cv_scores[k]:.3f}")
    print(f"\nChosen k = {best_k}")


    knn = KNearestNeighbors(k=best_k)
    knn.train(Xtr, ytr)
    yhat = knn.predict(Xte)
    acc = (yhat == yte).mean()
    print(f"\nTest accuracy: {acc:.3f}  (n_test={len(yte)})")


    def confusion(y_true, y_pred):
        L = np.unique(y_true)
        mat = np.zeros((len(L), len(L)), dtype=int)
        for i, yt in enumerate(L):
            for j, yp in enumerate(L):
                mat[i, j] = np.sum((y_true == yt) & (y_pred == yp))
        return mat
    cm = confusion(yte, yhat)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"labels: {list(label_names)}")
    print(cm)

    # 6) Hypothesis test: accuracy > 0.5
    x = int((yhat == yte).sum())
    n = len(yte)
    pval = binom_p_value_greater(x, n, p0=0.5)
    print(f"\nBinomial test H0: accuracy = 0.5  vs  H1: accuracy > 0.5")
    print(f"  successes = {x}/{n}  => p-value = {pval:.4g}")

    if pval < 0.05:
        print("  Result: Reject H0 at α=0.05. KNN predicts sex better than chance.")
    else:
        print("  Result: Fail to reject H0 at α=0.05.")

if __name__ == "__main__":
    main()
