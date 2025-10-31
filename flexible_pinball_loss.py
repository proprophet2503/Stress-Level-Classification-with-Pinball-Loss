import numpy as np
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, ClassifierMixin

class FPLinearPinballSVM(BaseEstimator, ClassifierMixin):
    """
    Linear Flexible Pinball SVM (primal QP).
    - Solves the primal convex QP:
        min 0.5 ||w||^2 + C * sum(zeta_j)
      s.t. zeta_j - tau1*(1 - y_j (w^T x_j + b)) >= 0
           zeta_j + tau2*(1 - y_j (w^T x_j + b)) >= 0
           zeta_j >= 0
    - Returns linear decision function f(x) = w^T x + b.
    - Multiclass: use OneVsRestClassifier(FPLinearPinballSVM(...))
    """

    def __init__(self, C=1.0, tau1=0.5, tau2=0.5, eps=1e-8):
        self.C = C
        self.tau1 = tau1
        self.tau2 = tau2
        self.eps = eps 
        self.w_ = None
        self.b_ = None
        self.zeta_ = None

    def _check_tau_validity(self):
        if not (-1.0 <= self.tau2 <= 1.0):
            raise ValueError("tau2 must be in [-1,1]")
        if not (-self.tau2 <= self.tau1 <= 1.0):
            raise ValueError("tau1 must satisfy tau1 in [-tau2, 1] to preserve convexity")
        

    def decision_function(self, X):
        """Return the raw signed distance from the hyperplane."""
        X = np.asarray(X, dtype=float)
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Predict class labels {-1, +1}."""
        return np.sign(self.decision_function(X))

    def predict_proba(self, X):
        """Return pseudo-probabilities by scaling the decision values with a sigmoid."""
        from scipy.special import expit
        decision = self.decision_function(X)
        probs = expit(decision)  # sigmoid to get [0,1]
        return np.vstack([1 - probs, probs]).T

    def score(self, X, y):
        """Return accuracy to allow sklearn scoring."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))



    def fit(self, X, y):
        self._check_tau_validity()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(float)

        if set(np.unique(y)) <= {0, 1}:
            y = np.where(y == 0, -1.0, 1.0)

        n, d = X.shape
        tau1, tau2, C = float(self.tau1), float(self.tau2), float(self.C)

        m = d + 1 + n

        # Quadratic cost
        P = np.zeros((m, m))
        P[:d, :d] = np.eye(d)
        P = matrix(P + 1e-6 * np.eye(m))
        q = np.zeros(m)
        q[d + 1:] = C
        q = matrix(q)

        G = np.zeros((3 * n, m))
        h = np.zeros(3 * n)

        for j in range(n):
            yj, xj = y[j], X[j]
            base = 3 * j
            G[base, :d] = -tau1 * yj * xj
            G[base, d] = -tau1 * yj
            G[base, d + 1 + j] = -1
            h[base] = -tau1

            G[base + 1, :d] = tau2 * yj * xj
            G[base + 1, d] = tau2 * yj
            G[base + 1, d + 1 + j] = -1
            h[base + 1] = tau2

            G[base + 2, d + 1 + j] = -1
            h[base + 2] = 0

        G, h = matrix(G), matrix(h)

        solvers.options.update({
            'show_progress': False,
            'abstol': 1e-6,
            'reltol': 1e-6,
            'feastol': 1e-6,
            'maxiters': 1000
        })

        sol = solvers.qp(P, q, G, h)
        x = np.array(sol['x']).ravel()

        self.w_ = x[:d]
        self.b_ = x[d]
        self.zeta_ = x[d + 1:]
        
        
        return self

