import numpy as np
from scipy.stats import norm


class Data_Model_ConceptShift_Uniform:
    """
    Concept shift with feature-dependent intrinsic ambiguity and no covariate shift.

    Historical: deterministic labeling Y = g(X) based on X[:,0].
    New: beta fraction ambiguous (uniform over K classes) based on X[:,1] < delta.

    P_S(X) = P_T(X) = Uniform([0,1]^p)  (no covariate shift).
    P_T(Y|X) differs from P_S(Y|X) (concept shift).

    The deterministic rule g(x) partitions X[:,0] into K equal-probability bins
    using quantiles of the standard uniform distribution.
    """

    def __init__(self, K=5, p=15, random_state=2025, verbose=0):
        self.K = K
        self.p = p
        self.verbose = verbose
        self.random_state = int(random_state)
        self._seed_seq = np.random.SeedSequence(self.random_state)

        if p < 2:
            raise ValueError("p must be ≥ 2 (uses X[:, 0] and X[:, 1]).")

        # Compute thresholds a_k = Phi^{-1}(k/K) for k = 1, ..., K-1
        # These define the deterministic labeling rule g(x)
        # self.a = np.array([norm.ppf(k / self.K) for k in range(1, self.K)])
        self.a = np.array([k / self.K for k in range(1, self.K)])

    def _rng(self):
        """Generate a new random number generator."""
        child = self._seed_seq.spawn(1)[0]
        return np.random.Generator(np.random.PCG64(child))

    def _g(self, x1):
        """
        Deterministic labeling rule: g(x) = k iff a_{k-1} < x_1 <= a_k

        Args:
            x1: array of first coordinates
        Returns:
            array of class labels in {0, 1, ..., K-1}
        """
        # digitize with right=True gives us k such that a[k-1] < x1 <= a[k]
        return np.digitize(x1, bins=self.a, right=True)

    def _sample_X(self, n):
        """Sample X ~ N(0, I_p)"""
        rng = self._rng()
        X = rng.uniform(0, 1, (n, self.p)).astype(np.float32)
        # X = rng.normal(0, 1, (n, self.p)).astype(np.float32)
        return X

    # ---------- Core P(Y|X) for both distributions ----------
    def _prob_y_given_x_historical(self, X):
        """
        P_S(Y|X): deterministic, Y = g(X) based on X[:,0]

        Returns: probability matrix (n, K) where each row has one entry = 1.0
        """
        n = X.shape[0]
        P = np.zeros((n, self.K), dtype=float)
        labels = self._g(X[:, 0])
        P[np.arange(n), labels] = 1.0
        return P

    def _prob_y_given_x_new(self, X, delta):
        """
        P_T(Y|X) with concept shift:
        - If x_2 < delta (ambiguous region): Uniform over K classes
        - If x_2 >= delta (deterministic region): same as g(x)

        Args:
            X: feature matrix
            delta: threshold for ambiguity (beta)
        """
        n = X.shape[0]
        P = np.zeros((n, self.K), dtype=float)

        # Identify ambiguous vs deterministic samples based on X[:,1]
        ambiguous = X[:, 1] < delta
        deterministic = ~ambiguous

        # Ambiguous region: uniform distribution over all K classes
        if np.any(ambiguous):
            P[ambiguous, :] = 1.0 / self.K

        # Deterministic region: same as historical g(x)
        if np.any(deterministic):
            labels = self._g(X[deterministic, 0])
            P[deterministic, labels] = 1.0

        return P

    # ---------- Public API ----------
    def generate_historical_data(self, n_samples=3000):
        """
        Generate historical data under P_S:
        - X ~ N(0, I_p)
        - Y = g(X) deterministically based on X[:,0]

        All samples are "easy" (deterministic).
        """
        if self.verbose:
            print(f"Generating {n_samples} historical samples (all deterministic).")

        X = self._sample_X(n_samples)
        # Since it's deterministic, we can directly use g(X)
        y = self._g(X[:, 0])

        return X, y

    def generate_new_data(self, n_samples=3000, beta=0.5):
        """
        Generate new data under P_T with concept shift:
        - X ~ N(0, I_p) (same marginal as historical - no covariate shift)
        - Y ~ P_T(Y|X) where beta fraction are intrinsically ambiguous

        Args:
            n_samples: number of samples to generate
            beta: fraction of samples that are ambiguous (0 <= beta <= 1)
                  beta = P(X_2 < delta) = Phi(delta)

        Returns:
            X: feature matrix (n_samples, p)
            y: labels (n_samples,)
            sample_types: indicator (n_samples,) where 1 = deterministic, 0 = ambiguous
        """
        self.beta = float(beta)
        # Compute delta such that P(X_2 < delta) = beta under N(0,1)
        # delta = norm.ppf(beta)
        delta = beta

        if self.verbose:
            print(f"Generating {n_samples} new samples with ambiguity rate beta={beta:.1%}.")
            print(f"  (delta threshold = {delta:.3f})")

        X = self._sample_X(n_samples)
        P = self._prob_y_given_x_new(X, delta)

        # Sample Y according to P_T(Y|X)
        rng = self._rng()
        y = np.array([rng.choice(self.K, p=P[i]) for i in range(n_samples)], dtype=int)

        # sample_types: 1 = deterministic (easy), 0 = ambiguous (hard)
        ambiguous_mask = (X[:, 1] < delta)
        sample_types = (1 - ambiguous_mask.astype(int)).astype(float)

        return X, y, sample_types

    # ---------- Oracles ----------
    def calculate_c_star(self, X, y_hat, beta=None):
        """
        c*(x) = P_T(Y = ŷ | X=x) under the target distribution.

        Args:
            X: feature matrix
            y_hat: predicted labels
            beta: ambiguity rate (uses self.beta if None)
        """
        if beta is None:
            if not hasattr(self, 'beta'):
                raise ValueError("Must specify beta or call generate_new_data first.")
            beta = self.beta

        # delta = norm.ppf(beta)
        delta = beta
        y_hat = np.asarray(y_hat)
        P = self._prob_y_given_x_new(X, delta)
        return P[np.arange(X.shape[0]), y_hat]

    def calculate_eta_star(self, X, y_hat, beta=None):
        """
        η*(x) over the non-ŷ classes under P_T(Y|X).

        Args:
            X: feature matrix
            y_hat: predicted labels
            beta: ambiguity rate (uses self.beta if None)
        """
        if beta is None:
            if not hasattr(self, 'beta'):
                raise ValueError("Must specify beta or call generate_new_data first.")
            beta = self.beta

        # delta = norm.ppf(beta)
        delta = beta
        y_hat = np.asarray(y_hat)
        P = self._prob_y_given_x_new(X, delta).copy()
        P[np.arange(X.shape[0]), y_hat] = 0.0
        denom = P.sum(axis=1, keepdims=True)
        out = np.zeros_like(P)
        nz = denom[:, 0] > 1e-12
        out[nz] = P[nz] / denom[nz]
        if (~nz).any():  # degenerate case: P puts mass 1 on ŷ
            K = P.shape[1]
            out[~nz] = 1.0 / (K - 1)
            out[~nz, y_hat[~nz]] = 0.0
        return out




class Data_Model_CovariateShift_Uniform:
    """
    Covariate shift with feature-dependent intrinsic ambiguity.

    Historical: X ~ Unif([0,1]^p).
    New:        X[:,1] ~ Unif([0,a]) with a < 1; all other features unchanged.

    P_S(X) != P_T(X) (covariate shift on X[:,1]).
    P(Y|X) is IDENTICAL under both distributions (no concept shift).

    The labeling rule is fixed:
      - If x_2 < beta: Y ~ Unif({0,...,K-1})  (ambiguous region)
      - If x_2 >= beta: Y = g(x)               (deterministic region)

    Because X[:,1] ~ Unif([0,a]) at deployment, P_T(X[:,1] < beta) = beta/a > beta,
    so a higher fraction of deployment samples fall in the ambiguous region.
    """

    def __init__(self, K=5, p=15, random_state=2025, verbose=0):
        self.K = K
        self.p = p
        self.verbose = verbose
        self.random_state = int(random_state)
        self._seed_seq = np.random.SeedSequence(self.random_state)

        if p < 2:
            raise ValueError("p must be >= 2 (uses X[:, 0] and X[:, 1]).")

        # Thresholds defining the deterministic labeling rule g(x)
        self.thresholds = np.array([k / self.K for k in range(1, self.K)])

    def _rng(self):
        """Generate a new random number generator."""
        child = self._seed_seq.spawn(1)[0]
        return np.random.Generator(np.random.PCG64(child))

    def _g(self, x1):
        """
        Deterministic labeling rule: g(x) = k iff a_{k-1} < x_1 <= a_k

        Args:
            x1: array of first coordinates
        Returns:
            array of class labels in {0, 1, ..., K-1}
        """
        return np.digitize(x1, bins=self.thresholds, right=True)

    def _sample_X(self, n, a_covariate=1.0):
        """
        Sample X with optional covariate shift on X[:,1].

        Historical: all features ~ Unif([0,1]).
        New:        X[:,1] ~ Unif([0, a_covariate]), rest ~ Unif([0,1]).

        Args:
            n:            number of samples
            a_covariate:  upper bound of X[:,1]; 1.0 = no shift
        """
        rng = self._rng()
        X = rng.uniform(0, 1, (n, self.p)).astype(np.float32)
        if a_covariate < 1.0:
            X[:, 1] = rng.uniform(0, a_covariate, n).astype(np.float32)
        return X

    # ---------- Core P(Y|X) — identical for both distributions ----------
    def _prob_y_given_x(self, X, beta):
        """
        P(Y|X): shared by both historical and new distributions.
          - x_2 < beta:  Unif over K classes  (ambiguous)
          - x_2 >= beta: point mass on g(x)   (deterministic)

        Args:
            X:    feature matrix (n, p)
            beta: ambiguity threshold on X[:,1]
        Returns:
            probability matrix (n, K)
        """
        n = X.shape[0]
        P = np.zeros((n, self.K), dtype=float)

        ambiguous     = X[:, 1] < beta
        deterministic = ~ambiguous

        if np.any(ambiguous):
            P[ambiguous, :] = 1.0 / self.K

        if np.any(deterministic):
            labels = self._g(X[deterministic, 0])
            P[deterministic, labels] = 1.0

        return P

    # ---------- Public API ----------
    def generate_historical_data(self, n_samples=3000, beta=0.1):
        """
        Generate historical data under P_S:
          - X ~ Unif([0,1]^p)
          - Y ~ P(Y|X) with ambiguity threshold beta

        Args:
            n_samples: number of samples
            beta:      ambiguity threshold
        Returns:
            X: feature matrix (n_samples, p)
            y: labels (n_samples,)
        """
        self.beta = float(beta)

        if self.verbose:
            print(f"Generating {n_samples} historical samples.")
            print(f"  P_S(X[:,1] < beta={beta}) = {beta:.1%} ambiguous")

        X = self._sample_X(n_samples, a_covariate=1.0)
        P = self._prob_y_given_x(X, beta)

        rng = self._rng()
        y = np.array([rng.choice(self.K, p=P[i]) for i in range(n_samples)], dtype=int)

        return X, y

    def generate_new_data(self, n_samples=3000, beta=0.1, a=0.5):
        """
        Generate new data under P_T with covariate shift on X[:,1]:
          - X[:,1] ~ Unif([0, a]) instead of Unif([0,1])
          - P(Y|X) is UNCHANGED (same labeling rule as historical)

        Because X[:,1] is compressed into [0, a], the effective fraction of
        ambiguous samples becomes beta/a > beta.

        Args:
            n_samples: number of samples
            beta:      ambiguity threshold (same as historical)
            a:         upper bound of X[:,1] at deployment; a < 1 means shift
        Returns:
            X:            feature matrix (n_samples, p)
            y:            labels (n_samples,)
            sample_types: 1 = deterministic (easy), 0 = ambiguous (hard)
        """
        self.beta        = float(beta)
        self.covariate_a = float(a)

        effective_ambiguous_rate = min(beta / a, 1.0)

        if self.verbose:
            print(f"Generating {n_samples} new samples with covariate shift (a={a}).")
            print(f"  P_T(X[:,1] < beta={beta}) = {effective_ambiguous_rate:.1%} ambiguous")

        X = self._sample_X(n_samples, a_covariate=a)
        P = self._prob_y_given_x(X, beta)

        rng = self._rng()
        y = np.array([rng.choice(self.K, p=P[i]) for i in range(n_samples)], dtype=int)

        ambiguous_mask = (X[:, 1] < beta)
        sample_types   = (1 - ambiguous_mask.astype(int)).astype(float)

        return X, y, sample_types

    # ---------- Oracles ----------
    def calculate_c_star(self, X, y_hat, beta=None):
        """
        c*(x) = P(Y = ŷ | X=x) — identical under P_S and P_T since P(Y|X) is unchanged.

        Args:
            X:     feature matrix
            y_hat: predicted labels
            beta:  ambiguity threshold (uses self.beta if None)
        """
        if beta is None:
            if not hasattr(self, 'beta'):
                raise ValueError("Must specify beta or call generate_historical_data/generate_new_data first.")
            beta = self.beta

        y_hat = np.asarray(y_hat)
        P = self._prob_y_given_x(X, beta)
        return P[np.arange(X.shape[0]), y_hat]

    def calculate_eta_star(self, X, y_hat, beta=None):
        """
        eta*(x): distribution of Y over non-ŷ classes given Y != ŷ, under P(Y|X).

        Args:
            X:     feature matrix
            y_hat: predicted labels
            beta:  ambiguity threshold (uses self.beta if None)
        """
        if beta is None:
            if not hasattr(self, 'beta'):
                raise ValueError("Must specify beta or call generate_historical_data/generate_new_data first.")
            beta = self.beta

        y_hat = np.asarray(y_hat)
        P = self._prob_y_given_x(X, beta).copy()
        P[np.arange(X.shape[0]), y_hat] = 0.0
        denom = P.sum(axis=1, keepdims=True)
        out = np.zeros_like(P)
        nz = denom[:, 0] > 1e-12
        out[nz] = P[nz] / denom[nz]
        if (~nz).any():
            K = P.shape[1]
            out[~nz] = 1.0 / (K - 1)
            out[~nz, y_hat[~nz]] = 0.0
        return out


