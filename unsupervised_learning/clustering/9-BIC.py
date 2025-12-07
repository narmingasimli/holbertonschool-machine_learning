#!/usr/bin/env python3
import numpy as np
"""try"""
# Assuming 8-EM.py is available and contains the expectation_maximization function
try:
    expectation_maximization = __import__('8-EM').expectation_maximization
except ImportError:
    # Handle the case where the dependency might not be found
    def expectation_maximization(X, k, iterations, tol, verbose):
        raise NotImplementedError("Dependency '8-EM.expectation_maximization' not found.")


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian Information Criterion.

    Args:
        X (np.ndarray): Dataset of shape (n, d).
        kmin (int): Minimum number of clusters to check for (inclusive).
        kmax (int): Maximum number of clusters to check for (inclusive).
                    If None, kmax is set to the maximum number of clusters possible (n).
        iterations (int): Maximum number of iterations for the EM algorithm.
        tol (float): Tolerance for the EM algorithm.
        verbose (bool): Determines if the EM algorithm should print information.

    Returns:
        tuple: (best_k, best_result, l, b) or (None, None, None, None) on failure.
               best_k (int): Best value for k based on its BIC.
               best_result (tuple): (pi, m, S) for the best k.
               l (np.ndarray): Log likelihood for each cluster size tested.
               b (np.ndarray): BIC value for each cluster size tested.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= 0 or kmax > n or kmax < kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    # Determine the number of cluster sizes to test
    num_k = kmax - kmin + 1

    # Initialize arrays to store results
    l = np.zeros(num_k)
    b = np.zeros(num_k)
    results = [None] * num_k
    k_values = range(kmin, kmax + 1)

    best_k = None
    best_result = None
    min_bic = float('inf')

    try:
        # We only need one loop to iterate through the range of k
        for i, k in enumerate(k_values):
            # 1. Run EM to get log likelihood and parameters
            pi, m, S, _, log_likelihood = expectation_maximization(
                X, k, iterations, tol, verbose)

            if log_likelihood is None:
                # Failure in EM
                return None, None, None, None

            # The final log likelihood is the last element
            current_log_likelihood = log_likelihood[-1]

            # 2. Calculate the number of parameters p
            # k - 1 parameters for pi (priors)
            # k * d parameters for m (means)
            # k * d * (d + 1) / 2 parameters for S (symmetric covariance matrices)
            p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)

            # 3. Calculate the BIC
            # BIC = p * ln(n) - 2 * l
            current_bic = p * np.log(n) - 2 * current_log_likelihood

            # 4. Store the results
            l[i] = current_log_likelihood
            b[i] = current_bic
            results[i] = (pi, m, S)

            # 5. Check for the best k (minimum BIC)
            if current_bic < min_bic:
                min_bic = current_bic
                best_k = k
                best_result = (pi, m, S)

        return best_k, best_result, l, b

    except Exception:
        # Catch any other potential errors
        return None, None, None, None
