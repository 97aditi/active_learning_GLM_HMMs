import numpy as np


def multinomial_rvs(count, p):
    """
    Sample from the multinomial distribution with multiple probability vectors.
    Inputs:
    Count: [n x 1] array to store the number of samples to be drawn from each of the nth distribiutions
    p: [n x k] holds sequence of n k-multinomial distributions

    Outputs:
    out: [n x k] array holding number of samples of each kind per distribution
    """
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out
