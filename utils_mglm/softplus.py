import numpy as np 

def softplus(x):
    """ Function to compute softplus and return its first and second derivatives
        f(x) = log(1+exp(x))
        Also known as the soft-rectification function
    """

    f = np.log(1+np.exp(x))
    df = np.exp(x)/(1+np.exp(x))
    ddf = np.exp(x)/((1+np.exp(x))**2)

    # Check for small values to avoid underflow errors
    if np.any(x<-20):
        iix = np.where(x<-20)
        f[iix] = np.exp(x[iix])
        df[iix] = f[iix]
        ddf[iix] = f[iix]

    # Check for large values to avoid overflow errors
    if np.any(x>500):
        iix = np.where(x>500)
        f[iix] = x[iix]
        df[iix] = 1;
        ddf[iix] = 0;

    return f, df, ddf


def only_softplus(x):
    """ Function to compute softplus and return its first and second derivatives
        f(x) = log(1+exp(x))
        Also known as the soft-rectification function
    """
    f = np.log(1+np.exp(x))

    # Check for small values to avoid underflow errors
    if np.any(x<-20):
        iix = np.where(x<-20)
        f[iix] = np.exp(x[iix])

    # Check for large values to avoid overflow errors
    if np.any(x>500):
        iix = np.where(x>500)
        f[iix] = x[iix]

    return f

