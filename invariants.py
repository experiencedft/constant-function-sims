import numpy as np

#A few example constant function invariants to test things with
def weightedConstantProduct(x, w):
    '''
    Balancer-style weighted constant product AMM

    Parameters:

        x: pool reserves vector
        w: weights, should be normalized such that sum(w) = 1

    Returns: 

        result: the value of the invariant at the specified parameters
    '''
    assert(sum(w) == 1)
    x = np.array(x)
    w = np.array(w)
    result = np.prod(x**w)
    return result

def weightedConstantSum(x, w):
    '''
    Constant sum AMM with weights, w1*x1 + ... + wn*xn

    Parameters:

        x: pool reserves vector
        w: weights, without any normalization constraint

    Returns:

        result: the value of the invariant at the specified parameters 
    '''
    x = np.array(x)
    w = np.array(w)
    return np.sum(x*w)

def modifiedStableSwap(x, weights_sum, weights_product, alpha):
    '''
    Modified StableSwap invariant: inspired by Angeris and Chitra's formulation, alpha weight for the constant sum and constant products part, weighted constant sum.

    Parameters:

        x: pool reserves vector

        weights_sum: weights of the weighted constant sum part

        weights_product: weights of the weighted constant product part

        alpha: weight of the invariant components. If alpha small, closer to a weighted constant sum. If alpha close to 1, closer to a weighted constant product.

    Returns: 

        result: the value of the invariant at the specified parameters

    '''
    x = np.array(x)
    weights_sum = np.array(weights_sum)
    weights_product = np.array(weights_product)
    result = (1-alpha)*np.sum(weights_sum*x) + alpha*np.prod(x**weights_product)
    return result