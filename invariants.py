import numpy as np

#A few example constant function invariants to test things with, all assuming 0 fees

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

class StableSwapInvariant():
    '''
    Original stableswap invariant. Defined as a class since it should be initialized with the proper value of D given the initial pool content, and that D must be kept in memory.

    Attributes:
    ____________

    A: the amplification factor of the particular invariant simulated

    D: the value of D for the invariant, always the same since the trades specifically do not change the invariant. 

    n = the number of tokens in the pool for the particular invariant simulated

    xp: pool balances vector

    Methods:
    ____________

    getInvariantValue(current_balances): given some pool balances, returns the value of the invariant for these pool balances, to be turned into a lambda function for use with solver.py
    '''
    def __init__(self, initial_pool_balances, amplification_factor):
        '''
        Params:

        initial_pool_balances: list[float]
            list whose entries represent the amount of each token in the pool 

        amplification_factor: float
        '''

        self.A = amplification_factor
        self.n = len(initial_pool_balances)
        self.xp = initial_pool_balances

        #Calculate D using the Newton method exactly like in the Curve contracts
        S = 0

        for _x in self.xp:
            S += _x
        D = S
        Ann = self.A * (self.n)**(self.n)
        for _i in range(255):
            D_P = D
            for _x in self.xp:
                D_P = D_P * D / (_x * self.n + 1)  # +1 is to prevent /0
            D = (Ann * S + D_P * self.n) * D / ((Ann - 1) * D + (self.n + 1) * D_P)
            #Not checking whether convergence happened, 255 steps is more than enough and D has the correct value at this stage

        self.D = D
    
    def getInvariantValue(self, current_balances):
        '''
        Params: 

            current_balances: list[float]
                the current balances of the pool 
        
        Returns:

            value: float
                the current value of the invariant, will always be 0 in the case of StableSwap but this is for use with the solver.py functions
        '''
        current_balances = np.array(current_balances)
        value = self.A*self.n**(self.n)*np.sum(current_balances) + self.D - self.A*self.D*self.n**(self.n) + ((self.D)**(self.n + 1))/(np.prod(current_balances)*self.n**(self.n))
        return value