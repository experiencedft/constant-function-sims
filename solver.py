import scipy 
from scipy import optimize
import numpy as np

def getSpotPrice(invariant, reserves, coordinates):
    '''
    Calculates the spot price of an asset denominated in another asset of an invariant given some specified reserves in the AMM pool. 
    Method described in Angeris and Chitra (2020): https://arxiv.org/pdf/2003.10001.pdf

    Parameters: 

        invariant (function): an invariant function taking *only* the reserves vector as argument.

        reserves (array[float]): the current reserves vector

        coordinates: coordinates vector specifying the coordinates along which we want to get the spot price. For example, if there is a 3 dimensional invariant f(x,y,z) and one wishes to get the spot price of the asset whose reserves are given by x (first dimension) denominated in the asset whose reserves are given by z (third dimension),  then one should call getSpotPrice(f, reserves, [0, 2]) where reserves are the current reserves. For example, calling spot(A, B) will give the price of token A in token B = some amount of A per B. 

    Returns: 
    
        spot: the spot price along the specified dimensions
    '''
    #Get gradient vector at the given reserves
    gradient = scipy.optimize.approx_fprime(reserves, invariant, 0.0001)
    #Calculate spot price
    spot = gradient[coordinates[1]]/gradient[coordinates[0]]
    return spot

def swapAmountIn(invariant, reserves, amount_in, coordinates):
    '''
    Given a amount_in amount of tokens along the coordinates[0] dimension added to the reserves, returns the amount of tokens along the coordinates[1] dimension to be given to the trader (removed from the reserves) to satisfy the invariant equation, assuming 0 fees. 

    Parameters: 

        invariant (function): an invariant function taking *only* the reserves vector as argument.

        reserves (array[float]): the current reserves vector

        coordinates (array[int]): coordinates along which the swap amount should be calculated. See getSpotPrice Docstring for a full explanation.

    Returns: 

        amount_out[0]: solution to the invariant equation given an amount in, using the fsolve method from scipy.optimize
    '''
    #Prepare the equation to feed to solver: solve phi(R, Delta, Gamma) = phi(R, 0, 0) for Gamma, see Angeris and Chitra paper for formalism.
    def invariantsDifference(amount_out):
        perturbed_reserves = reserves.copy()
        zeros = np.zeros(len(reserves))
        zeros[coordinates[0]] += amount_in
        zeros[coordinates[1]] -= amount_out
        perturbed_reserves = sum(np.array([zeros, reserves]))
        new_value = invariant(perturbed_reserves)
        difference = new_value - invariant(reserves)
        return difference
    #We should have perturbedInvariant - invariant(reserves) == 0. Solve root finding problem with SciPy.
    np.random.seed(0)
    amount_out = scipy.optimize.fsolve(invariantsDifference, np.random.rand(1)*amount_in)
    return amount_out[0]

def effectivePrice(invariant, reserves, amount_in, coordinates):
    '''
    Returns the effective price (amountIn/amountOut) of a trade tokens along the coordinates[0] dimension for tokens along the coordinates[1] dimension, denominated in the token that the user put in. 

    Parameters: 

        invariant (function): an invariant function taking *only* the reserves vector as argument.

        reserves (array[float]): the current reserves vector

        amount_in (float): the amount to trade in

        coordinates (array[int]): coordinates along which we want to get the effective price. See getSpotPrice Docstring for a full explanation
    '''
    amount_out = swapAmountIn(invariant, reserves, amount_in, coordinates)
    return amount_in/amount_out

def getSlippage(invariant, reserves, amount_in, coordinates):
    '''
    Calculates the slippage of a trade in %.

    Parameters: 

        invariant (function): an invariant function taking *only* the reserves vector as argument

        reserves (array[float]): the current reserves vector

        amount_in (float): the amount to trade in

        coordinates (array[int]): coordinates along which we want to get the slippage. See getSpotPrice Docstring for a full explanation
    '''
    effective_price = effectivePrice(invariant, reserves, amount_in, coordinates)
    spot_price = getSpotPrice(invariant, reserves, coordinates)
    return 100*(effective_price/spot_price - 1)