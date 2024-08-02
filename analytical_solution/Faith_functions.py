import math
import numpy as np
from typing import Union, Tuple
from scipy.special import hermite
from structs import FunctionRInput, MuInput, FunctionStateNInput

def function_R(FunctionRInput: FunctionRInput, compute_phi = False) -> Union[float, Tuple[float, float]]:
    """
    Computes the function R(t, Omega, epsilon) and optionally the cosine of the phase angle phi.

    Args:
        FunctionRInput (FunctionRInput): A named tuple containing the input parameters t, Omega, and epsilon.
        compute_cos_phi (bool, optional): If True, also computes and returns the cosine of the phase angle phi. Defaults to False.

    Returns:
        Union[float, Tuple[float, float]]: If compute_cos_phi is False, returns the value of the function R.
                                        If compute_cos_phi is True, returns a tuple containing the value of R and the cosine of the phase angle phi.
    """
    # destructure the input
    t = FunctionRInput.t
    Omega = FunctionRInput.Omega
    epsilon = FunctionRInput.epsilon
    
    # compute the function
    R = (np.sin(Omega * t / 2 * np.sqrt(1 + epsilon ** 2)) ** 2) / (1 + epsilon ** 2)

    if compute_phi:
        phi = np.arccos(- epsilon * np.sqrt(R / (1 - R)))
        return R, phi
    
    return R



def function_mu(inputs: MuInput) -> float:
    """
     Computes the function mu(R, phi) based on the input parameters R and phi.
     
     Args:
         inputs (MuInput): A named tuple containing the input parameters R and phi.
     
     Returns:
         float: The computed value of the function mu.
     """
          # destructure the inputs
    R = inputs.R
    phi = inputs.phi

    # compute the output
    mu = np.sqrt(1 + ((1 - R) / R) * pow(np.cos(phi), 2)) - (np.cos(phi) * np.sqrt((1 - R) / R))
    return mu



def func_state_n(inputs: FunctionStateNInput) -> np.ndarray:
    def norm_constant(n: int) -> float:
        return 1.0 / np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi))

    C_n = norm_constant(inputs.n)
    H_n = hermite(inputs.n)
    psi_n = C_n * np.exp(- pow(inputs.x, 2) / 2) * H_n(inputs.x)
    
    return psi_n