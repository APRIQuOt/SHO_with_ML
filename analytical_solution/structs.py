"""
This module defines several NamedTuple classes that serve as input data structures for various functions in the analytical_solution package.

ABInput:
    Represents input data for a function that takes 6 parameters: w1, w2, A12, A21, B, and C.

MuInput:
    Represents input data for a function that takes 2 parameters: R and phi.

FunctionAInput:
    Represents input data for a function that takes 3 parameters: mu, s1, and s2.

FunctionCInput:
    Represents input data for a function that takes 3 parameters: A, R, and phi.

FunctionStateNInput:
    Represents input data for a function that takes 2 parameters: n and x.

FunctionRInput:
    Represents input data for a function that takes 3 parameters: Omega, t, and epsilon.
"""
import numpy as np
from typing import NamedTuple, Union

class ABInput(NamedTuple):
    w1: Union[float, int]
    w2: Union[float, int]
    A12: Union[float, int]
    A21: Union[float, int]
    B: Union[float, int]
    C: Union[float, int]

class MuInput(NamedTuple):
    R: float
    phi: float

class FunctionAInput(NamedTuple):
    mu: float
    s1: int
    s2: int

class FunctionCInput(NamedTuple):
    A: np.ndarray
    R: float
    phi: float

class FunctionStateNInput(NamedTuple):
    n: int
    x: np.ndarray

class FunctionRInput(NamedTuple):
    Omega: float
    t: float
    epsilon: float