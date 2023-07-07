"""Base-classes for NP-hard problems

.. autoclass:: neurop.Problem.Problem.Problem
"""

import numpy as np
from collections.abc import Iterable

class Problem(object):
    """Base class to define an optimization problem that can be solved via QUBO.
    This class acts as an interface, and its methods should be implemented by derived classes.

    For derived classes, implement the following methods:
    
    - :meth:`to_qubo`
    - :meth:`to_ising`  (optional)
    - :meth:`evaluate_objective`
    - :meth:`evaluate_constraints`
    - :meth:`parameters_to_qubo`
    - :meth:`parameters_from_qubo`
    """
    def __init__(self) -> None:
        pass
    
    def to_qubo(self) -> np.ndarray:
        """Generates the matrix of the QUBO problem."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")
    
    def to_ising(self):
        """Converts the problem to Ising form (optional)."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")
    
    def evaluate_objective(self, params: np.ndarray) -> float:
        """Evaluates the objective function of the problem for a given set of (binary) parameter values."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")
    
    def evaluate_constraints(self, params: np.ndarray) -> Iterable[bool]:
        """Evaluates the constraints of the problem for a given set of (binary) parameter values."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")
    
    def parameters_to_qubo(self, params: dict) -> np.ndarray:
        """Converts a given set of integer parameters to the corresponding binary expansions."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")
    
    def parameters_from_qubo(self, params: np.ndarray) -> dict:
        """Converts a given set of binary parameters to integer parameters of the original problem."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")