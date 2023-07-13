from typing import Tuple
import numpy as np
from collections.abc import Iterable

class BaseProblem(object):
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
    
    def __init__(self, initializer) -> None:
        self.initializer = initializer
    
    def supports_model(self, model_type) -> bool:
        """Returns whether exporting into the given model type is supported by this problem."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")
    
    def convert_to_model(self, model_type, backend=None):
        """Exports the problem into the given model type, respecting the backend, if given."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")
    
    def evaluate_objective(self, params: np.ndarray) -> float:
        """Evaluates the objective function of the problem for a given set of (binary) parameter values."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")
    
    def evaluate_constraints(self, params: np.ndarray) -> Iterable[bool]:
        """Evaluates the constraints of the problem for a given set of (binary) parameter values."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")
    
    
class BaseModel(object):
    """Base-class for models that can be derived from various optimization problems."""
    
    def __init__(self, variables, initializer, from_problem_parameters, to_problem_parameters, backend) -> None:
        self.variables = variables
        self.backend = backend
        self.initializer = initializer
        
        self._from_problem_parameters = from_problem_parameters
        self._to_problem_parameters = to_problem_parameters
        
        if not backend.supports_model(self):
            raise ValueError("The given backend does not support this model!")
        super().__init__()
        
    def run(self, **kwargs) -> Tuple[int, dict]:
        """Execute the model with the given optional arguments."""
        return self.backend.run(self, **kwargs)
    
    def from_problem_parameters(self, params):
        return self._from_problem_parameters(params)
    
    def to_problem_parameters(self, params):
        return self._to_problem_parameters(params)

class BaseBackend(object):
    """Base-class for neuromorphic hardware that can run models derived from QUBO."""
    
    def __init__(object):
        pass
    
    def supports_model(self, model: BaseModel) -> bool:
        raise NotImplementedError("Each backend must implement this method to specify which models it supports!")
    
    def run(self, model: BaseModel, **kwargs) -> Tuple[int, dict]:
        """Run the given model with the given optional arguments"""
        raise NotImplementedError("Each backend must implement this method to specify how to run a model!")
