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
        super().__init__()
    
    def evaluate_objective(self, params: np.ndarray) -> float:
        """Evaluates the objective function of the problem for a given set of (binary) parameter values."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")
    
    def evaluate_constraints(self, params: np.ndarray) -> Iterable[bool]:
        """Evaluates the constraints of the problem for a given set of (binary) parameter values."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")

class BaseModel(object):
    """Base-class for models that can be derived from various optimization problems."""
    
    def __init__(self, variables, initializer) -> None:
        self.variables = variables
        self.initializer = initializer
        super().__init__()

class BaseBackend(object):
    """Base-class for neuromorphic hardware that can run models derived from QUBO."""
    
    def __init__(object):
        pass
    
    def run(self, executable, **kwargs) -> Tuple[int, dict]:
        """Run the given model with the given optional arguments"""
        raise NotImplementedError("Each backend must implement this method to specify how to run a model!")

class BaseCompiler(object):
    """Base-class for compilers that can convert optimization problems to models and compile models to a specific backend."""
    
    def __init__(self, problem: BaseProblem, modelType: type, backend: BaseBackend=None, expansion=None) -> None:
        self.problem = problem
        self.backend = backend
        self.modelType = modelType
        self.expansion = expansion
        
    def problem_to_model_parameters(self, params):
        """Converts the problem parameters to model parameters."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")
    
    def model_to_problem_parameters(self, params):
        """Converts the model parameters to problem parameters."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")
        
    def compile(self) -> BaseModel:
        """Exports the problem into the given model type, respecting the backend, if given."""
        raise NotImplementedError("This method should be defined in a concrete derived class!")
    
    