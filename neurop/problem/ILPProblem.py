"""Integer linear programming (ILP) problems"""

from collections.abc import Iterable
import numpy as np
from optlang import Model as OptProblem

from neurop import BaseProblem, BaseModel
from neurop.model import QUBOModel
from neurop.base import BaseProblem


class ILPProblem(BaseProblem):
    """General integer linear programming (ILP) problem.
    
    ILPs are defined by a set of (in)equality constraints and an objective function over integer variables.
    The ILP can be formulated using the `optlang package <https://github.com/opencobra/optlang>`_.
    """
    
    @property
    def variables(self):
        return self.problem.variables
    
    @property
    def constraints(self):
        return self.problem.constraints
    
    @property
    def objective(self):
        return self.problem.objective
    
    def __init__(self, problem: OptProblem, initializer=None):
        """Initializes the ILP problem.

        Args:
            problem (OptProblem): the problem in `optlang.Model <https://optlang.readthedocs.io/en/latest/Model.html>`_ format
            initializer (_type_, optional): function to initialize the parameters of the problem. Defaults to None, which means that parameters will be randomly initialized.
        """
        self.problem = problem
        
        if initializer is None:
            initializer = lambda: dict((var, np.random.randint(var.lb, var.ub+1)) for var in self.problem.variables)
        
        super().__init__(initializer=initializer)
    
    def supports_model(self, model_type) -> bool:
        return model_type in [QUBOModel]    
    
    def evaluate_objective(self, params: np.ndarray) -> float:
        return self.problem.objective.expression.subs(params)
    
    def evaluate_constraints(self, params: np.ndarray) -> Iterable[bool]:
        sat = np.zeros((len(self.problem.constraints)+len(self.problem.variables),2), dtype=bool)
        i=0
        for v in self.problem.variables:
            sat[i,:] = (True if v.lb is None else v.lb <= int(params[v]), True if v.ub is None else params[v] <= v.ub)
            i += 1
            
        for c in self.problem.constraints:
            sat[i,:] = (True if c.lb is None else c.lb <= int(c.expression.subs(params)), True if c.ub is None else int(c.expression.subs(params)) <= c.ub)
            i += 1
            
        return sat 

    