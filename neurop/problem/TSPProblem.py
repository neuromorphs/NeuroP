"""Traveling salesperson (TSP) problems
"""

from collections.abc import Iterable
import numpy as np
import sympy as sp

import networkx as nx

from neurop import BaseProblem
from neurop import BaseModel
from neurop.model import QUBOModel
from neurop.expansions import expand_integer_to_binary_encoding, substitute_binary_to_integer, substitute_integer_to_binary
from neurop.utils import range_of_polynomial_with_bounded_vars


class TSPProblem(BaseProblem):
    """Traveling Salesperson Problems"""
    
    def __init__(self, graph: nx.Graph, initializer=None, penalty=10):
        """Initializes the TSP problem.

        Args:
            problem (OptProblem): the optlang problem
            initializer (_type_, optional): function to initialize the parameters of the problem. Defaults to None, which means that parameters will be randomly initialized.
        """
        self.graph = graph
        self.penalty = penalty
        
        if initializer is None:
            
            initializer = lambda: dict(zip(
                self.graph.nodes(),
                np.random.permutation(len(self.graph.nodes()))
                ))
        
        super().__init__(initializer=initializer)
    
    def supports_model(self, model_type) -> bool:
        return model_type in [QUBOModel]
    
    def convert_to_model(self, model_type, backend=None) -> BaseModel:
        """Converts the problem to a model of given type, if supported.

        Args:
            backend (Backend, optional): The backend for which to derive the QUBO form. Defaults to None, which means no special requirements are imposed on the Q matrix.

        Raises:
            ValueError: May return a ValueError if the problem cannot be converted to the required model type.

        Returns:
            np.ndarray: returns the QUBO matrix
        """
        
        if model_type == QUBOModel:
            # convert the problem to QUBO form
            return self.to_qubo(backend=backend)
        else:
            raise ValueError("Cannot convert problem to model type {}".format(model_type))
    
    
    def to_qubo(self, backend=None) -> QUBOModel:
        
        locations = list(self.graph.nodes())
        locations_index = {loc: i for i,loc in enumerate(locations)}
        
        distances = nx.adjacency_matrix(self.graph)
        
        # create the variables
        variables = np.array(sp.MatrixSymbol("X", len(locations), len(locations)))
        
        # create the objective function
        cost = np.einsum("ij,ip,jp", distances.todense(), variables, np.roll(variables, -1, axis=1), dtype=object, optimize=True).item()
        
        # column and row constraints
        all_visited_constraint = ((1-variables.sum(axis=0))**2).sum()
        all_steps_constraint = ((1-variables.sum(axis=1))**2).sum()
        
        # compute the total cost
        total_cost = cost + self.penalty*(all_visited_constraint + all_steps_constraint)
        
        # extract the QUBO matrix from the objective function        
        new_cost = 0.0
        
        for term,coeff in total_cost.expand().as_coefficients_dict().items():
            if type(term) == sp.matrices.expressions.matexpr.MatrixElement:
                # square the linear terms
                new_cost += coeff*(term**2)
            else:
                new_cost += coeff*term

        print(new_cost)
        Q = sp.hessian(new_cost.expand(), variables.flatten())

        def from_problem_parameters(params: dict) -> dict:
            """Convert from a dictionary of (node: order) pairs to a dictionary of variable assignments."""
            solution = np.zeros((len(locations), len(locations)), dtype=bool)
            for node, order in params.items():
                solution[locations_index[node], order] = True
            return dict(zip(variables.flatten(), solution.flatten()))
        
        def to_problem_parameters(params: dict) -> dict:
            """Convert from a dictionary of variable assignments to a dictionary of (node: order) pairs."""
            solution = np.array([int(params[loc]) for loc in variables.flatten()]).reshape((len(locations), len(locations)))
            order = np.argmax(solution, axis=1)
            return dict(zip(locations, order))
        
        def initializer():
            return from_problem_parameters(self.initializer())
        
        return QUBOModel(np.array(Q, dtype=float), variables=variables.flatten(), initializer=initializer, to_problem_parameters=to_problem_parameters, from_problem_parameters=from_problem_parameters, backend=backend)
    
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