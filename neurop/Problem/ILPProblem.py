"""Integer linear programming (ILP) problems

.. autoclass:: neurop.Problem.ILPProblem
"""

from collections.abc import Iterable
import numpy as np
import sympy
from optlang import Model as OptProblem, Variable, Constraint, Objective

from neurop import BaseProblem
from neurop import BaseModel
from neurop.Model import QUBOModel
from neurop.utils import binary_expansion, range_of_polynomial_with_bounded_vars, substitute_binary_to_integer, substitute_integer_to_binary


class ILPProblem(BaseProblem):
    """Description of ILP Problems for conversion to QUBO."""
    
    def __init__(self, problem: OptProblem, initializer=None, penalty=10):
        """Initializes the ILP problem.

        Args:
            problem (OptProblem): the optlang problem
            initializer (_type_, optional): function to initialize the parameters of the problem. Defaults to None, which means that parameters will be randomly initialized.
        """
        self.problem = problem
        self.penalty = penalty
        
        if initializer is None:
            initializer = lambda: dict((var, np.random.randint(var.lb, var.ub+1)) for var in self.problem.variables)
        
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
        substitutions = dict()
        variables = []
        leq_constraints = []
        constraints = []

        # go through all variables
        for var_name, var in self.problem.variables.iteritems():
            
            # expand every non-binary variable
            if var.type != "binary":
                sub_vars, sub, constraint = binary_expansion(var)
                substitutions[var] = sub
                variables.extend(sub_vars)
                leq_constraints.append(constraint)
            else:
                variables.add(var)
            
        # copy over all the explicit constraints, but substituting in the new variables and breaking them up into positive and negative constraints
        for constraint in self.problem.constraints:
            # substitute the expanded variables into the constraint expression
            expr = constraint.expression.subs(substitutions, simultaneous=False)
            
            # if both upper and lower bounds coincide, this is not an inequality constraint but rather an equation
            if constraint.ub is not None and constraint.lb is not None and constraint.ub==constraint.lb:
                # compute the value range this expression can take
                computed_lb, computed_ub = range_of_polynomial_with_bounded_vars(expr)    
            
                # check if it is even possible to satisfy this constraint, at all
                if not (computed_lb <= constraint.lb <= computed_ub):
                    raise ValueError("Cannot satisfy constraint {}<={}<={} because {}<={}<={}".format(constraint.lb, constraint.expression, constraint.ub, computed_lb, expr, computed_ub))

                constraints.append(constraint.expression-constraint.ub)
            else:
                # otherwise, add the individual constraints to the list
                if constraint.ub is not None:
                    leq_constraints.append(Constraint(expr, ub=constraint.ub))
                if constraint.lb is not None:
                    leq_constraints.append(Constraint(-expr, ub=-constraint.lb))


        s = 0
        # introduce slack variables for all constraint equations
        while len(leq_constraints) != 0:
            # take out a constraint
            constraint = leq_constraints.pop()
            
            # compute the value range this expression can take
            computed_lb, computed_ub = range_of_polynomial_with_bounded_vars(constraint.expression)    
            
            # check if it is even possible to satisfy this constraint, at all
            if computed_lb>constraint.ub:
                raise ValueError("Cannot satisfy constraint {}<={}<={} because {}>={}".format(computed_lb, constraint.expression, constraint.ub))

            # check if the constraint is trivial and can be ignored
            if computed_ub<=constraint.ub:
                print("Trivial! Contraint {} <= {} <= {}".format(constraint.expression, computed_ub, constraint.ub))
                continue

            #     constraint.expression <= constraint.ub
            # <=> constraint.expression - computed_lb + c <= constraint.ub - computed_lb + c
            # <=> constraint.expression - computed_lb + c = s;  0 <= s <= constraint.ub - computed_lb + c
            # <=> constraint.expression - computed_lb + c - s = 0; s <= constraint.ub - computed_lb + c
            
            # compute the next power of 2 for the rhs
            next_power_of_2 = (1<<int(np.ceil(np.log2(float(constraint.ub+1-computed_lb)))))-1
            
            # compute the offset needed to make the rhs a power of two
            c = next_power_of_2 - (constraint.ub - computed_lb)

            # do a binary expansion of the slack variable
            slack_vars, slack_eq, slack_constraint = binary_expansion(Variable("s_{}".format(s), type="integer", lb=0, ub=next_power_of_2))
            s+=1
            
            # add the variables
            variables.extend(slack_vars)
            
            # add the equation "constraint.expression + c - computed_lb - s = 0"; the inequality should now be trivial
            constraints.append(constraint.expression + c - computed_lb - slack_eq)


        # copy over the objective, but with the new variables
        self.objective = Objective((1 if self.problem.objective.direction=="min" else -1)*self.problem.objective.expression.subs(substitutions), direction="min")
        
        Q = sympy.zeros(len(variables),len(variables))

        # make matrix with coefficients on the diagonal
        Q += sympy.diag(*sympy.Matrix([self.objective.expression]).jacobian(variables))
        zero_equations_mat = sympy.Matrix(constraints)
        variables_mat = sympy.Matrix(variables)
        # get the coefficients of each equation
        zero_equations_Jac = zero_equations_mat.jacobian(variables)

        # Add quadratic terms from the constraints:
        Q += self.penalty*zero_equations_Jac.T*zero_equations_Jac
        # Q += penalty*sympy.Add(*(sympy.hessian(eq**2, variables)  for eq in zero_equations))

        # Get the constant term (= the rest)
        cs = zero_equations_mat-zero_equations_Jac*variables_mat

        # Add the linear terms from the constraints:
        Q += sympy.diag(*(self.penalty*2*(cs.T*zero_equations_Jac)))
        
            
        def to_problem_parameters(params: dict) -> dict:
            return substitute_binary_to_integer(params, substitutions, variables)
        
        def from_problem_parameters(params: dict) -> dict:
            return substitute_integer_to_binary(params, substitutions, variables)
        
        def initializer():
            return from_problem_parameters(self.initializer())
        
        return QUBOModel(np.array(Q, dtype=int), variables=variables, initializer=initializer, to_problem_parameters=to_problem_parameters, from_problem_parameters=from_problem_parameters, backend=backend)
    
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