"""Integer linear programming (ILP) problems

.. autoclass:: neurop.Problem.ILPProblem.ILPProblem
"""

import optlang
import numpy as np
import sympy
from optlang import Model, Variable, Constraint, Objective

from neurop.Problem.Problem import Problem
from neurop.utils import binary_expansion, range_of_polynomial_with_bounded_vars, substitute_binary_to_integer, substitute_integer_to_binary


class ILPProblem(Problem):
    """Description of ILP Problems for conversion to QUBO."""
    
    def __init__(self, model: Model):
        
        self.model = model
        
        super().__init__()
    
    
    def to_qubo(self):
        self.substitutions = dict()
        self.variables = []
        leq_constraints = []
        self.constraints = []

        # go through all self.variables
        for var_name, var in self.model.variables.iteritems():
            
            # expand every non-binary variable
            if var.type != "binary":
                sub_vars, sub, constraint = binary_expansion(var)
                self.substitutions[var] = sub
                self.variables.extend(sub_vars)
                leq_constraints.append(constraint)
            else:
                self.variables.add(var)
            
        # copy over all the explicit constraints, but substituting in the new self.variables and breaking them up into positive and negative constraints
        for constraint in self.model.constraints:
            # substitute the expanded self.variables into the constraint expression
            expr = constraint.expression.subs(self.substitutions, simultaneous=False)
            
            # if both upper and lower bounds coincide, this is not an inequality constraint but rather an equation
            if constraint.ub is not None and constraint.lb is not None and constraint.ub==constraint.lb:
                # compute the value range this expression can take
                computed_lb, computed_ub = range_of_polynomial_with_bounded_vars(expr)    
            
                # check if it is even possible to satisfy this constraint, at all
                if not (computed_lb <= constraint.lb <= computed_ub):
                    raise ValueError("Cannot satisfy constraint {}<={}<={} because {}<={}<={}".format(constraint.lb, constraint.expression, constraint.ub, computed_lb, expr, computed_ub))

                self.constraints.append(constraint.expression-constraint.ub)
            else:
                # otherwise, add the individual constraints to the list
                if constraint.ub is not None:
                    leq_constraints.append(Constraint(expr, ub=constraint.ub))
                if constraint.lb is not None:
                    leq_constraints.append(Constraint(-expr, ub=-constraint.lb))


        s = 0
        # introduce slack self.variables for all constraint equations
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
            
            # add the self.variables
            self.variables.extend(slack_vars)
            
            # add the equation "constraint.expression + c - computed_lb - s = 0"; the inequality should now be trivial
            self.constraints.append(constraint.expression + c - computed_lb - slack_eq)


        # copy over the objective, but with the new self.variables
        self.objective = Objective((1 if self.model.objective.direction=="min" else -1)*self.model.objective.expression.subs(self.substitutions), direction="min")
        
        self.Q = sympy.zeros(len(self.variables),len(self.variables))

        # make matrix with coefficients on the diagonal
        self.Q += sympy.diag(*sympy.Matrix([self.objective.expression]).jacobian(self.variables))
        penalty=10
        zero_equations_mat = sympy.Matrix(self.constraints)
        variables_mat = sympy.Matrix(self.variables)
        # get the coefficients of each equation
        zero_equations_Jac = zero_equations_mat.jacobian(self.variables)

        # Add quadratic terms from the constraints:
        self.Q += penalty*zero_equations_Jac.T*zero_equations_Jac
        # Q += penalty*sympy.Add(*(sympy.hessian(eq**2, variables)  for eq in zero_equations))

        # Get the constant term (= the rest)
        cs = zero_equations_mat-zero_equations_Jac*variables_mat

        # Add the linear terms from the constraints:
        self.Q += sympy.diag(*(penalty*2*(cs.T*zero_equations_Jac)))
        
        return self.Q
    
    def parameters_from_qubo(self, params: np.ndarray) -> dict:
        return substitute_binary_to_integer(params, self.substitutions, self.variables)
    
    def parameters_to_qubo(self, params: dict) -> np.ndarray:
        return substitute_integer_to_binary(params, self.substitutions, self.variables)
    
    def evaluate_objective(self, params_bin: np.ndarray) -> float:
        return self.objective.expression.subs(dict(zip(self.variables, params_bin)))