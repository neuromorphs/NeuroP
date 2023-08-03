from neurop import BaseCompiler, BaseProblem
from neurop.Problem import ILPProblem
from neurop.Model import QUBOModel
from neurop.base import BaseBackend, BaseProblem
from neurop.expansions import expand_integer_to_binary_encoding, substitute_binary_to_integer, substitute_integer_to_binary
from neurop.utils import range_of_polynomial_with_bounded_vars


import numpy as np
import sympy
from optlang import Constraint, Objective, Variable


import random
from collections import defaultdict


class ILPQUBOCompiler(BaseCompiler):
    def __init__(self, problem: ILPProblem, backend: BaseBackend = None, expansion = expand_integer_to_binary_encoding, penalty=1) -> None:
        self.penalty = penalty
        super().__init__(problem, QUBOModel, backend, expansion)

    def model_to_problem_parameters(self, params: dict) -> dict:
        return substitute_binary_to_integer(params, self.substitutions, self.variables)

    def problem_to_model_parameters(self, params: dict) -> dict:
        return substitute_integer_to_binary(params, self.substitutions, self.variables)

    def compile(self) -> QUBOModel:
        """Convert the ILP problem to QUBO form."""

        self.substitutions = dict()
        self.variables = []
        leq_constraints = []
        equal_zero_constraints = []

        def split_constraint(constraint):
            """If the constraint is an equality constraint, store it in equal_zero_constraints, otherwise store it in leq_constraints."""
            # substitute the expanded variables into the constraint expression
            expr = constraint.expression.subs(self.substitutions, simultaneous=False)

            # if both upper and lower bounds coincide, this is not an inequality constraint but rather an equation
            if constraint.ub is not None and constraint.lb is not None and constraint.ub==constraint.lb:
                # compute the value range this expression can take
                computed_lb, computed_ub = range_of_polynomial_with_bounded_vars(expr)

                # check if it is even possible to satisfy this constraint, at all
                if not (computed_lb <= constraint.lb <= constraint.ub <= computed_ub):
                    raise ValueError("Cannot satisfy constraint {}<={}<={} because these terms evaluate to {}, {}, {}".format(constraint.lb, constraint.expression, constraint.ub, computed_lb, expr, computed_ub))

                # add this constraint as an equality with zero rhs
                equal_zero_constraints.append(expr-constraint.ub)

            else:
                # otherwise, add the individual inequality constraints to the list
                if constraint.lb is not None:
                    leq_constraints.append(Constraint(-expr, ub=-constraint.lb))

                if constraint.ub is not None:
                    leq_constraints.append(Constraint(expr, ub=constraint.ub))


        # go through all variables
        for var_name, var in self.problem.variables.iteritems():

            # expand every non-binary variable
            # this addes new constraints to the problem
            if var.type != "binary":
                sub_vars, sub, constraint = self.expansion(var)
                self.substitutions[var] = sub
                self.variables.extend(sub_vars)
                split_constraint(constraint)
            else:
                self.variables.add(var)

        # copy over all the explicit constraints, but substituting in the new variables and breaking them up into positive and negative constraints
        for constraint in self.problem.constraints:
            split_constraint(constraint)


        # iterate through and introduce slack variables for all constraint equations
        # (here, only the upper bound needs to be considered, since range inequalities are already split into two inequalities, 
        # and equality constraints are handled separately)
        s = 0
        while len(leq_constraints) != 0:
            # take out a constraint
            constraint = leq_constraints.pop()

            # compute the value range this expression can take
            computed_lb, computed_ub = range_of_polynomial_with_bounded_vars(constraint.expression)

            # the expression can take on values in the range [computed_lb, computed_ub]
            # the constraint is that the expression is less than or equal to constraint.ub
            # there are three cases:
            # if computed_ub <= constraint.ub, then the constraint is trivially satisfied and can be ignored
            # if computed_lb > constraint.ub, then the constraint is impossible to satisfy and we can stop here
            # if computed_lb <= constraint.ub < computed_ub, then we need to introduce a slack variable

            # check if it is even possible to satisfy this constraint, at all
            if computed_lb>constraint.ub:
                raise ValueError("Cannot satisfy constraint {}<={}<={} because {}>={}".format(computed_lb, constraint.expression, constraint.ub))

            # check if the constraint is trivial and can be ignored
            if computed_ub<=constraint.ub:
                print("Trivial! Contraint {} <= {} <= {}".format(constraint.expression, computed_ub, constraint.ub))
                continue

            # We are now in the case where computed_lb <= constraint.ub < computed_ub
            # Originally, the constraint is as follows:
            #     constraint.expression  <= constraint.ub
            # Now, we can introduce a slack variable s to make this an equality:
            #    constraint.expression + s = constraint.ub, where s >= 0
            #<=> constraint.expression + s - constraint.ub = 0
            # To dimension s, we need to design for the worst case:
            #    s <= constraint.ub - computed_lb <= 2^k for some k

            # compute the next power of 2 for the rhs
            next_power_of_2 = (1<<int(np.ceil(np.log2(float(constraint.ub+1-computed_lb)))))-1

            # do a binary expansion of the slack variable
            slack_vars, slack_eq, slack_constraint = self.expansion(Variable("slack_{}".format(s), type="integer", lb=0, ub=next_power_of_2))
            s+=1

            # add the variables
            self.variables.extend(slack_vars)

            equal_zero_constraints.append(constraint.expression + slack_eq - constraint.ub)


        # copy over the objective, but with the new variables
        self.objective = Objective((1 if self.problem.objective.direction=="min" else -1)*self.problem.objective.expression.subs(self.substitutions), direction="min")


        # at this point, we have reduced the problem to a single minimization objective and a list of equalities with zero rhs

        # create the new combined objective functions using the (now) all binary variables
        new_objective = self.objective.expression

        # penalize deviations from the constraints using a quadratic regularizer
        for constraint in equal_zero_constraints:
            new_objective += self.penalty*constraint**2

        # now, simplify the objective function by expanding it to a multi-linear form
        new_objective = sympy.expand(new_objective)

        # now we need to manually simplify the objective function further
        # by removing powers and introducing new coupling variables for higher order terms
        # to do this, collect coefficients of the objective function in its multi-linear form

        coeffs = defaultdict(lambda: 0)

        # make a manual pass over the objective, apply simplifications
        for term, coeff in new_objective.as_coefficients_dict().items():
            if term.is_Number:
                #coeffs[frozenset()] += coeff
                # ignore constant terms
                continue
            elif term.is_Symbol:
                coeffs[frozenset([term])] += coeff
            elif term.is_Pow:
                # get the variable without powers
                coeffs[frozenset([term.args[0]])] += coeff
            elif term.is_Mul:
                # get the set of variables without powers
                coeffs[frozenset(var if var.is_Symbol else var.args[0] for var in term.args)] += coeff

        # now, keep iterating over the coefficients until we cannot simplify any further
        M = 1 + 2 * sum(abs(coeff) for coeff in coeffs.values())

        keep_going = True
        while keep_going:
            keep_going = False

            for term, coeff in coeffs.items():
                if len(term)>2 and coeff != 0:
                    keep_going = True

                    # get the first two variables
                    x1, x2 = term[:2]

                    # add a coupling variable to replace x1*x2*... with ...*z, where z = x1*x2
                    z = Variable("{}_{}".format(term[0].name, term[1].name), type="binary")
                    self.variables.append(z)

                    # add a M*x1*x2 penalty term to the objective
                    coeffs[frozenset(x1,x2)] += M

                    # add a -2M*x1*z and -2M*x2*z penalty term to the objective
                    coeffs[frozenset(x1,z)] = coeffs[frozenset(x2,z)] = -2*M

                    # add a 3M*z penalty term to the objective
                    coeffs[frozenset(z)] = 3*M

                    # remove every occurrence of the old term x1*x2 from all other terms of the objective                
                    for term2, coeff2 in coeffs.items():
                        if x1 in term2 and x2 in term2:
                            coeffs[term2] = 0
                            coeffs[term2.difference((x1,x2)).union((z,))] = coeff2

        # now transform the coefficients into a QUBO matrix        

        variable_index = {var: i for i, var in enumerate(self.variables)}

        # construct the actual QUBO matrix Q
        Q = np.zeros((len(self.variables),len(self.variables)), dtype=int)
        for term, coeff in coeffs.items():
            if coeff == 0:
                continue

            if len(term)==0:
                # constant term can be ignored
                continue
            elif len(term)==1:
                (v1,) = (v2,) = term
            elif len(term)==2:
                v1,v2 = term
            else:
                raise ValueError("Cannot handle terms with more than 2 variables! (This should have been taken care of by the model reduction operation.)")

            Q[variable_index[v1], variable_index[v2]] += coeff
            Q[variable_index[v2], variable_index[v1]] += coeff

        # define how to convert between the problem parameters and the QUBO parameters

        def initializer():
            params = self.problem_to_model_parameters(self.problem.initializer())
            for var in self.variables:
                if var not in params:
                    params[var] = random.randint(0,1)
            return params

        return QUBOModel(np.array(Q, dtype=int), variables=self.variables, initializer=initializer)