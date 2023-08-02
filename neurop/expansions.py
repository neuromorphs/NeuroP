import numpy as np
from optlang import Constraint, Variable
import sympy

def expand_integer_to_binary_encoding(var: Variable):
    """Expands an integer variable with given lower and upper bound to a binary encoding.
    
    Args:
        var (Variable): The variable to expand.
    
    Raises:
        ValueError: if the variables is not an integer variable with defined bounds
        
    Returns:
        sub_vars: The list of binary variables that represent the binary expansion of the integer variable.
        sub: The linear combination of the binary variables that represent the binary expansion of the integer variable.
        constraint: The constraint that enforces the bounds of the integer variable on the binary expansion.
    """
    
    if var.type != "integer" and var.type != "binary":
        raise ValueError("Variable {} is not an integer, but this package can only handle integer variables!".format(var.name))

    # check if the variable has bounds - if so, add them as constraints, otherwise complain
    if var.lb is None:
        raise ValueError("Variable {} is does not have a defined lower bound!".format(var.name))
    if var.ub is None:
        raise ValueError("Variable {} is does not have a defined upper bound!".format(var.name))

    # compute value range and required number of bits to represent it
    num_bits = (int(np.ceil(np.log2(float(var.ub)-float(var.lb+1)))))

    # add binary variables
    sub_vars = []
    sub = var.lb
    for i in range(num_bits):
        # add the variable for the binary expansion
        sub_var = Variable("{}_{}".format(var.name, i), type="binary")
        sub_vars.append(sub_var)

        # add the monomial term to the expansion of the integer variable
        sub += (2**i) * sub_var

    # add the upper bound as a constraint on the binary expansion of the integer variable
    # the lower bound is automatically enforced by the expansion
    constraint = Constraint(sub, lb=None, ub=var.ub)

    return sub_vars, sub, constraint

def expand_integer_to_onehot_encoding(var: Variable):
    """_summary_

    Args:
        var (Variable): _description_

    Raises:
        ValueError: if the variables is not an integer variable with defined bounds

    Returns:
        _type_: _description_
    """
    if var.type != "integer" and var.type != "binary":
        raise ValueError("Variable {} is not an integer, but this package can only handle integer variables!".format(var.name))

    # check if the variable has bounds - if so, add them as constraints, otherwise complain
    if var.lb is None:
        raise ValueError("Variable {} is does not have a defined lower bound!".format(var.name))
    if var.ub is None:
        raise ValueError("Variable {} is does not have a defined upper bound!".format(var.name))

    # compute value range and required number of bits to represent it (one per value)
    num_bits = int(np.ceil(float(var.ub)-float(var.lb+1)))

    # add binary variables
    sub_vars = []
    sub = var.lb
    s = 0
    for i in range(num_bits):
        # add the variable for the one-hot expansion
        sub_var = Variable("{}_{}".format(var.name, i), type="binary")
        sub_vars.append(sub_var)

        # add the monomial term to the expansion of the integer variable
        sub += i * sub_var
        s += sub_var

    # one-hot encoding requires that exactly one bit is set
    constraint = Constraint(s, lb=1, ub=1)

    return sub_vars, sub, constraint


def substitute_integer_to_binary(values: dict, substitutions: dict, binary_variables):
    substitutions_mat = sympy.Matrix(list(substitutions.values()))
    variables_mat = sympy.Matrix(binary_variables)
    M = substitutions_mat.jacobian(binary_variables)
    b = substitutions_mat-M*variables_mat
    var_index = dict(zip(substitutions.keys(),range(len(substitutions))))
    ret = np.zeros((M.shape[1],), dtype=bool)
    for var,val in values.items():
        i = var_index[var]
        grad = M[i,:]
        bb = b[i]
        ret += np.array([(g != 0 and (val-bb) % (2*g) >= g) for g in grad], dtype=bool)
    return dict(zip(binary_variables, ret))


def substitute_binary_to_integer(values: dict, substitutions, binary_variables):
    # substitutions_mat = sympy.Matrix(list(substitutions.values()))
    # variables_mat = sympy.Matrix(variables)
    # M = substitutions_mat.jacobian(variables)
    # b = substitutions_mat-M*variables_mat
    s = dict(((var, values[var]) for var in binary_variables))
    return dict( (k,eq.subs(s)) for k,eq in substitutions.items())