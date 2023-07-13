import sympy
import numpy as np
from optlang import Variable, Constraint

def range_of_polynomial_with_bounded_vars(expr):
    expr = expr.expand()
    if not expr.is_polynomial():
        raise ValueError("The expression {} is not polynomial!".format(expr))

    min_v = 0
    max_v = 0

    # go through sum terms
    for summand in sympy.Add.make_args(expr):
        # get coefficient and product of variables
        c,vs = summand.as_coeff_Mul()
        
        min_fac = c
        max_fac = c
        # go through all factors
        for var,exp in vs.as_powers_dict().items():
            # get min and max value of factor
            val0 = var.lb**exp
            val1 = var.ub**exp
            new_vals= (min_fac*val0, min_fac*val1, max_fac*val0, max_fac*val1)
            min_fac = min(new_vals)
            max_fac = max(new_vals)
        
        min_v += min_fac
        max_v += max_fac
    
    return (min_v, max_v)

def binary_expansion(var: Variable):
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
    
    # add the upper and lower bounds as a constraint on the binary expansion of the integer variable
    constraint = Constraint(sub, lb=var.lb, ub=var.ub)
    
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

# binary = substitute_integer_to_binary({x1:7, x2:1, x3: 31}, substitutions, variables)
# substitute_binary_to_integer(binary, substitutions, variables)