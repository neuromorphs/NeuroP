import sympy

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
