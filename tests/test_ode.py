import numpy as np
from numpy import isclose

from neurodiffeq import diff
from neurodiffeq.ode import InitialValueProblem, solve_system

def test_ode_system_parametric_circle():
    
    parametric_circle = lambda x1, x2, t : [diff(x1, t) - x2, 
                                            diff(x2, t) + x1]
    init_vals_pc = [
        InitialValueProblem(t_0=0.0, x_0=0.0),
        InitialValueProblem(t_0=0.0, x_0=1.0)
    ]
    
    solution_pc, _ = solve_system(ode_system=parametric_circle, 
                                  conditions=init_vals_pc, 
                                  t_min=0.0, t_max=2*np.pi)
    
    ts = np.linspace(0, 2*np.pi, 100)
    x1_net, x2_net = solution_pc(ts)
    x1_ana, x2_ana = np.sin(ts), np.cos(ts)
    assert isclose(x1_net, x1_ana, atol=0.1).all()
    assert isclose(x2_net, x2_ana, atol=0.1).all()