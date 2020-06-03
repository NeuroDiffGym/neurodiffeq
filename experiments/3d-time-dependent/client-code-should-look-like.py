import torch
from torch import nn, optim
from neurodiffeq import diff
from neurodiffeq.networks import FCNN
from neurodiffeq.temporal import generator_3dspatial_body, generator_3dspatial_surface, generator_temporal
from neurodiffeq.temporal import FirstOrderInitialCondition, BoundaryCondition
from neurodiffeq.temporal import SingleNetworkApproximator3DSpatialTemporal
from neurodiffeq.temporal import MonitorMinimal
from neurodiffeq.temporal import _solve_3dspatial_temporal

def some_3d_time_dependent_pde(u, x, y, z, t):
    return diff(u, x) + diff(u, y) + diff(u, z) + diff(u, t) ...

# e.g. make u(x, y, z, t) = x^2 +y^2 + z^2 at the boundary
boundary_surface_1 = BoundaryCondition(
    form=lambda u, x, y, z: u - (x**2 + y**2 + z**2),
    points_generator=generator_3dspatial_surface( ... )
)
boundary_surface_2 = BoundaryCondition(
    form=lambda u, x, y, z: u - (x**2 + y**2 + z**2),
    points_generator=generator_3dspatial_surface( ... )
)
boundary_surface_3 = BoundaryCondition(
    form=lambda u, x, y, z: u - (x**2 + y**2 + z**2),
    points_generator=generator_3dspatial_surface( ... )
)

fcnn = FCNN(
    n_input_units=4,
    n_output_units=1,
    n_hidden_units=32,
    n_hidden_layers=1,
    actv=nn.Tanh
)
fcnn_approximator = SingleNetworkApproximator3DSpatialTemporal(
    single_network=fcnn,
    pde=some_3d_time_dependent_pde,
    boundary_conditions=[
        boundary_surface_1,
        boundary_surface_2,
        boundary_surface_3,
    ]
)
adam = optim.Adam(fcnn_approximator.parameters(), lr=0.001)

train_gen_spatial = generator_3dspatial_body(...)
train_gen_temporal = generator_temporal(...)
valid_gen_spatial = generator_3dspatial_body(...)
valid_gen_temporal = generator_temporal(...)

some_3d_time_dependent_pde_solution, _ = _solve_3dspatial_temporal(
    train_generator_spatial=train_gen_spatial,
    train_generator_temporal=train_gen_temporal,
    valid_generator_spatial=valid_gen_spatial,
    valid_generator_temporal=valid_gen_temporal,
    approximator=fcnn_approximator,
    optimizer=adam,
    batch_size=512,
    max_epochs=5000,
    shuffle=True,
    metrics={},
    monitor=MonitorMinimal(check_every=10)
)