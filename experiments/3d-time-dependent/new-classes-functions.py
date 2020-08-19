# the model and the loss
# should be similar to `SingleNetworkApproximator2DSpatialTemporal`
class SingleNetworkApproximator3DSpatialTemporal(Approximator):
    def __init__(self, single_network, pde, initial_condition, boundary_conditions, boundary_strictness=1.):
        self.single_network = single_network
        self.pde = pde
        self.u0 = initial_condition.u0
        self.u0dot = initial_condition.u0dot if hasattr(initial_condition, 'u0dot') else None
        self.boundary_conditions = boundary_conditions
        self.boundary_strictness = boundary_strictness

    def __call__(self, xx, yy, zz, tt):
        xx = torch.unsqueeze(xx, dim=1)
        yy = torch.unsqueeze(yy, dim=1)
        zz = torch.unsqueeze(zz, dim=1)
        tt = torch.unsqueeze(tt, dim=1)
        xyzt = torch.cat((xx, yy, zz, tt), dim=1)
        uu_raw = self.single_network(xyzt)
        return uu_raw  # TODO reparameterize the output uu_raw (use the information carried by self.u0 and self.u0dot)

    def parameters(self):
        return self.single_network.parameters()

    def calculate_loss(self, xx, yy, zz, tt, x, y, z, t):
        uu = self.__call__(xx, yy, zz, tt)
        equation_mse = torch.mean(self.pde(uu, xx, yy, zz, tt)**2)
        return equation_mse  # TODO plus the regularization term (use the information carried by self.boundary_conditions)

    def calculate_metrics(self, xx, yy, zz, tt, metrics):
        uu = self.__call__(xx, yy, zz)

        return {
            metric_name: metric_func(*uu, xx, yy, zz)
            for metric_name, metric_func in metrics.items()
        }


# A generator for generating 3D points in the problem domain: yield (xx, yy, zz) where xx, yy, zz are 1-D tensors
def generator_3dspatial_body(...):
    pass

# A generator for generating 3D points on the boundary: yield (xx, yy, zz) where xx, yy, zz are 1-D tensors
def generator_3dspatial_surface(...):
    pass


def _solve_3dspatial_temporal(
    train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
    approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor
):
    return _solve_spatial_temporal(
        train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
        approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,
        train_routine=_train_3dspatial_temporal, valid_routine=_valid_3dspatial_temporal
    )


# the logic for training one epoch
# should be similar to `_train_2dspatial_temporal`
def _train_3dspatial_temporal(train_generator_spatial, train_generator_temporal, approximator, optimizer, metrics, shuffle, batch_size):
    
    # generate x, y, z dimensions from train_generator_spatial
    # generate time slices from train_generator_temporal
    # Do a cartesian product of the above two to make the training set
    # pass the training points to the neural network in batches
    # for each batch calculate the loss and update the weights
    # calculate the loss custom metrics of this epoch

    return epoch_loss, epoch_metrics


# the logic for training one epoch
# should be similar to `_valid_2dspatial_temporal`
def _valid_3dspatial_temporal(valid_generator_spatial, valid_generator_temporal, approximator, metrics):

    # generate x, y, z dimensions from train_generator_spatial
    # generate time slices from train_generator_temporal
    # Do a cartesian product of the above two to make the training set
    # calculate the loss custom metrics of this epoch

    return epoch_loss, epoch_metrics
