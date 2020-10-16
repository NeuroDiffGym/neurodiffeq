import sys
import torch
import numpy as np
from pytest import raises
# atomic-ish generator classes
from neurodiffeq.generators import Generator1D
from neurodiffeq.generators import Generator2D
from neurodiffeq.generators import Generator3D
from neurodiffeq.generators import GeneratorSpherical
# complex generator classes
from neurodiffeq.generators import ConcatGenerator
from neurodiffeq.generators import StaticGenerator
from neurodiffeq.generators import PredefinedGenerator
from neurodiffeq.generators import TransformGenerator
from neurodiffeq.generators import EnsembleGenerator
from neurodiffeq.generators import FilterGenerator
from neurodiffeq.generators import ResampleGenerator
from neurodiffeq.generators import BatchGenerator

MAGIC = 42
torch.manual_seed(MAGIC)
np.random.seed(MAGIC)


def _check_shape_and_grad(generator, target_size, *xs):
    if target_size is not None:
        if target_size != generator.size:
            print(f"size mismatch {target_size} != {generator.size}", file=sys.stderr)
            return False
    for x in xs:
        if x.shape != (generator.size,):
            print(f"Bad shape: {x.shape} != {generator.size}", file=sys.stderr)
            return False
        if not x.requires_grad:
            print(f"Doesn't require grad: {x}", file=sys.stderr)
            return False
    return True


def _check_boundary(xs, xs_min, xs_max):
    for x, x_min, x_max in zip(xs, xs_min, xs_max):
        if x_min is not None and (x < x_min).any():
            print(f"Lower than minimum: {x} <= {x_min}", file=sys.stderr)
            return False
        if x_max is not None and (x > x_max).any():
            print(f"Higher than maximum: {x} >= {x_max}", file=sys.stderr)
            return False
    return True


def _check_iterable_equal(x, y, eps=1e-5):
    for a, b in zip(x, y):
        if abs(float(a) - float(b)) > eps:
            print(f"Different values: {a} != {b}", file=sys.stderr)
            return False
    return True


def test_generator1d():
    size = 32
    generator = Generator1D(size=size, t_min=0.0, t_max=2.0, method='uniform')
    x = generator.getter()
    assert _check_shape_and_grad(generator, size, x)

    generator = Generator1D(size=size, t_min=0.0, t_max=2.0, method='equally-spaced')
    x = generator.getter()
    assert _check_shape_and_grad(generator, size, x)

    generator = Generator1D(size=size, t_min=0.0, t_max=2.0, method='equally-spaced-noisy')
    x = generator.getter()
    assert _check_shape_and_grad(generator, size, x)

    generator = Generator1D(size=size, t_min=np.log10(0.1), t_max=np.log10(2.0), method='log-spaced')
    x = generator.getter()
    assert _check_shape_and_grad(generator, size, x)

    generator = Generator1D(size=size, t_min=np.log10(0.1), t_max=np.log10(2.0), method='log-spaced-noisy')
    x = generator.getter()
    assert _check_shape_and_grad(generator, size, x)

    generator = Generator1D(size=size, t_min=np.log10(0.1), t_max=np.log10(2.0), method='log-spaced-noisy',
                            noise_std=0.01)
    x = generator.getter()
    assert _check_shape_and_grad(generator, size, x)

    with raises(ValueError):
        generator = Generator1D(size=size, t_min=0.0, t_max=2.0, method='magic')
    print('ExampleGenerator test passed.')


def test_generator2d():
    grid = (5, 6)
    size = grid[0] * grid[1]
    x_min, x_max = 0.0, 1.0
    y_min, y_max = -1.0, 0.0
    x_std, y_std = 0.05, 0.06

    generator = Generator2D(grid=grid, xy_min=(x_min, y_min), xy_max=(x_max, y_max), method='equally-spaced-noisy')
    x, y = generator.getter()
    assert _check_shape_and_grad(generator, size, x, y)

    generator = Generator2D(grid=grid, xy_min=(x_min, y_min), xy_max=(x_max, y_max), method='equally-spaced-noisy',
                            xy_noise_std=(x_std, y_std))
    x, y = generator.getter()
    assert _check_shape_and_grad(generator, size, x, y)

    generator = Generator2D(grid=grid, xy_min=(x_min, y_min), xy_max=(x_max, y_max), method='equally-spaced')
    x, y = generator.getter()
    assert _check_shape_and_grad(generator, size, x, y)
    assert _check_boundary((x, y), (x_min, y_min), (x_max, y_max))


def test_generator3d():
    grid = (5, 6, 7)
    size = grid[0] * grid[1] * grid[2]
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 1.0, 2.0
    z_min, z_max = 2.0, 3.0

    generator = Generator3D(grid=grid, xyz_min=(x_min, y_min, z_min), xyz_max=(x_max, y_max, z_max),
                            method='equally-spaced-noisy')
    x, y, z = generator.getter()
    assert _check_shape_and_grad(generator, size, x, y, z)

    generator = Generator3D(grid=grid, xyz_min=(x_min, y_min, z_min), xyz_max=(x_max, y_max, z_max),
                            method='equally-spaced')
    x, y, z = generator.getter()
    assert _check_shape_and_grad(generator, size, x, y, z)
    assert _check_boundary((x, y, z), (x_min, y_min, z_min), (x_max, y_max, z_max))


def test_generator_spherical():
    size = 64
    r_min, r_max = 0.0, 1.0

    generator = GeneratorSpherical(size, r_min=r_min, r_max=r_max, method='equally-spaced-noisy')
    r, theta, phi = generator.get_examples()
    assert _check_shape_and_grad(generator, size, r, theta, phi)
    assert _check_boundary((r, theta, phi), (r_min, 0.0, 0.0), (r_max, np.pi, np.pi * 2))

    generator = GeneratorSpherical(size, r_min=r_min, r_max=r_max, method='equally-radius-noisy')
    r, theta, phi = generator.get_examples()
    assert _check_shape_and_grad(generator, size, r, theta, phi)
    assert _check_boundary((r, theta, phi), (r_min, 0.0, 0.0), (r_max, np.pi, np.pi * 2))


def test_concat_generator():
    size1, size2 = 10, 20
    t_min, t_max = 0.5, 1.5
    generator1 = Generator1D(size1, t_min=t_min, t_max=t_max)
    generator2 = Generator1D(size2, t_min=t_min, t_max=t_max)
    concat_generator = ConcatGenerator(generator1, generator2)
    x = concat_generator.get_examples()
    assert _check_shape_and_grad(concat_generator, size1 + size2, x)

    grid1 = (4, 4, 4)
    size1, size2, size3 = grid1[0] * grid1[1] * grid1[2], 100, 200
    generator1 = Generator3D(grid=grid1)
    generator2 = GeneratorSpherical(size2)
    generator3 = GeneratorSpherical(size3)
    concat_generator = ConcatGenerator(generator1, generator2, generator3)
    r, theta, phi = concat_generator.get_examples()
    assert _check_shape_and_grad(concat_generator, size1 + size2 + size3, r, theta, phi)

    added_generator = generator1 + generator2 + generator3
    r, theta, phi = added_generator.get_examples()
    assert _check_shape_and_grad(added_generator, size1 + size2 + size3, r, theta, phi)


def test_static_generator():
    size = 100
    generator = Generator1D(size)
    static_generator = StaticGenerator(generator)
    x1 = static_generator.get_examples()
    x2 = static_generator.get_examples()
    assert _check_shape_and_grad(generator, size)
    assert _check_shape_and_grad(static_generator, size, x1, x2)
    assert (x1 == x2).all()

    size = 100
    generator = GeneratorSpherical(size)
    static_generator = StaticGenerator(generator)
    r1, theta1, phi1 = static_generator.get_examples()
    r2, theta2, phi2 = static_generator.get_examples()
    assert _check_shape_and_grad(generator, size)
    assert _check_shape_and_grad(static_generator, size, r1, theta1, phi1, r2, theta2, phi2)
    assert (r1 == r2).all() and (theta1 == theta2).all() and (phi1 == phi2).all()


def test_predefined_generator():
    size = 100

    old_x = torch.arange(size, dtype=torch.float, requires_grad=False)
    predefined_generator = PredefinedGenerator(old_x)
    x = predefined_generator.get_examples()
    assert _check_shape_and_grad(predefined_generator, size, x)
    assert _check_iterable_equal(old_x, x)

    old_x = torch.arange(size, dtype=torch.float, requires_grad=False)
    old_y = torch.arange(size, dtype=torch.float, requires_grad=True)
    old_z = torch.arange(size, dtype=torch.float, requires_grad=False)
    predefined_generator = PredefinedGenerator(old_x, old_y, old_z)
    x, y, z = predefined_generator.get_examples()
    assert _check_shape_and_grad(predefined_generator, size, x, y, z)
    assert _check_iterable_equal(old_x, x)
    assert _check_iterable_equal(old_y, y)
    assert _check_iterable_equal(old_z, z)

    x_list = [i * 2.0 for i in range(size)]
    y_tuple = tuple([i * 3.0 for i in range(size)])
    z_array = np.array([i * 4.0 for i in range(size)]).reshape(-1, 1)
    w_tensor = torch.arange(size, dtype=torch.float)
    predefined_generator = PredefinedGenerator(x_list, y_tuple, z_array, w_tensor)
    x, y, z, w = predefined_generator.get_examples()
    assert _check_shape_and_grad(predefined_generator, size, x, y, z, w)
    assert _check_iterable_equal(x_list, x)
    assert _check_iterable_equal(y_tuple, y)
    assert _check_iterable_equal(z_array, z)
    assert _check_iterable_equal(w_tensor, w)


def test_transform_generator():
    size = 100
    x = np.arange(0, size, dtype=np.float32)
    x_expected = np.sin(x)
    generator = PredefinedGenerator(x)
    transform_generator = TransformGenerator(generator, [torch.sin])
    x = transform_generator.get_examples()
    assert _check_shape_and_grad(transform_generator, size, x)
    assert _check_iterable_equal(x, x_expected)

    x = np.arange(0, size, dtype=np.float32)
    y = np.arange(0, size, dtype=np.float32)
    z = np.arange(0, size, dtype=np.float32)
    x_expected = np.sin(x)
    y_expected = y
    z_expected = -z
    generator = PredefinedGenerator(x, y, z)
    transform_generator = TransformGenerator(generator, [torch.sin, None, lambda a: -a])
    x, y, z = transform_generator.get_examples()
    assert _check_shape_and_grad(transform_generator, size, x, y, z)
    assert _check_iterable_equal(x, x_expected)
    assert _check_iterable_equal(y, y_expected)
    assert _check_iterable_equal(z, z_expected)


def test_ensemble_generator():
    size = 100

    generator1 = Generator1D(size)
    ensemble_generator = EnsembleGenerator(generator1)
    x = ensemble_generator.get_examples()
    assert _check_shape_and_grad(ensemble_generator, size, x)

    old_x = torch.rand(size)
    old_y = torch.rand(size)
    old_z = torch.rand(size)
    generator1 = PredefinedGenerator(old_x)
    generator2 = PredefinedGenerator(old_y)
    generator3 = PredefinedGenerator(old_z)
    ensemble_generator = EnsembleGenerator(generator1, generator2, generator3)
    x, y, z = ensemble_generator.get_examples()
    assert _check_shape_and_grad(ensemble_generator, size, x, y, z)
    assert _check_iterable_equal(old_x, x)
    assert _check_iterable_equal(old_y, y)
    assert _check_iterable_equal(old_z, z)

    old_x = torch.rand(size)
    old_y = torch.rand(size)
    generator1 = PredefinedGenerator(old_x)
    generator2 = PredefinedGenerator(old_y)
    product_generator = generator1 * generator2
    x, y = product_generator.get_examples()
    assert _check_shape_and_grad(product_generator, size, x, y)
    assert _check_iterable_equal(old_x, x)
    assert _check_iterable_equal(old_y, y)


def test_filter_generator():
    grid = (10, 10)
    size = 100

    x = [i * 1.0 for i in range(size)]
    filter_fn = lambda a: (a[0] % 2 == 0)
    filter_fn_2 = lambda a: (a % 2 == 0)
    x_expected = filter(filter_fn_2, x)

    generator = PredefinedGenerator(x)
    filter_generator = FilterGenerator(generator, filter_fn=filter_fn, update_size=True)
    x = filter_generator.get_examples()
    assert _check_shape_and_grad(filter_generator, size // 2, x)
    assert _check_iterable_equal(x_expected, x)

    x = [i * 1.0 for i in range(size)]
    y = [-i * 1.0 for i in range(size)]
    filter_fn = lambda ab: (ab[0] % 2 == 0) & (ab[1] > -size / 2)
    x_expected, y_expected = list(zip(*filter(filter_fn, zip(x, y))))
    generator = PredefinedGenerator(x, y)
    filter_generator = FilterGenerator(generator, filter_fn)
    x, y = filter_generator.get_examples()
    assert _check_shape_and_grad(filter_generator, size // 4, x, y)
    assert _check_iterable_equal(x_expected, x)
    assert _check_iterable_equal(y_expected, y)

    generator = Generator2D(grid)
    filter_fn = lambda ab: (ab[0] > 0.5) & (ab[1] < 0.5)
    filter_generator = FilterGenerator(generator, filter_fn)
    for _ in range(5):
        x, y = filter_generator.get_examples()
        assert _check_shape_and_grad(filter_generator, None, x, y)

    fixed_size = 42
    filter_generator = FilterGenerator(generator, filter_fn, size=fixed_size, update_size=False)
    for _ in range(5):
        assert _check_shape_and_grad(filter_generator, fixed_size)
        filter_generator.get_examples()


def test_resample_generator():
    size = 100

    sample_size = size
    x_expected = np.arange(size, dtype=np.float32)
    generator = PredefinedGenerator(x_expected)
    resample_generator = ResampleGenerator(generator, size=sample_size, replacement=False)
    x = resample_generator.get_examples()
    assert _check_shape_and_grad(resample_generator, sample_size, x)
    # noinspection PyTypeChecker
    assert _check_iterable_equal(torch.sort(x)[0], x_expected)

    sample_size = size // 2
    x = np.arange(size, dtype=np.float32)
    y = np.arange(size, size * 2, dtype=np.float32)
    generator = PredefinedGenerator(x, y)
    resample_generator = ResampleGenerator(generator, size=sample_size, replacement=False)
    x, y = resample_generator.get_examples()
    assert _check_shape_and_grad(resample_generator, sample_size, x, y)
    assert _check_iterable_equal(x + 100, y)
    assert len(torch.unique(x.detach())) == len(x)

    sample_size = size * 3 // 4
    x = np.arange(size, dtype=np.float32)
    y = np.arange(size, size * 2, dtype=np.float32)
    generator = PredefinedGenerator(x, y)
    resample_generator = ResampleGenerator(generator, size=sample_size, replacement=True)
    x, y = resample_generator.get_examples()
    assert _check_shape_and_grad(resample_generator, sample_size, x, y)
    assert _check_iterable_equal(x + 100, y)
    assert len(torch.unique(x.detach())) < len(x)

    sample_size = size * 2
    x = np.arange(size, dtype=np.float32)
    y = np.arange(size, size * 2, dtype=np.float32)
    z = np.arange(size * 2, size * 3, dtype=np.float32)
    generator = PredefinedGenerator(x, y, z)
    resample_generator = ResampleGenerator(generator, size=sample_size, replacement=True)
    x, y, z = resample_generator.get_examples()
    assert _check_shape_and_grad(resample_generator, sample_size, x, y, z)
    assert _check_iterable_equal(x + 100, y)
    assert _check_iterable_equal(y + 100, z)
    assert len(torch.unique(x.detach())) < len(x)


def test_batch_generator():
    size = 10
    batch_size = 3
    x = np.arange(size, dtype=np.float32)
    answer_x = np.arange(batch_size) % size
    generator = PredefinedGenerator(x)
    batch_generator = BatchGenerator(generator, batch_size)
    for _ in range(50):
        x = batch_generator.get_examples()
        assert _check_shape_and_grad(batch_generator, batch_size, x)
        assert _check_iterable_equal(answer_x, x)
        assert len(batch_generator.cached_xs) <= size + max(size, batch_size)
        # update answer for next iteration
        answer_x = (answer_x + batch_size) % size

    size = 3
    batch_size = 10
    x = np.arange(size, dtype=np.float32)
    y = np.arange(size, dtype=np.float32)
    answer_x = np.arange(batch_size) % size
    answer_y = np.arange(batch_size) % size
    generator = PredefinedGenerator(x, y)
    batch_generator = BatchGenerator(generator, batch_size)
    for _ in range(50):
        x, y = batch_generator.get_examples()
        assert _check_shape_and_grad(batch_generator, batch_size, x, y)
        assert _check_iterable_equal(answer_x, x)
        assert _check_iterable_equal(answer_y, y)
        assert len(batch_generator.cached_xs) <= size + max(size, batch_size)
        # update answer for next iteration
        answer_x = (answer_x + batch_size) % size
        answer_y = (answer_y + batch_size) % size
