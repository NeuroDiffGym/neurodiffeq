import sys
import torch
import numpy as np
from pytest import raises
# atomic-ish generator classes
from neurodiffeq.generator import Generator1D
from neurodiffeq.generator import Generator2D
from neurodiffeq.generator import Generator3D
from neurodiffeq.generator import GeneratorSpherical
# complex generator classes
from neurodiffeq.generator import ConcatGenerator

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
