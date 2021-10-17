"""This module contains atomic generator classes and useful tools to construct complex generators out of atomic ones
"""
import torch
import numpy as np
from typing import List


def _chebyshev_first(a, b, n):
    nodes = torch.cos(((torch.arange(n) + 0.5) / n) * np.pi)
    nodes = ((a + b) + (b - a) * nodes) / 2
    nodes.requires_grad_(True)
    return nodes


def _chebyshev_second(a, b, n):
    nodes = torch.cos(torch.arange(n) / float(n - 1) * np.pi)
    nodes = ((a + b) + (b - a) * nodes) / 2
    nodes.requires_grad_(True)
    return nodes


def _compute_log_negative(t_min, t_max, whence):
    if t_min <= 0 or t_max <= 0:
        suggested_t_min = 10 ** t_min
        suggested_t_max = 10 ** t_max
        raise ValueError(
            f"In this version of neurodiffeq, "
            f"the interval [{t_min}, {t_max}] cannot be used for log-sampling in {whence} "
            f"If you meant to sample from the interval [10 ^ {t_min}, 10 ^ {t_max}], "
            f"please pass in {suggested_t_min} and {suggested_t_max}"
        )

    return np.log10(t_min), np.log10(t_max)


class BaseGenerator:
    """Base class for all generators; Children classes must implement a `.get_examples` method and a `.size` field.
    """

    def __init__(self):
        self.size = None

    def get_examples(self) -> List[torch.Tensor]:
        pass  # pragma: no cover

    @staticmethod
    def check_generator(obj):
        if not isinstance(obj, BaseGenerator):
            raise ValueError(f"{obj} is not a generator")

    def __add__(self, other):
        self.check_generator(other)
        return ConcatGenerator(self, other)

    def __mul__(self, other):
        self.check_generator(other)
        return EnsembleGenerator(self, other)

    def __xor__(self, other):
        self.check_generator(other)
        return MeshGenerator(self, other)

    def _internal_vars(self) -> dict:
        return dict(size=self.size)

    @staticmethod
    def _obj_repr(obj) -> str:
        if isinstance(obj, tuple):
            return '(' + ', '.join(BaseGenerator._obj_repr(item) for item in obj) + ')'
        if isinstance(obj, list):
            return '[' + ', '.join(BaseGenerator._obj_repr(item) for item in obj) + ']'
        if isinstance(obj, set):
            return '{' + ', '.join(BaseGenerator._obj_repr(item) for item in obj) + '}'
        if isinstance(obj, dict):
            return '{' + ', '.join(
                BaseGenerator._obj_repr(k) + ': ' + BaseGenerator._obj_repr(obj[k])
                for k in obj
            ) + '}'

        if isinstance(obj, torch.Tensor):
            return f'tensor(shape={tuple(obj.shape)})'
        if isinstance(obj, np.ndarray):
            return f'ndarray(shape={tuple(obj.shape)})'
        return repr(obj)

    def __repr__(self):
        d = self._internal_vars()
        keys = ', '.join(f'{k}={self._obj_repr(d[k])}' for k in d)
        return f'{self.__class__.__name__}({keys})'


class Generator1D(BaseGenerator):
    """An example generator for generating 1-D training points.

    :param size: The number of points to generate each time `get_examples` is called.
    :type size: int
    :param t_min: The lower bound of the 1-D points generated, defaults to 0.0.
    :type t_min: float, optional
    :param t_max: The upper boound of the 1-D points generated, defaults to 1.0.
    :type t_max: float, optional
    :param method:
        The distribution of the 1-D points generated.

        - If set to 'uniform',
          the points will be drew from a uniform distribution Unif(t_min, t_max).
        - If set to 'equally-spaced',
          the points will be fixed to a set of linearly-spaced points that go from t_min to t_max.
        - If set to 'equally-spaced-noisy', a normal noise will be added to the previously mentioned set of points.
        - If set to 'log-spaced', the points will be fixed to a set of log-spaced points that go from t_min to t_max.
        - If set to 'log-spaced-noisy', a normal noise will be added to the previously mentioned set of points,
        - If set to 'chebyshev1' or 'chebyshev', the points are chebyshev nodes of the first kind over (t_min, t_max).
        - If set to 'chebyshev2', the points will be chebyshev nodes of the second kind over [t_min, t_max].

        defaults to 'uniform'.
    :type method: str, optional
    :raises ValueError: When provided with an unknown method.
    """

    def __init__(self, size, t_min=0.0, t_max=1.0, method='uniform', noise_std=None):
        r"""Initializer method

        .. note::
            A instance method `get_examples` is dynamically created to generate 1-D training points.
            It will be called by the function `solve` and `solve_system`.
        """
        super(Generator1D, self).__init__()
        self.size = size
        self.t_min, self.t_max = t_min, t_max
        self.method = method
        if noise_std:
            self.noise_std = noise_std
        else:
            self.noise_std = ((t_max - t_min) / size) / 4.0
        if method == 'uniform':
            self.examples = torch.zeros(self.size, requires_grad=True)
            self.getter = lambda: self.examples + torch.rand(self.size) * (self.t_max - self.t_min) + self.t_min
        elif method == 'equally-spaced':
            self.examples = torch.linspace(self.t_min, self.t_max, self.size, requires_grad=True)
            self.getter = lambda: self.examples
        elif method == 'equally-spaced-noisy':
            self.examples = torch.linspace(self.t_min, self.t_max, self.size, requires_grad=True)
            self.getter = lambda: torch.normal(mean=self.examples, std=self.noise_std)
        elif method == 'log-spaced':
            start, end = _compute_log_negative(t_min, t_max, self.__class__)
            self.examples = torch.logspace(start, end, self.size, requires_grad=True)
            self.getter = lambda: self.examples
        elif method == 'log-spaced-noisy':
            start, end = _compute_log_negative(t_min, t_max, self.__class__)
            self.examples = torch.logspace(start, end, self.size, requires_grad=True)
            self.getter = lambda: torch.normal(mean=self.examples, std=self.noise_std)
        elif method in ['chebyshev', 'chebyshev1']:
            self.examples = _chebyshev_first(t_min, t_max, size)
            self.getter = lambda: self.examples
        elif method == 'chebyshev2':
            self.examples = _chebyshev_second(t_min, t_max, size)
            self.getter = lambda: self.examples
        else:
            raise ValueError(f'Unknown method: {method}')

    def get_examples(self):
        return self.getter()

    def _internal_vars(self):
        d = super(Generator1D, self)._internal_vars()
        d.update(dict(
            t_min=self.t_min,
            t_max=self.t_max,
            method=self.method,
            noise_std=self.noise_std,
        ))
        return d


class Generator2D(BaseGenerator):
    r"""An example generator for generating 2-D training points.

        :param grid:
            The discretization of the 2 dimensions.
            If we want to generate points on a :math:`m \times n` grid, then `grid` is `(m, n)`.
            Defaults to `(10, 10)`.
        :type grid: tuple[int, int], optional
        :param xy_min:
            The lower bound of 2 dimensions.
            If we only care about :math:`x \geq x_0` and :math:`y \geq y_0`, then `xy_min` is `(x_0, y_0)`.
            Defaults to `(0.0, 0.0)`.
        :type xy_min: tuple[float, float], optional
        :param xy_max:
            The upper boound of 2 dimensions.
            If we only care about :math:`x \leq x_1` and :math:`y \leq y_1`, then `xy_min` is `(x_1, y_1)`.
            Defaults to `(1.0, 1.0)`.
        :type xy_max: tuple[float, float], optional
        :param method:
            The distribution of the 2-D points generated.

            - If set to 'equally-spaced', the points will be fixed to the grid specified.
            - If set to 'equally-spaced-noisy', a normal noise will be added to the previously mentioned set of points.
            - If set to 'chebyshev' or 'chebyshev1', the points will be 2-D chebyshev points of the first kind.
            - If set to 'chebyshev2', the points will be 2-D chebyshev points of the second kind.

            Defaults to 'equally-spaced-noisy'.
        :type method: str, optional
        :param xy_noise_std:
            The standard deviation of the noise on the x and y dimension.
            If not specified, the default value will be
            (``grid step size on x dimension`` / 4, ``grid step size on y dimension`` / 4).
        :type xy_noise_std: tuple[int, int], optional, defaults to None
        :raises ValueError: When provided with an unknown method.
    """

    def __init__(self, grid=(10, 10), xy_min=(0.0, 0.0), xy_max=(1.0, 1.0), method='equally-spaced-noisy',
                 xy_noise_std=None):
        r"""Initializer method

        .. note::
            A instance method `get_examples` is dynamically created to generate 2-D training points.
            It will be called by the function `solve2D`.
        """
        super(Generator2D, self).__init__()
        self.grid = grid
        self.size = grid[0] * grid[1]
        self.xy_min = xy_min
        self.xy_max = xy_max
        self.method = method
        self.xy_noise_std = xy_noise_std

        if method == 'equally-spaced':
            x = torch.linspace(xy_min[0], xy_max[0], grid[0], requires_grad=True)
            y = torch.linspace(xy_min[1], xy_max[1], grid[1], requires_grad=True)
            grid_x, grid_y = torch.meshgrid(x, y)
            self.grid_x, self.grid_y = grid_x.flatten(), grid_y.flatten()
            self.getter = lambda: (self.grid_x, self.grid_y)
        elif method == 'equally-spaced-noisy':
            x = torch.linspace(xy_min[0], xy_max[0], grid[0], requires_grad=True)
            y = torch.linspace(xy_min[1], xy_max[1], grid[1], requires_grad=True)
            grid_x, grid_y = torch.meshgrid(x, y)
            self.grid_x, self.grid_y = grid_x.flatten(), grid_y.flatten()
            if xy_noise_std:
                self.noise_xstd, self.noise_ystd = xy_noise_std
            else:
                self.noise_xstd = ((xy_max[0] - xy_min[0]) / grid[0]) / 4.0
                self.noise_ystd = ((xy_max[1] - xy_min[1]) / grid[1]) / 4.0
            self.getter = lambda: (
                torch.normal(mean=self.grid_x, std=self.noise_xstd),
                torch.normal(mean=self.grid_y, std=self.noise_ystd)
            )
        elif method in ['chebyshev1', 'chebyshev']:
            x = _chebyshev_first(xy_min[0], xy_max[0], grid[0])
            y = _chebyshev_first(xy_min[1], xy_max[1], grid[1])
            grid_x, grid_y = torch.meshgrid(x, y)
            self.grid_x, self.grid_y = grid_x.flatten(), grid_y.flatten()
            self.getter = lambda: (self.grid_x, self.grid_y)
        elif method == 'chebyshev2':
            x = _chebyshev_second(xy_min[0], xy_max[0], grid[0])
            y = _chebyshev_second(xy_min[1], xy_max[1], grid[1])
            grid_x, grid_y = torch.meshgrid(x, y)
            self.grid_x, self.grid_y = grid_x.flatten(), grid_y.flatten()
            self.getter = lambda: (self.grid_x, self.grid_y)
        else:
            raise ValueError(f'Unknown method: {method}')

    def get_examples(self):
        return self.getter()

    def _internal_vars(self) -> dict:
        d = super(Generator2D, self)._internal_vars()
        d.update(dict(
            grid=self.grid,
            xy_min=self.xy_min,
            xy_max=self.xy_max,
            method=self.method,
            xy_noise_std=self.xy_noise_std,
        ))
        return d


class Generator3D(BaseGenerator):
    r"""An example generator for generating 3-D training points. NOT TO BE CONFUSED with `GeneratorSpherical`

        :param grid:
            The discretization of the 3 dimensions.
            If we want to generate points on a :math:`m \times n \times k` grid,
            then `grid` is `(m, n, k)`, defaults to `(10, 10, 10)`.
        :type grid: tuple[int, int, int], optional
        :param xyz_min:
            The lower bound of 3 dimensions.
            If we only care about :math:`x \geq x_0`, :math:`y \geq y_0`,
            and :math:`z \geq z_0` then `xyz_min` is :math:`(x_0, y_0, z_0)`.
            Defaults to `(0.0, 0.0, 0.0)`.
        :type xyz_min: tuple[float, float, float], optional
        :param xyz_max:
            The upper bound of 3 dimensions.
            If we only care about :math:`x \leq x_1`, :math:`y \leq y_1`, i
            and :math:`z \leq z_1` then `xyz_max` is :math:`(x_1, y_1, z_1)`.
            Defaults to `(1.0, 1.0, 1.0)`.
        :type xyz_max: tuple[float, float, float], optional
        :param method:
            The distribution of the 3-D points generated.

            - If set to 'equally-spaced', the points will be fixed to the grid specified.
            - If set to 'equally-spaced-noisy', a normal noise will be added to the previously mentioned set of points.
            - If set to 'chebyshev' or 'chebyshev1', the points will be 3-D chebyshev points of the first kind.
            - If set to 'chebyshev2', the points will be 3-D chebyshev points of the second kind.

            Defaults to 'equally-spaced-noisy'.
        :type method: str, optional
        :raises ValueError: When provided with an unknown method.
    """

    def __init__(self, grid=(10, 10, 10), xyz_min=(0.0, 0.0, 0.0), xyz_max=(1.0, 1.0, 1.0),
                 method='equally-spaced-noisy'):
        r"""Initializer method

        .. note::
            A instance method `get_examples` is dynamically created to generate 2-D training points.
            It will be called by the function `solve2D`.
        """
        super(Generator3D, self).__init__()
        self.size = grid[0] * grid[1] * grid[2]
        self.grid = grid
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.method = method

        if method in ['equally-spaced', 'equally-spaced-noisy']:
            x = torch.linspace(xyz_min[0], xyz_max[0], grid[0], requires_grad=True)
            y = torch.linspace(xyz_min[1], xyz_max[1], grid[1], requires_grad=True)
            z = torch.linspace(xyz_min[2], xyz_max[2], grid[2], requires_grad=True)
        elif method in ['chebyshev', 'chebyshev1']:
            x = _chebyshev_first(xyz_min[0], xyz_max[0], grid[0])
            y = _chebyshev_first(xyz_min[1], xyz_max[1], grid[1])
            z = _chebyshev_first(xyz_min[2], xyz_max[2], grid[2])
        elif method == 'chebyshev2':
            x = _chebyshev_second(xyz_min[0], xyz_max[0], grid[0])
            y = _chebyshev_second(xyz_min[1], xyz_max[1], grid[1])
            z = _chebyshev_second(xyz_min[2], xyz_max[2], grid[2])
        else:
            raise ValueError(f"Unknown method: {method}")

        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
        self.grid_x, self.grid_y, self.grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()

        if method in ['equally-spaced', 'chebyshev', 'chebyshev1', 'chebyshev2']:
            self.getter = lambda: (self.grid_x, self.grid_y, self.grid_z)
        elif method == 'equally-spaced-noisy':
            self.noise_xmean = torch.zeros(self.size)
            self.noise_ymean = torch.zeros(self.size)
            self.noise_zmean = torch.zeros(self.size)
            self.noise_xstd = torch.ones(self.size) * ((xyz_max[0] - xyz_min[0]) / grid[0]) / 4.0
            self.noise_ystd = torch.ones(self.size) * ((xyz_max[1] - xyz_min[1]) / grid[1]) / 4.0
            self.noise_zstd = torch.ones(self.size) * ((xyz_max[2] - xyz_min[2]) / grid[2]) / 4.0
            self.getter = lambda: (
                self.grid_x + torch.normal(mean=self.noise_xmean, std=self.noise_xstd),
                self.grid_y + torch.normal(mean=self.noise_ymean, std=self.noise_ystd),
                self.grid_z + torch.normal(mean=self.noise_zmean, std=self.noise_zstd),
            )
        else:
            raise ValueError(f'Unknown method: {method}')

    def get_examples(self):
        return self.getter()

    def _internal_vars(self) -> dict:
        d = super(Generator3D, self)._internal_vars()
        d.update(dict(
            grid=self.grid,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
            method=self.method,
        ))
        return d


class GeneratorND(BaseGenerator):
    r"""An example generator for generating N-D training points.

        :param grid:
            The discretization of the N dimensions.
            If we want to generate points on a :math:`n_1 \times n_2 \times ... \times n_N` grid,
            then `grid` is `(n_1, n_2, ... , n_N)`.
            Defaults to `(10, 10)`.
        :type grid: tuple[int, int, ... , int], or it can be int if N=1, optional
        :param r_min:
            The lower bound of N dimensions.
            If we only care about :math:`r_1 \geq r_1^{min}`, :math:`r_2 \geq r_2^{min}`, ... ,
            and :math:`r_N \geq r_N^{min}` then `r_min` is `(r_1_min, r_2_min, ... , r_N_min)`.
            Defaults to `(0.0, 0.0)`.
        :type r_min: tuple[float, ... , float], or it can be float if N=1, optional
        :param r_max:
            The upper boound of N dimensions.
            If we only care about :math:`r_1 \leq r_1^{max}`, :math:`r_2 \leq r_2^{max}`, ... ,
            and :math:`r_N \leq r_N^{max}` then `r_max` is `(r_1_max, r_2_max, ... , r_N_max)`.
            Defaults to `(1.0, 1.0)`.
        :type r_max: tuple[float, ... , float], or it can be float if N=1, optional
        :param methods:
            The a list of the distributions of each of the 1-D points generated that make the total N-D points.

            - If set to 'uniform',
              the points will be drew from a uniform distribution Unif(r_min[i], r_max[i]).
            - If set to 'equally-spaced',
              the points will be fixed to a set of linearly-spaced points that go from r_min[i] to r_max[i].
            - If set to 'log-spaced',
              the points will be fixed to a set of log-spaced points that go from r_min[i] to r_max[i].
            - If set to 'exp-spaced',
              the points will be fixed to a set of exp-spaced points that go from r_min[i] to r_max[i].
            - If set to 'chebyshev' or 'chebyshev1',
              the points will be chebyshev points of the first kind that go from r_min[i] to r_max[i].
            - If set to 'chebyshev2',
              the points will be chebyshev points of the second kind that go from r_min[i] to r_max[i].

            Defaults to ['equally-spaced', 'equally-spaced'].
        :type methods: list[str, str, ... , str], or it can be str if N=1, optional
        :param noisy:
            if set to True a normal noise will be added to all of the N sets of points that make the generator.
            Defaults to True.
        :type noisy: bool
        :raises ValueError: When provided with unknown methods.
    """

    def __init__(self, grid=(10, 10), r_min=(0.0, 0.0), r_max=(1.0, 1.0),
                 methods=['equally-spaced', 'equally-spaced'], noisy=True, r_noise_std=None,
                 **kwargs):

        super(GeneratorND, self).__init__()
        self.size = np.prod(grid)
        self.grid = grid
        self.r_min = r_min
        self.r_max = r_max
        self.methods = methods
        self.noisy = noisy
        self.r_noise_std = r_noise_std

        if isinstance(methods, str):
            methods = [methods]
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(r_min, float) or isinstance(r_min, int):
            r_min = (r_min,)
        if isinstance(r_max, float) or isinstance(r_max, int):
            r_max = (r_max,)
        if isinstance(r_noise_std, float) or isinstance(r_noise_std, int):
            r_noise_std = (r_noise_std,)

        N = len(grid)
        cut = kwargs.pop('cut', tuple((None, None) for i in range(N)))
        base = kwargs.pop('base', tuple(10 for i in range(N)))
        abs_value = kwargs.pop('abs_value', False)

        if kwargs:
            raise ValueError(f'Unknown keyword argument(s): {list(kwargs.keys())}')
        if isinstance(base, float) or isinstance(base, int):
            base = (base,)
        if isinstance(cut[0], float) or isinstance(cut[0], int) or cut[0] is None:
            cut = (cut,)

        r = []
        r_noise_std_list = []
        for i in range(N):

            method = methods[i]

            if r_noise_std:
                noise_rstd = r_noise_std[i]
            else:
                noise_rstd = ((r_max[i] - r_min[i]) / grid[i]) / 4.0

            if method == 'equally-spaced':
                x = torch.linspace(r_min[i], r_max[i], grid[i], requires_grad=True)
                noise_rstd_tensor = noise_rstd * torch.ones(x.size())
            elif method == 'uniform':
                x = torch.zeros(grid[i], requires_grad=True)
                x = x + torch.rand(grid[i]) * (r_max[i] - r_min[i]) + r_min[i]
                noise_rstd_tensor = torch.zeros(x.size())
            elif method == 'log-spaced':
                r_min_log = np.log10(r_min[i])
                r_max_log = np.log10(r_max[i])
                x = torch.logspace(r_min_log, r_max_log, grid[i], requires_grad=True)
                noise_rstd_tensor = (noise_rstd * torch.logspace(r_min_log, r_max_log, grid[i]))
            elif method == 'exp-spaced':
                r_min_exp = base[i] ** r_min[i]
                r_max_exp = base[i] ** r_max[i]
                x = torch.linspace(r_min_exp, r_max_exp, grid[i], requires_grad=True)
                x = (torch.log(x) / np.log(base[i])).clone().detach().requires_grad_(True)
                noise_rstd_tensor = (noise_rstd * x).clone().detach()
            elif method in ['chebyshev', 'chebyshev1']:
                x = _chebyshev_first(r_min[i], r_max[i], grid[i])
                noise_rstd_tensor = noise_rstd * torch.ones(x.size())
            elif method == 'chebyshev2':
                x = _chebyshev_second(r_min[i], r_max[i], grid[i])
                noise_rstd_tensor = noise_rstd * torch.ones(x.size())
            else:
                raise ValueError(f'Unknown method: {method}')

            x = x[cut[i][0]:cut[i][1]]
            noise_rstd_tensor = noise_rstd_tensor[cut[i][0]:cut[i][1]]
            r.append(x)
            r_noise_std_list.append(noise_rstd_tensor)

        grid_r = torch.meshgrid(r)
        grid_std = torch.meshgrid(r_noise_std_list)
        self.grid_r = [grid_r[j].flatten() for j in range(N)]
        self.grid_std = [grid_std[j].flatten() for j in range(N)]
        if noisy:
            if abs_value:
                self.getter = lambda: tuple(torch.abs(torch.normal(self.grid_r[n], self.grid_std[n])) for n in range(N))
            else:
                self.getter = lambda: tuple(torch.normal(self.grid_r[n], self.grid_std[n]) for n in range(N))
        else:
            self.getter = lambda: tuple(self.grid_r[n] for n in range(N))

    def get_examples(self):
        return self.getter()

    def _internal_vars(self) -> dict:
        d = super(GeneratorND, self)._internal_vars()
        d.update(dict(
            grid=self.grid,
            r_min=self.r_min,
            r_max=self.r_max,
            methods=self.methods,
            noisy=self.noisy,
            r_noise_std=self.r_noise_std
        ))
        return d


class GeneratorSpherical(BaseGenerator):
    r"""A generator for generating points in spherical coordinates.

    :param size: Number of points in 3-D sphere.
    :type size: int
    :param r_min: Radius of the interior boundary.
    :type r_min: float, optional
    :param r_max: Radius of the exterior boundary.
    :type r_max: float, optional
    :param method:
        The distribution of the 3-D points generated.

        - If set to 'equally-radius-noisy', radius of the points will be drawn
          from a uniform distribution :math:`r \sim U[r_{min}, r_{max}]`.
        - If set to 'equally-spaced-noisy', squared radius of the points will be drawn
          from a uniform distribution :math:`r^2 \sim U[r_{min}^2, r_{max}^2]`

        Defaults to 'equally-spaced-noisy'.

    :type method: str, optional

    .. note::
        Not to be confused with ``Generator3D``.
    """

    # noinspection PyMissingConstructor
    def __init__(self, size, r_min=0., r_max=1., method='equally-spaced-noisy'):
        super(GeneratorSpherical, self).__init__()
        if r_min < 0 or r_max < r_min:
            raise ValueError(f"Illegal range [{r_min}, {r_max}]")

        if method == 'equally-spaced-noisy':
            lower = r_min ** 2
            upper = r_max ** 2
            rng = upper - lower
            self.get_r = lambda: torch.sqrt(rng * torch.rand(self.shape) + lower)
        elif method == "equally-radius-noisy":
            lower = r_min
            upper = r_max
            rng = upper - lower
            self.get_r = lambda: rng * torch.rand(self.shape) + lower
        else:
            raise ValueError(f'Unknown method: {method}')

        self.size = size  # stored for `solve_spherical_system` to access
        self.r_min = r_min
        self.r_max = r_max
        self.method = method
        self.shape = (size,)  # used for `self.get_example()`

    def get_examples(self):
        a = torch.rand(self.shape)
        b = torch.rand(self.shape)
        c = torch.rand(self.shape)
        denom = a + b + c
        # `x`, `y`, `z` here are just for computation of `theta` and `phi`
        epsilon = 1e-6
        x = torch.sqrt(a / denom) + epsilon
        y = torch.sqrt(b / denom) + epsilon
        z = torch.sqrt(c / denom) + epsilon
        # `sign_x`, `sign_y`, `sign_z` are either -1 or +1
        sign_x = torch.randint(0, 2, self.shape, dtype=x.dtype) * 2 - 1
        sign_y = torch.randint(0, 2, self.shape, dtype=y.dtype) * 2 - 1
        sign_z = torch.randint(0, 2, self.shape, dtype=z.dtype) * 2 - 1

        x = x * sign_x
        y = y * sign_y
        z = z * sign_z

        theta = torch.acos(z).requires_grad_(True)
        phi = -torch.atan2(y, x) + np.pi  # atan2 ranges (-pi, pi] instead of [0, 2pi)
        phi.requires_grad_(True)
        r = self.get_r().requires_grad_(True)

        return r, theta, phi

    def _internal_vars(self) -> dict:
        d = super(GeneratorSpherical, self)._internal_vars()
        d.update(dict(
            r_min=self.r_min,
            r_max=self.r_max,
            method=self.method,
        ))
        return d


class ConcatGenerator(BaseGenerator):
    r"""An concatenated generator for sampling points,
    whose ``get_examples()`` method returns the concatenated vector of the samples returned by its sub-generators.

    :param generators: a sequence of sub-generators, must have a ``.size`` field and a ``.get_examples()`` method
    :type generators: Tuple[BaseGenerator]

    .. note::
        Not to be confused with ``EnsembleGenerator`` which returns all the samples of its sub-generators.
    """

    def __init__(self, *generators):
        super(ConcatGenerator, self).__init__()
        self.generators = generators
        self.size = sum(gen.size for gen in generators)

    def get_examples(self):
        all_examples = [gen.get_examples() for gen in self.generators]
        if isinstance(all_examples[0], torch.Tensor):
            return torch.cat(all_examples)
        # zip(*sequence) is just `unzip`ping a sequence into sub-sequences, refer to this post for more
        # https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip
        segmented = zip(*all_examples)
        return [torch.cat(seg) for seg in segmented]

    def _internal_vars(self) -> dict:
        d = super(ConcatGenerator, self)._internal_vars()
        d.update(dict(
            generators=self.generators,
        ))
        return d


class StaticGenerator(BaseGenerator):
    """A generator that returns the same samples every time.
    The static samples are obtained by the sub-generator at instantiation time.

    :param generator: a generator used to generate the static samples
    :type generator: BaseGenerator
    """

    def __init__(self, generator):
        super(StaticGenerator, self).__init__()
        self.generator = generator
        self.size = generator.size
        self.examples = generator.get_examples()

    def get_examples(self):
        return self.examples

    def _internal_vars(self) -> dict:
        d = super(StaticGenerator, self)._internal_vars()
        d.update(dict(
            generator=self.generator,
            examples=self.examples,
        ))
        return d


class PredefinedGenerator(BaseGenerator):
    """A generator for generating points that are fixed and predefined.

    :param xs: The x-dimension of the trianing points
    :type xs: `torch.Tensor`
    :param ys: The y-dimension of the training points
    :type ys: `torch.Tensor`
    """

    def __init__(self, *xs):
        super(PredefinedGenerator, self).__init__()
        self.size = len(xs[0])
        for x in xs:
            if self.size != len(x):
                raise ValueError('tensors of different lengths encountered {self.size} != {len(x)}')
        xs = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in xs]
        self.xs = [torch.flatten(x).requires_grad_(True) for x in xs]

        if len(self.xs) == 1:
            self.xs = self.xs[0]

    def get_examples(self):
        """Returns the training points. Points are fixed and predefined.

            :returns: The predefined training points
            :rtype: tuple[`torch.Tensor`]
        """
        return self.xs

    def _internal_vars(self) -> dict:
        d = super(PredefinedGenerator, self)._internal_vars()
        d.update(dict(
            xs=self.xs,
        ))
        return d


class TransformGenerator(BaseGenerator):
    """A generator which applies certain transformations on the sample vectors.

    :param generator:
        A generator used to generate samples on which transformations will be applied.
    :type generator: BaseGenerator
    :param transforms:
        A list of transformations to be applied on the sample vectors.
        Identity transformation can be replaced with None
    :type transforms: list[callable]
    :param transform:
        A callable that transforms the output(s) of base generator to another (tuple of) coordinate(s).
    :type transform: callable
    """

    def __init__(self, generator, transforms=None, transform=None):
        super(TransformGenerator, self).__init__()
        self.generator = generator
        self.size = generator.size
        if transforms is not None and transform is not None:
            raise ValueError("transform and transforms cannot be both specified")
        if transforms is not None:
            self.trans = [
                (lambda x: x) if t is None else t
                for t in transforms
            ]
        elif transform is not None:
            self.trans = transform
        else:
            self.trans = lambda x: x

    def get_examples(self):
        xs = self.generator.get_examples()
        if isinstance(xs, torch.Tensor):
            if callable(self.trans):
                return self.trans(xs)
            else:
                return self.trans[0](xs)
        if callable(self.trans):
            return self.trans(*xs)
        else:
            return tuple(t(x) for t, x in zip(self.trans, xs))

    def _internal_vars(self) -> dict:
        d = super(TransformGenerator, self)._internal_vars()
        d.update(dict(
            generator=self.generator,
            trans=self.trans,
        ))
        return d


class EnsembleGenerator(BaseGenerator):
    r"""A generator for sampling points whose `get_examples` method returns all the samples of its sub-generators.
    All sub-generator must return tensors of the same shape.
    The number of tensors returned by each sub-generator can be different.

    :param generators: a sequence of sub-generators, must have a .size field and a .get_examples() method
    :type generators: Tuple[BaseGenerator]

    .. note::
        Not to be confused with ``ConcatGenerator`` which returns
        the concatenated vector of samples returned by its sub-generators.
    """

    def __init__(self, *generators):
        super(EnsembleGenerator, self).__init__()
        self.size = generators[0].size
        for i, gen in enumerate(generators):
            if gen.size != self.size:
                raise ValueError(f"gens[{i}].size ({gen.size}) != gens[0].size ({self.size})")
        self.generators = generators

    def get_examples(self):
        ret = tuple()
        for g in self.generators:
            ex = g.get_examples()
            if isinstance(ex, list):
                ex = tuple(ex)
            elif isinstance(ex, torch.Tensor):
                ex = (ex,)
            ret += ex

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def _internal_vars(self) -> dict:
        d = super(EnsembleGenerator, self)._internal_vars()
        d.update(dict(
            generators=self.generators,
        ))
        return d


class MeshGenerator(BaseGenerator):
    r"""A generator for sampling points whose `get_examples` method returns a mesh of the samples of its sub-generators.
    All sub-generators must return tensors of the same shape, or a tuple of tensors of the same shape.
    The number of tensors returned by each sub-generator can be different, but the intent behind
    this class is to create an N dimensional generator from several 1 dimensional generators, so each input generator
    should represents one of the dimensions of your problem. An exception is made for
    using a ``MeshGenerator`` as one of the inputs of another ``MeshGenerator``. In that case the original
    meshed generators are extracted from the input ``MeshGenerator``, and the final mesh is used using those
    (e.g ``MeshGenerator(MeshGenerator(g1, g2), g3)`` is equivalent to ``MeshGenerator(g1, g2, g3)``, where
    g1, g2 and g3 are ``Generator1D``).
    This is done to make the use of the ^ infix consistent with the use of
    the ``MeshGenerator`` class itself (e.g ``MeshGenerator(g1, g2, g3)`` is equivalent to g1 ^ g2 ^ g3), where
    g1, g2 and g3 are ``Generator1D``).

    :param generators: a sequence of sub-generators, must have a .size field and a .get_examples() method
    :type generators: Tuple[BaseGenerator]
    """

    def __init__(self, *generators):
        super(MeshGenerator, self).__init__()
        self.generators = []
        for g in generators:
            if isinstance(g, MeshGenerator):
                for s in g.generators:
                    self.generators.append(s)
            else:
                self.generators.append(g)
        self.size = np.prod(tuple(g.size for g in self.generators))

    def get_examples(self):
        ret = tuple()
        for g in self.generators:
            ex = g.get_examples()
            if isinstance(ex, list):
                ex = tuple(ex)
            elif isinstance(ex, torch.Tensor):
                ex = (ex,)
            ret += ex

        if len(ret) == 1:
            return ret[0]
        else:
            ret = torch.meshgrid(ret)
            ret_f = tuple()
            for r in ret:
                ret_f += (r.flatten(),)
            return ret_f

    def _internal_vars(self) -> dict:
        d = super(MeshGenerator, self)._internal_vars()
        d.update(dict(
            generators=self.generators,
        ))
        return d


class FilterGenerator(BaseGenerator):
    """A generator which applies some filtering before samples are returned

    :param generator:
        A generator used to generate samples to be filtered.
    :type generator: BaseGenerator
    :param filter_fn:
        A filter to be applied on the sample vectors; maps a list of tensors to a mask tensor.
    :type filter_fn: callable
    :param size:
        Size to be used for `self.size`.
        If not given, this attribute is initialized to the size of ``generator``.
    :type size: int
    :param update_size:
        Whether or not to update `.size` after each call of `self.get_examples`.
        Defaults to True.
    :type update_size: bool
    """

    def __init__(self, generator, filter_fn, size=None, update_size=True):
        super(FilterGenerator, self).__init__()
        self.generator = generator
        self.filter_fn = filter_fn
        if size is None:
            self.size = generator.size
        else:
            self.size = size
        self.update_size = update_size

    def get_examples(self):
        xs = self.generator.get_examples()
        if isinstance(xs, torch.Tensor):
            xs = [xs]
        mask = self.filter_fn(xs)
        xs = [x[mask] for x in xs]
        if self.update_size:
            self.size = len(xs[0])
        if len(xs) == 1:
            return xs[0]
        else:
            return xs

    def _internal_vars(self) -> dict:
        d = super(FilterGenerator, self)._internal_vars()
        d.update(dict(
            generator=self.generator,
            filter_fn=self.filter_fn,
        ))
        return d


class ResampleGenerator(BaseGenerator):
    """A generator whose output is shuffled and resampled every time

    :param generator: A generator used to generate samples to be shuffled and resampled.
    :type generator: BaseGenerator
    :param size: Size of the shuffled output. Defaults to the size of ``generator``.
    :type size: int
    :param replacement: Whether to sample with replacement or not. Defaults to False.
    :type replacement: bool
    """

    def __init__(self, generator, size=None, replacement=False):
        super(ResampleGenerator, self).__init__()
        self.generator = generator
        if size is None:
            self.size = generator.size
        else:
            self.size = size
        self.replacement = replacement

    def get_examples(self):
        if self.replacement:
            indices = torch.randint(self.generator.size, (self.size,))
        else:
            indices = torch.randperm(self.generator.size)[:self.size]

        xs = self.generator.get_examples()
        if isinstance(xs, torch.Tensor):
            return xs[indices]
        else:
            return [x[indices] for x in xs]

    def _internal_vars(self) -> dict:
        d = super(ResampleGenerator, self)._internal_vars()
        d.update(dict(
            generator=self.generator,
            replacement=self.replacement,
        ))
        return d


class BatchGenerator(BaseGenerator):
    """A generator which caches samples and returns a single batch of the samples at a time

    :param generator:
        A generator used for getting (cached) examples.
    :type generator: BaseGenerator
    :param batch_size:
        Number of batches to be returned.
        It can be larger than size of ``generator``, but inefficient if so.
    :type batch_size: int
    """

    def __init__(self, generator, batch_size):
        super(BatchGenerator, self).__init__()

        if generator.size <= 0:
            raise ValueError(f"generator has size {generator.size} <= 0")
        self.generator = generator
        self.size = batch_size
        self.cached_xs = self.generator.get_examples()
        if isinstance(self.cached_xs, torch.Tensor):
            self.cached_xs = [self.cached_xs]
        if isinstance(self.cached_xs, tuple):
            self.cached_xs = list(self.cached_xs)

    def get_examples(self):
        # update cache so that we have enough samples in a batch
        while len(self.cached_xs[0]) < self.size:
            new = self.generator.get_examples()
            if isinstance(new, torch.Tensor):
                new = [new]
            self.cached_xs = [torch.cat([x, n]) for x, n in zip(self.cached_xs, new)]

        batch = [x[:self.size] for x in self.cached_xs]
        # drop the returned samples
        self.cached_xs = [x[self.size:] for x in self.cached_xs]

        if len(batch) == 1:
            return batch[0]
        else:
            return batch

    def _internal_vars(self) -> dict:
        d = super(BatchGenerator, self)._internal_vars()
        d.update(dict(
            generator=self.generator,
        ))
        return d


class SamplerGenerator(BaseGenerator):
    def __init__(self, generator):
        super(SamplerGenerator, self).__init__()
        self.generator = generator
        self.size = generator.size

    def get_examples(self) -> List[torch.Tensor]:
        samples = self.generator.get_examples()
        if isinstance(samples, torch.Tensor):
            samples = [samples]
        samples = [u.reshape(-1, 1) for u in samples]
        return samples

    def _internal_vars(self) -> dict:
        d = super(SamplerGenerator, self)._internal_vars()
        d.update(dict(
            generator=self.generator,
        ))
        return d
