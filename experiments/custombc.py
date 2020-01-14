import numpy as np
import torch


class Point:
    
    def __repr__(self):
        return f'Point({self.loc})'
    
    def __init__(self, loc):
        self.loc = tuple(float(d) for d in loc)
        self.dim = len(loc)


class DirichletControlPoint(Point):
    
    def __repr__(self):
        return f'DirichletControlPoint({self.loc}, val={self.val})'
    
    def __init__(self, loc, val):
        super().__init__(loc)
        self.val = val


class Condition:
    
    def enforce(self, net, *dimensions):
        raise NotImplemented
    
    @staticmethod
    def _nn_output(net, *dimensions):
        original_shape = dimensions[0].shape
        output = net(torch.cat(dimensions, 1))
        return output.reshape(original_shape)

class CustomDirichletBoundaryCondition(Condition):
    
    def __init__(self, dirichlet_control_points, center_point):
        # drop deplicates and sort 'clockwise' (while not changing the original copy)
        dirichlet_control_points = self._clean_control_points(dirichlet_control_points, center_point)
        # fit Dirichlet dummy solution (A_D(x) in MacFall's paper)
        self.a_d_interp = InterpolatorCreator.fit_dirichlet_dummy_solution(dirichlet_control_points)
        # fit Dirichlet length factor (L_D(x) in MacFall's paper)
        self.l_d_interp = InterpolatorCreator.fit_dirichlet_length_factor(dirichlet_control_points)
    
    def a_d(self, *dimensions):
        return self.a_d_interp.interpolate(*dimensions)
    
    def l_d(self, *dimensions):
        return self.l_d_interp.interpolate(*dimensions)
    
    def in_domain(self, *dimensions):
        return self.l_d(*dimensions) > 0.0
    
    def enforce(self, net, *dimensions):
        # enforce Dirichlet boundary condition u_t(x) = A_D(x) + L_D(x)u_N(x)
        return self.a_d(*dimensions) + self.l_d(*dimensions) * self._nn_output(net, *dimensions)
    
    @staticmethod
    def _clean_control_points(control_points, center_point):
        # remove the control points that are defined more than once
        locs = set()
        unique_control_points = []
        for cp in control_points:
            if cp.loc not in locs:
                locs.add(cp.loc)
                unique_control_points.append(cp)

        # sort the control points 'clockwise' (from 0 to -2pi)
        # needs a better way to implement this, edge cases are many
        def clockwise(cp, significant_digit=1e-7):
            def gt_zero(number):
                return number >= significant_digit
            def lt_zero(number):
                return number <= -significant_digit
            def eq_zero(number):
                return abs(number) < significant_digit
            
            px, py = cp.loc
            cx, cy = center_point.loc
            dx, dy = px-cx, py-cy
            if   gt_zero(dx) and eq_zero(dy):
                tier = 0
            elif gt_zero(dx) and lt_zero(dy):
                tier = 1
            elif eq_zero(dx) and lt_zero(dy):
                tier = 2
            elif lt_zero(dx) and lt_zero(dy):
                tier = 3
            elif lt_zero(dx) and eq_zero(dy):
                tier = 4
            elif lt_zero(dx) and gt_zero(dy):
                tier = 5
            elif eq_zero(dx) and gt_zero(dy):
                tier = 6
            elif gt_zero(dx) and gt_zero(dy):
                tier = 7
            # assume that the second key won't be used 
            # - i.e. on the same side of center point (left or right)
            # there won't be multiple control points that
            # has the same y-coordinate as the center point
            return (tier, dx/dy if not eq_zero(dy) else 0) 
        unique_control_points.sort(key=clockwise)
        return unique_control_points


class InterpolatorCreator:

    @staticmethod
    def fit_dirichlet_dummy_solution(dirichlet_control_points):
        # specify input and output of thin plate spline
        from_points = dirichlet_control_points
        to_values = [dcp.val for dcp in dirichlet_control_points]
        # fit thin plate spline and save coefficients
        coefs = InterpolatorCreator._solve_thin_plate_spline(from_points, to_values)
        return ADInterpolator(coefs, dirichlet_control_points)
        
    @staticmethod
    def fit_dirichlet_length_factor(dirichlet_control_points, radius=0.5):
        # specify input and output of thin plate spline
        from_points = dirichlet_control_points
        to_points = InterpolatorCreator._create_circular_targets(dirichlet_control_points, radius)
        n_dim = to_points[0].dim
        to_values_each_dim = [[tp.loc[i] for tp in to_points] for i in range(n_dim)]
        # fit thin plate spline and save coefficients
        coefs_each_dim = [
            InterpolatorCreator._solve_thin_plate_spline(from_points, to_values)
            for to_values in to_values_each_dim
        ]
        return LDInterpolator(coefs_each_dim, dirichlet_control_points, radius)
    
    @staticmethod
    def _solve_thin_plate_spline(from_points, to_values):
        assert len(from_points) == len(to_values)
        n_dims = from_points[0].dim
        n_pnts = len(from_points)
        n_eqs = n_dims+n_pnts+1
        
        # weights of the eq_no'th equation
        def equation_weights(eq_no):
            
            weights = np.zeros(n_eqs)
            
            # the first M equations (M is the number of control points)
            if eq_no < n_pnts:
                p = from_points[eq_no]
                # the first M weights 
                for i, fp in enumerate(from_points):
                    ri_sq = Interpolator._ri_sq_thin_plate_spline_pretrain(p, fp)
                    weights[i] = ri_sq * np.log(ri_sq)
                # the M+1'th weight
                weights[n_pnts] = 1.0
                # the rest #dimension weights
                for j in range(n_dims):
                    weights[n_pnts+1+j] = p.loc[j]
            # the M+1'th equation
            elif eq_no <  n_pnts + n_dims:
                j = eq_no - n_pnts
                for i in range(n_pnts):
                    weights[i] = from_points[i].loc[j]
            # the rest #dimension equations
            elif eq_no == n_pnts + n_dims:
                weights[:n_pnts] = 1.0
            else:
                raise ValueError(f'Invalid equation number: {eq_no}')
                
            return weights
        
        # create linear system
        W = np.zeros((n_eqs, n_eqs))
        for eq_no in range(n_eqs):
            W[eq_no] = equation_weights(eq_no)
        b = np.zeros(n_eqs)
        b[:n_pnts] = to_values
        
        # solve linear system and return coefficients
        return np.linalg.solve(W, b)  
    
    @staticmethod
    def _create_circular_targets(control_points, radius):
        # create equally spaced target points, this is for 2-d control points
        # TODO 3-d control points
        return [
            Point( (radius*np.cos(theta), radius*np.sin(theta)) )
            for theta in -np.linspace(0, 2*np.pi, len(control_points), endpoint=False)
        ]

class Interpolator:
    
    def interpolate(self, *dimensions):
        raise NotImplementedError

    @staticmethod
    def _interpolate_by_thin_plate_spline(coefs, control_points, *dimensions):
        n_pnts = len(control_points)
        to_value_unfinished = torch.zeros_like(dimensions[0])
        # the first M basis functions (M is the number of control points)
        for coef, cp in zip(coefs, control_points):
            ri_sq = Interpolator._ri_sq_thin_plate_spline_trainval(cp, *dimensions)
            to_value_unfinished += coef * ri_sq * torch.log(ri_sq)
        # the M+1'th basis function
        to_value_unfinished += coefs[n_pnts]
        # the rest #dimension basis functions
        for j, d in enumerate(dimensions):
            to_value_unfinished += coefs[n_pnts+1+j] * d
        return to_value_unfinished

    # to be used in fitting soefficients of thin plate spline
    @staticmethod
    def _ri_sq_thin_plate_spline_pretrain(point_i, point_j, stiffness=0.01):
        return sum((di-dj)**2 for di, dj in zip(point_i.loc, point_j.loc)) + stiffness**2

    # to be used in transforming output of neural networks
    @staticmethod
    def _ri_sq_thin_plate_spline_trainval(point_i, *dimensions, stiffness=0.01):
        return sum((d-di)**2 for di, d in zip(point_i.loc, dimensions)) + stiffness**2

class ADInterpolator(Interpolator):

    def __init__(self, coefs, control_points):
        self.coefs = coefs
        self.control_points = control_points
    
    def interpolate(self, *dimensions):
        return Interpolator._interpolate_by_thin_plate_spline(
                self.coefs, self.control_points, *dimensions
            )

class LDInterpolator(Interpolator):

    def __init__(self, coefs_each_dim, control_points, radius):
        self.coefs_each_dim = coefs_each_dim
        self.control_points = control_points
        self.radius = radius

    def interpolate(self, *dimensions):
        dimensions_mapped = [
            Interpolator._interpolate_by_thin_plate_spline(
                coefs_dim, self.control_points, *dimensions
            )
            for coefs_dim in self.coefs_each_dim
        ]
        return self.radius**2 - sum(d**2 for d in dimensions_mapped)