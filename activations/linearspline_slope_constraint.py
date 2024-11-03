import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractproperty, abstractmethod


### Slope projection function (To enforce the slope constraint)
def project_slopes_vectorized(nodal_values, knot_positions, smin, smax):
    """
    Project the slopes of a given set of nodal values with respect to non-uniform knot positions.

    Parameters:
    - nodal_values: Tensor of nodal values (f_n).
    - knot_positions: Tensor of corresponding knot positions (t_n).
    - smin: Minimum allowable slope.
    - smax: Maximum allowable slope.

    Returns:
    - new_nodal_values: Tensor of adjusted nodal values after slope projection.
    """
    # Number of nodal values
    N = nodal_values.size(0)

    # Calculate the differences in nodal values and knot positions
    # I have a feeling that we might need slopes outside this function as well. we will see!
    delta_f = nodal_values[1:] - nodal_values[:-1]
    delta_t = knot_positions[1:] - knot_positions[:-1]

    # Calculate slopes
    slopes = torch.zeros(N)
    slopes[1:] = delta_f / delta_t

    # Handle s1 = s2 condition
    slopes[0] = slopes[1]

    # Clip the slopes to the range [smin, smax]
    clipped_slopes = torch.clamp(slopes, smin, smax)

    # Calculate new nodal values
    new_nodal_values = torch.zeros_like(nodal_values)
    new_nodal_values[0] = nodal_values[0]  # Set the first nodal value as it is

    # Compute cumulative sum for the adjustment using the clipped slopes
    new_nodal_values[1:] = new_nodal_values[0] + torch.cumsum(clipped_slopes[:-1] * delta_t, dim=0)
    # Apply the projection of clipped slopes to nodal values
    # for n in range(1, len(nodal_values)): # 
    #     new_nodal_values[n] = new_nodal_values[n - 1] + clipped_slopes[n - 1] * (knot_positions[n] - knot_positions[n - 1])

    # Adjust to preserve the mean
    mean_adjustment = torch.mean(nodal_values) - torch.mean(new_nodal_values)
    new_nodal_values += mean_adjustment

    return new_nodal_values

### COMMENT: I DONT THINK WE NEED GRID AND SIZE AS AN ARGUMENT IN THIS FN 
# AS WE ARE NOT USING THEM ANYWHHERE 
def initialize_coeffs(init, grid_tensor, grid, size):
        """The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since B1 splines are interpolators)."""
        
        if init == 'identity':
            coefficients = grid_tensor
        elif init == 'zero':
            coefficients = torch.zeros(grid_tensor.shape)
        elif init == 'relu':
            coefficients = F.relu(grid_tensor)
        elif init == 'absolute_value':
            coefficients = torch.abs(grid_tensor)
            
        elif init == 'maxmin':
            # initalize half of the activations with the absolute and the other half with the 
            # identity. This is similar to maxmin because max(x1, x2) = (x1 + x2)/2 + |x1 - x2|/2 
            # and min(x1, x2) = (x1 + x2)/2 - |x1 - x2|/2
            coefficients = torch.zeros(grid_tensor.shape)
            coefficients[::2, :] = (grid_tensor[::2, :]).abs()
            coefficients[1::2, :] = grid_tensor[1::2, :]
        
        else:
            raise ValueError('init should be in [identity, relu, absolute_value, maxmin, max_tv].')
        
        return coefficients

## TO INCORPORATE NON-UNIFORM CASE, WE WOULD HAVE TO MAKE SOME CHANGES HERE AS WELL!
class LinearSpline_Func(torch.autograd.Function): 
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients_vect, grid_tensor, zero_knot_indexes, size, even):
        # Clamping x to be within the range of the grid
        x_clamped = x.clamp(min=grid_tensor.min(), max=grid_tensor.max())

        # Finding left_index which represents the index of the left grid point
        left_index = torch.searchsorted(grid_tensor, x_clamped, right=False) - 1

        # Explicitly handle negative left_index values by ensuring they start from 0
        left_index = left_index.where(left_index >= 0, torch.tensor(0, device=x.device))

        # Calculate fracs based on non-uniform intervals
        fracs = (x_clamped - grid_tensor[left_index]) / (grid_tensor[left_index + 1] - grid_tensor[left_index])

        # Now calculate indexes for the coefficients
        # This gives the indexes (in coefficients_vect) of the left coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + left_index).long()

        # Now compute the activation output using the two coefficients
        activation_output = coefficients_vect[indexes + 1] * fracs + \
            coefficients_vect[indexes] * (1 - fracs)

        if even: # I am not fully sure about this yet!
            activation_output = activation_output + (grid_tensor.max() - grid_tensor.min()) / 2  # Adjust if needed for even grid

        ctx.save_for_backward(fracs, coefficients_vect, indexes, grid_tensor)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):
        fracs, coefficients_vect, indexes, grid_tensor = ctx.saved_tensors

        # Calculate the gradients with respect to x
        # grad_x represents the slope between the two coefficients
        grad_x = (coefficients_vect[indexes + 1] - coefficients_vect[indexes]) / (grid_tensor[indexes + 1] - grid_tensor[indexes]) * grad_out

        # Next, add the gradients with respect to each coefficient
        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        
        # Right coefficients gradients
        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 1,
                                            (fracs * grad_out).view(-1))
        # Left coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                            ((1 - fracs) * grad_out).view(-1))

        return grad_x, grad_coefficients_vect, None, None, None, None

class LinearSplineSlopeConstrained(ABC, nn.Module): ### changes mainly here!
    """
    Class for LinearSpline activation functions

    Args:
        mode (str): 'conv' (convolutional) or 'fc' (fully-connected).
        num_activations (int) : number of activation functions
        size (int): number of coefficients of spline grid; the number of knots K = size - 2.
        range_ (float) : positive range of the B-spline expansion. B-splines range = [-range_, range_].
        init (str): Function to initialize activations as (e.g. 'relu', 'identity', 'absolute_value').
        smin (float): minimum bound on the slope
        smax (float): maximum bound on the slope
        lipschitz_constrained (bool): Constrain the activation to be 1-Lipschitz
    """

    def __init__(self, mode, num_activations, size, range_, init,smin, smax,
                lipschitz_constrained, grid_values =None,
                uniform_grid = False, **kwargs):

        if mode not in ['conv', 'fc']:
            raise ValueError('Mode should be either "conv" or "fc".')
        if int(num_activations) < 1:
            raise TypeError('num_activations needs to be a '
                            'positive integer...')

        super().__init__()

        # Initialize the grid_tensor based on uniform_grid and grid_values
        if uniform_grid:
            # Generate a uniform grid
            self.grid_tensor = torch.linspace(-range_, range_, size).expand((num_activations, size))
            self.grid = torch.Tensor([2 * range_ / (size - 1)])  # uniform spacing for D2_filter
        else:
            if grid_values is not None:
                # Use provided grid_values, ensuring correct length
                assert len(grid_values) == size, "grid_values length must match size"
                self.grid_tensor = torch.tensor(grid_values).expand((num_activations, size))
            else:
                # Generate a non-uniform grid by sampling random points within the range
                random_grid = torch.rand(size) * (2 * range_) - range_  # Values within [-range_, range_]
                random_grid, _ = torch.sort(random_grid)  # Sort to ensure monotonicity
                self.grid_tensor = random_grid.expand((num_activations, size))

            # Calculate differences for both provided and random non-uniform grids
            self.grid = self.grid_tensor[:, 1:] - self.grid_tensor[:, :-1]


        self.mode = mode
        self.size = int(size)
        self.even = self.size % 2 == 0
        self.num_activations = int(num_activations)
        self.init = init
        self.range_ = float(range_)
        self.smin = smin
        self.smax = smax
        # grid = 2 * self.range_ / (self.size-1) # defined earlier 
        # self.grid = torch.Tensor([grid]) # define earlier 
        self.init_zero_knot_indexes()
        # self.D2_filter = Tensor([1, -2, 1]).view(1, 1, 3).div(self.grid)### defined below
        self.lipschitz_constrained = lipschitz_constrained # do we need it this time?

        ### D2 filters 
        # Filter for computing (earlier) relu_slopes  function (2nd-order finite differences) UPDATE (changed name to slope_difference)
        ### update: I don't think we will be using this anywhere anymore now. let's see
        if uniform_grid:
            # Use uniform spacing
            self.D2_filter = Tensor([1, -2, 1]).view(1, 1, 3).div(self.grid)  # Using the uniform grid spacing
        else:
            # Use differences for the non-uniform grid
            self.D2_filter = Tensor([1, -2, 1]).view(1, 1, 3).div(self.grid_tensor[:, 1:] - self.grid_tensor[:, :-1])

        ### tensor with locations of spline coefficients
        #we have already done this above in uniform condition of if-else statement
        #self.grid_tensor = torch.linspace(-self.range_, self.range_, self.size).expand((self.num_activations, self.size))

        ### INITIALISE SPLINE COEFFICIENTS
        # (maybe we might need to make some changes in the initialize_coeffs function as well. we will see)
        coefficients = initialize_coeffs(init, self.grid_tensor, self.grid, self.size)  # spline coefficients
        # Need to vectorize coefficients to perform specific operations
        # size: (num_activations*size)
        self.coefficients_vect = nn.Parameter(coefficients.contiguous().view(-1))
        self.scaling_coeffs_vect = nn.Parameter(torch.ones((1, self.num_activations, 1, 1)))

        ### SLOPES
        # Define slopes based on the initialized coefficients and grid_tensor
        delta_f = self.coefficients_vect[:, 1:] - self.coefficients_vect[:, :-1]
        delta_t = self.grid_tensor[:, 1:] - self.grid_tensor[:, :-1]
        slopes = torch.zeros((self.num_activations, self.size), device=self.coefficients_vect.device)
        # Calculate slopes using the finite-difference formula, ensuring the boundary condition s1 = s2
        slopes[:, 1:] = delta_f / delta_t
        slopes[:, 0] = slopes[:, 1]  # Boundary condition (s1 = s2)
        # Store slopes as a member of the class
        self.slopes = slopes  # Tensor of initial slopes


    def init_zero_knot_indexes(self):
        """ Initialize indexes of zero knots of each activation.
        """
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = torch.arange(0, self.num_activations)
        self.zero_knot_indexes = (activation_arange * self.size + (self.size // 2))

    @property
    def coefficients(self):
        """ B-spline coefficients. """
        return self.coefficients_vect.view(self.num_activations, self.size)
    
    @property
    def lipschitz_coefficients(self): # UPDATE: I HAVE TO CHANGE ITS NAME TO SOMETHING ELSE, LEAVING IT LIPSCHITZ_COEFFICIENTS FOR THE TIME BEING
        """Projection of B-spline coefficients such that the slope is constrained"""
        # return slope_clipping(self.coefficients, self.grid.item())
        return project_slopes_vectorized(nodal_values=self.coefficients, knot_positions=self.grid_tensor, smin = self.smin, smax = self.smax)
    
    @property
    def lipschitz_coefficients_vect(self):
        """Projection of B-spline coefficients such that they are 1-Lipschitz"""
        return self.lipschitz_coefficients.contiguous().view(-1)

    @property
    def slope_difference(self): ### THIS WILL CHANGE FOR SURE!
        """ Getting vector of difference of slopes which we need later for TV2 condition.

        ref: Page 17 of the paper:
            "Controlled learning of Pointwise Non-linearities in Neural-Network-like architectures" by Unser et. al.
        """
        diff_slopes = self.slope[1:] - self.slope[:-1]  
        return diff_slopes

    def reshape_forward(self, x):
        """
        Reshape inputs for deepspline activation forward pass, depending on
        mode ('conv' or 'fc').
        """
        input_size = x.size()
        if self.mode == 'fc':
            if len(input_size) == 2:
                # one activation per conv channel
                # transform to 4D size (N, num_units=num_activations, 1, 1)
                x = x.view(*input_size, 1, 1)
            else:
                raise ValueError(f'input size is {len(input_size)}D but should be 2D')
        else:
            assert len(input_size) == 4, 'input to activation should be 4D (N, C, H, W) if mode="conv".'

        return x

    def reshape_back(self, output, input_size):
        """
        Reshape back outputs after deepspline activation forward pass,
        depending on mode ('conv' or 'fc').
        """
        if self.mode == 'fc':
            # transform back to 2D size (N, num_units)
            output = output.view(*input_size)

        return output


    def forward(self, input):
        """
        Args:                                                                                                                                                                                                                            
            input (torch.Tensor):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """
        input_size = input.size()
        x = self.reshape_forward(input)
        assert x.size(1) == self.num_activations, \
            'Wrong shape of input: {} != {}.'.format(x.size(1), self.num_activations)


        ## transfering the grid tensor and then zero_knot_tensors to the same device as coefficients_vect
        # grid = self.grid.to(self.coefficients_vect.device) 
        grid_tensor = self.grid_tensor.to(self.coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(grid_tensor.device)

        x = x.mul(self.scaling_coeffs_vect)### refer to equation (14) in the paper (section 3.3.2 Scaling Parameter)

        if self.lipschitz_constrained: ### THIS I NEED TO LOOK INTO! IM NOT ENTIRELY SURE ABOUT THIS!
            output = LinearSpline_Func.apply(x, self.lipschitz_coefficients_vect, grid_tensor, zero_knot_indexes, \
                                        self.size, self.even)

        else:
            output = LinearSpline_Func.apply(x, self.coefficients_vect, grid_tensor, zero_knot_indexes, \
                                        self.size, self.even)

        output = output.div(self.scaling_coeffs_vect) ### refer to equation (14) in the paper (section 3.3.2 Scaling Parameter) 
        output = self.reshape_back(output, input_size)

        return output


    def extra_repr(self):
        """ repr for print(model) """

        s = ('mode={mode}, num_activations={num_activations}, '
             'init={init}, size={size}, grid={grid[0]:.3f}, '
             'lipschitz_constrained={lipschitz_constrained}.')

        return s.format(**self.__dict__)

    def totalVariation(self, **kwargs):
        """
        Computes the second-order total-variation regularization.
        Reference: "Controlled learning of Pointwise Non-linearities in Neural-Network-like architectures" by Unser et. al. 
        TV(2)(deepsline) = ||slope_difference_vector||_1.
        """
        return self.slope_difference.norm(1, dim=1)