import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractproperty, abstractmethod


### Slope projection function (To enforce the slope constraint)
def project_slopes_vectorized(fn_values, knot_positions, 
                            smin, smax):
    """
    Project the slopes of a given set of fn (coefficient) values with respect to non-uniform knot positions.

    Parameters:
    - fn_values: Tensor of coefficient values (f_n).
    - knot_positions: Tensor of corresponding knot positions (t_n).
    - smin: Minimum allowable slope.
    - smax: Maximum allowable slope.

    Returns:
    - new_fn_values: Tensor of adjusted nodal values after slope projection.
    """

    # Number of nodal values (assuming nodal_values is 2D)
    N = fn_values.size(1)  # Change to size(1) if it's a 1D tensor

    # # Calculate the differences in nodal values and knot positions
    delta_f = fn_values[:, 1:] - fn_values[:, :-1]
    delta_t = knot_positions[:, 1:] - knot_positions[:, :-1]

    # Calculate slopes
    slopes = torch.zeros_like(fn_values)  # Keep the same shape as nodal_values
    slopes[:, 1:] = delta_f / delta_t

    # Handle s1 = s2 condition
    slopes[:, 0] = slopes[:, 1]  # Use the first computed slope

    # Clip the slopes to the range [smin, smax]
    clipped_slopes = torch.clamp(slopes, smin, smax)

    # Calculate new nodal values
    new_fn_values = torch.zeros_like(fn_values)
    new_fn_values[:, 0] = fn_values[:, 0]  # Set the first nodal value as it is

    # Compute cumulative sum for the adjustment using the clipped slopes
    for i in range(1, N):
        new_fn_values[:, i] = new_fn_values[:, i - 1] + clipped_slopes[:, i] * (knot_positions[:, i] - knot_positions[:, i - 1])

    # Adjust to preserve the mean
    mean_adjustment = torch.mean(fn_values) - torch.mean(new_fn_values)
    new_fn_values += mean_adjustment

    return new_fn_values


### COMMENT: I DONT THINK WE NEED GRID AND SIZE AS AN ARGUMENT IN THIS FN 
# AS WE ARE NOT USING THEM ANYWHHERE 
def initialize_coeffs(init, nodal_val_loc_tensor, grid, size, manual_init_tensor=None):
        """The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since B1 splines are interpolators)."""
        
        if init == 'identity':
            coefficients = nodal_val_loc_tensor
        elif init == 'manual' and manual_init_tensor.all()!=None:
            coefficients = manual_init_tensor
        elif init == "double":
            coefficients = 2* nodal_val_loc_tensor
        elif init == "random": 
            coefficients = 1.1*torch.rand_like(nodal_val_loc_tensor) + nodal_val_loc_tensor
            #print('shape of nodal vector is:');print(nodal_val_loc_tensor.shape)
            # coefficients = torch.zeros_like(nodal_val_loc_tensor)  # Start with all zeros
            # coefficients[:,0] = nodal_val_loc_tensor[:,0]       # First row as identity
            # coefficients[:,-3] = nodal_val_loc_tensor[:,-3]     # Last row as identity
            # coefficients[:,3:-3] = 1.5 * torch.rand_like(nodal_val_loc_tensor[:,3:-3]) \
            #                     + nodal_val_loc_tensor[:,3:-3]  # Middle rows as random
        elif init == 'zero':
            coefficients = torch.zeros(nodal_val_loc_tensor.shape)
        elif init == 'relu':
            coefficients = F.relu(nodal_val_loc_tensor)
        elif init == 'absolute_value':
            coefficients = torch.abs(nodal_val_loc_tensor)
            
        elif init == 'maxmin':
            # initalize half of the activations with the absolute and the other half with the 
            # identity. This is similar to maxmin because max(x1, x2) = (x1 + x2)/2 + |x1 - x2|/2 
            # and min(x1, x2) = (x1 + x2)/2 - |x1 - x2|/2
            coefficients = torch.zeros(nodal_val_loc_tensor.shape)
            coefficients[::2, :] =  (nodal_val_loc_tensor[::2, :]).abs()
            coefficients[1::2, :] =  nodal_val_loc_tensor[1::2, :]
        
        else:
            raise ValueError('init should be in [identity,zero, relu, absolute_value, maxmin, max_tv].')
        return coefficients

## TO INCORPORATE NON-UNIFORM CASE, WE WOULD HAVE TO MAKE SOME CHANGES HERE AS WELL!
class LinearSpline_Func(torch.autograd.Function): 
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients_vect, nodal_val_loc_tensor, 
                zero_knot_indexes, size, even
                , want_grad_x=False):
        
        # print("x is:"); print(x)
        ### Step 1: Find the index of the left and right term's posn/nodal point location
        nodal_val_loc_tensor = nodal_val_loc_tensor.contiguous()
        x_sq_and_transpose = x.squeeze(-1).squeeze(-1).transpose(0, 1).contiguous()# squeezed and transposed
        left_indices = torch.searchsorted(nodal_val_loc_tensor, x_sq_and_transpose)-1#Shape:[num_activ, batch_size]
        # clipping left indices (worked out in notes)
        left_indices = torch.clamp(left_indices, min=0, max =size-2)#shape:(num_activ,batch_size)
        
        ### Step 2: calcuating fractions/left and right basis
        num_activations, _ = left_indices.shape
        activation_indices = torch.arange(num_activations).unsqueeze(1)  # Shape: [num_activations, 1]
        # calculating left and right nodal points (t_j, t_{j+1})
        left_values =nodal_val_loc_tensor[activation_indices, left_indices]# Shape: [num_activations, batch_size]
        right_values = nodal_val_loc_tensor[activation_indices, left_indices+1]# Shape: [num_activations, batch_size]
        # print("left_values:"); print(left_values)
        # print("right_values:"); print(right_values)
        # Calculate the left basis
        left_basis = (right_values - x_sq_and_transpose)/ (right_values - left_values)
        # print("left basis is:"); print(left_basis)
        # indices for coefficient vector:
        index_coeffs = left_indices + zero_knot_indexes.unsqueeze(1)
        # print("coeff with left basis"); print(coefficients_vect[index_coeffs])
        # print("coeff with right basis"); print(coefficients_vect[index_coeffs+1])

        ### Step 3: Compute activation output with 2 coefficients and corresponding basis
        activation_output = coefficients_vect[index_coeffs] * left_basis + \
                            coefficients_vect[index_coeffs+1] * (1-left_basis)

        # reshape the output:
        activation_output = activation_output.transpose(0,1).view(x.shape)
        # print("activation_output:"); print(activation_output)

        ### Step 4: save for backward propagation
        ctx.save_for_backward(left_basis, coefficients_vect, index_coeffs, 
                            nodal_val_loc_tensor,activation_indices,left_indices)# add 2 more things
        
        # we need the following to calculate the jacobian explicitly
        grad_x_temp = (coefficients_vect[index_coeffs + 1] - coefficients_vect[index_coeffs]) / (nodal_val_loc_tensor[activation_indices, left_indices+1] - nodal_val_loc_tensor[activation_indices, left_indices])
        
        return activation_output, grad_x_temp

    @staticmethod
    def backward(ctx, grad_out, grad_x_temp_out):

        # ADDED "activation_indices" and "left_indices" in the code below
        fracs, coefficients_vect, indexes, nodal_val_loc_tensor,activation_indices, left_indices = ctx.saved_tensors
        # note: fracs is left basis
        # Calculate the gradients with respect to x
        grad_x = (coefficients_vect[indexes + 1] - coefficients_vect[indexes]) / (nodal_val_loc_tensor[activation_indices, left_indices+1] - nodal_val_loc_tensor[activation_indices, left_indices]) * grad_out
        # Next, add the gradients with respect to each coefficient
        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1),
                                            (fracs * grad_out).reshape(-1))
        # right coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1)+1,
                                            ((1 - fracs) * grad_out).reshape(-1))

        # return grad_x, grad_coefficients_vect, None, None, None, None, None
        return None,grad_coefficients_vect, None, None, None, None
        # return grad_x,grad_coefficients_vect, None, None, None, None


class LinearSplineSlopeConstrained(ABC, nn.Module): ### changes mainly here!
    """
    Class for LinearSpline activation functions

    Args:
        mode (str): 'conv' (convolutional) or 'fc' (fully-connected).
        num_activations (int) : number of activation functions
        size (int): number of nodal values, f, in the spline grid; the number of knots K = size - 2.
        range_ (float) : positive range of the B-spline expansion. B-splines range = [-range_, range_].
        init (str): Function to initialize activations as (e.g. 'relu', 'identity', 'absolute_value').
        smin (float): minimum bound on the slope
        smax (float): maximum bound on the slope
        ### I will change the name of the next variable for sure.
        slope_constrained (bool): Constrain the activation to be 1-slope
    """

    def __init__(self, mode, num_activations, size, range_, init,smin, smax,
                slope_constrained, grid_values =None, manual_init_fn_tensor=None, print_initialisation=0,
                uniform_grid = False, **kwargs):

        if mode not in ['conv', 'fc']:
            raise ValueError('Mode should be either "conv" or "fc".')
        if int(num_activations) < 1:
            raise TypeError('num_activations needs to be a '
                            'positive integer...')

        super().__init__()

        # Initialize the nodal_val_loc_tensor based on uniform_grid and grid_values
        if uniform_grid:
            # Generate a uniform grid
            self.nodal_val_loc_tensor = torch.linspace(-range_, range_, size).expand((num_activations, size))
            self.grid = torch.Tensor([2 * range_ / (size - 1)])  # uniform spacing for D2_filter
        else:
            if grid_values is not None:
                # Use provided grid_values, ensuring correct length
                assert len(grid_values) == size, "grid_values length must match size"
                self.nodal_val_loc_tensor = torch.tensor(grid_values).expand((num_activations, size))

            else:
                # Generate a non-uniform grid by sampling random points within the range
                random_grid = torch.rand(size) * (2 * range_) - range_  # Values within [-range_, range_]
                random_grid, _ = torch.sort(random_grid)  # Sort to ensure monotonicity
                self.nodal_val_loc_tensor = random_grid.expand((num_activations, size))
                

            # Calculate differences for both provided and random non-uniform grids
            ### imma keep the name grid for now, but later, I will change it to diff_nodal_val_loc
            self.grid = self.nodal_val_loc_tensor[:, 1:] - self.nodal_val_loc_tensor[:, :-1]

        self.mode = mode
        self.size = int(size)
        self.even = self.size % 2 == 0
        self.num_activations = int(num_activations)
        self.init = init
        self.range_ = float(range_)
        self.smin = smin
        self.smax = smax
        self.init_zero_knot_indexes()
        self.slope_constrained = slope_constrained # do we need it this time?
        self.manual_init_tensor = manual_init_fn_tensor

        ### INITIALISE SPLINE COEFFICIENTS (NODAL VALUES, fn)
        if self.init == 'manual':
            coefficients = initialize_coeffs(init, self.nodal_val_loc_tensor, self.grid, self.size, 
                                        manual_init_tensor=self.manual_init_tensor)
        else:    
            coefficients = initialize_coeffs(init, self.nodal_val_loc_tensor, self.grid, self.size)  # spline coefficients
        
        # Need to vectorize coefficients to perform specific operations
        # size: (num_activations*size)
        self.coefficients_vect = nn.Parameter(coefficients.contiguous().view(-1))
        if print_initialisation:
            print("initial nodal_val_locs:", self.nodal_val_loc_tensor)
            print("initial fn values:", self.coefficients_vect)
        self.grad_x_temp = 0 # We need this for slope jacobian which in turn is needed in cal of loss (1d case)


    def init_zero_knot_indexes(self):
        """ Initialize indexes of zero knots of each activation.
        """
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = torch.arange(0, self.num_activations)
        self.zero_knot_indexes = (activation_arange * self.size)

    @property
    def coefficients(self):
        """ B-spline coefficients. """
        return self.coefficients_vect.view(self.num_activations, self.size)
    
    @property
    def slope_constrained_coefficients(self): # UPDATE: I HAVE TO CHANGE ITS NAME TO SOMETHING ELSE, LEAVING IT slope_constrained_coefficients FOR THE TIME BEING
        """Projection such that the slope is constrained in the range [smin, smax]"""
        return project_slopes_vectorized(fn_values=self.coefficients, 
                                        knot_positions=self.nodal_val_loc_tensor, 
                                        smin = self.smin, smax = self.smax)
    
    @property
    def slope_constrained_coefficients_vect(self):
        """Projection s.t slope of f_spline is constrained in the range [smin, smax]"""
        return self.slope_constrained_coefficients.contiguous().view(-1)
    
    # @property
    def slopes_tensor(self, for_projected_coeffs=False):
        ''' returns slopes'''
        # Calculate the differences in nodal values and knot positions
        delta_f = self.coefficients[:, 1:] - self.coefficients[:, :-1]
        delta_t = self.nodal_val_loc_tensor[:, 1:] - self.nodal_val_loc_tensor[:, :-1]

        # Calculate slopes
        slopes = torch.zeros_like(self.coefficients)  # Keep the same shape as nodal_values
        slopes[:, 1:] = delta_f / delta_t

        # Handle s1 = s2 condition
        slopes[:, 0] = slopes[:, 1]  # Use the first computed slope
        if for_projected_coeffs:
            return torch.clamp(slopes, self.smin, self.smax)
        return slopes# these values are not quite correct for constraint case due to the floating point error
    
    # @property
    def slopes_contiguous(self, for_projected_coeffs=False):
        get_slope= self.slopes_tensor(for_projected_coeffs=for_projected_coeffs)
        return get_slope.contiguous().view(-1)

    @property
    def slope_difference(self):
        """ Getting vector of difference of slopes which we need later for TV2 condition.

        ref: Page 17 of the paper:
            "Controlled learning of Pointwise Non-linearities in Neural-Network-like architectures" by Unser et. al.
        """
        get_slopes = self.slopes_contiguous(for_projected_coeffs=True)
        diff_slopes = get_slopes[1:] - get_slopes[:-1] 
        # print("diff_slope:"); print(diff_slopes)
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
        nodal_val_loc_tensor = self.nodal_val_loc_tensor.to(self.coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(nodal_val_loc_tensor.device)
        # print("in the forward function:")
        # print("what im calling nodal_val_loc_tensor is:"); print(nodal_val_loc_tensor)

        # x = x.mul(self.scaling_coeffs_vect)### refer to equation (14) in the paper (section 3.3.2 Scaling Parameter)

        if self.slope_constrained: ### THIS I NEED TO LOOK INTO! IM NOT ENTIRELY SURE ABOUT THIS!
            ## Do i need to change the self.slope to self.slope_new which are new constrained slopes
            # self.slopes = self.slopes_vector
            output, grad_x_temp = LinearSpline_Func.apply(x, 
                                    self.slope_constrained_coefficients_vect, nodal_val_loc_tensor,
                                    zero_knot_indexes, \
                                    self.size, self.even)

        else:
            output, grad_x_temp = LinearSpline_Func.apply(x, self.coefficients_vect, 
                                    nodal_val_loc_tensor, zero_knot_indexes, \
                                    self.size, self.even)

        # output = output.div(self.scaling_coeffs_vect) ### refer to equation (14) in the paper (section 3.3.2 Scaling Parameter) 
        output = self.reshape_back(output, input_size)
        self.grad_x_temp = grad_x_temp

        return output

    # def extra_repr(self):
    #     """ Custom repr for print(model)."""
    #     s = ('mode={mode}, num_activations={num_activations}, '
    #         'init={init}, size={size}, '
    #         'grid_min={self.nodal_val_loc_tensor}, grid_max={self.nodal_val_loc_tensor}, '
    #         'slope_constrained={slope_constrained}, '
    #         'coefficients_vect_shape={self.coefficients_vect.shape}, '
    #         'slopes_mean={slopes.mean():.3f}, slopes_std={slopes.std():.3f}.')
    #     return s.format(**self.__dict__)
    def extra_repr(self):
        """ repr for print(model) """

        s = ('mode={mode}, num_activations={num_activations}, '
             'init={init}, size(Num of nodes (ti))={size}, '# removed grid={grid} from here
             'slope_constrained={slope_constrained}.'
             )

        return s.format(**self.__dict__)


    def totalVariation(self, **kwargs):
        """
        Computes the second-order total-variation regularization.
        Reference: "Controlled learning of Pointwise Non-linearities in Neural-Network-like architectures" by Unser et. al. 
        TV(2)(deepsline) = ||slope_difference_vector||_1.
        """
        return self.slope_difference.norm(1)#self.slope_difference.norm(1, dim=1)