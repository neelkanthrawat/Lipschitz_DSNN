"""
Wrap around nn.Module with all LinearSpline functionalities.

"""

import torch
import math
import numpy as np
import torch.nn as nn
from torch import Tensor
# from activations.linearspline import LinearSpline
from activations.linearspline_slope_constraint import LinearSplineSlopeConstrained
from activations.groupsort import GroupSort
from activations.householder import HouseHolder
from activations.basic_activations import AbsoluteValue, LipschitzPReLU


class BaseModel(nn.Module):
    """
    Parent class for neural networks.
    """

    def __init__(self,
                 activation_type=None,
                 spline_size=None,
                 spline_range=None,
                 spline_init=None,
                 slope_min = None,
                 slope_max = None,
                 slope_constrained=True,
                 groupsort_groupsize=None,
                 prelu_init=None,
                 **kwargs):

        super().__init__()

        # general attributes
        self.activation_type = activation_type
        
        # linearspline attributes
        self.spline_init = spline_init
        self.spline_size = spline_size
        self.spline_range = spline_range
        self.slope_min = slope_min
        self.slope_max = slope_max
        self.slope_constrained = slope_constrained

        # groupsort attributes
        self.groupsort_groupsize = groupsort_groupsize

        #PReLU attributes
        self.prelu_init = prelu_init

        self.linearspline = None
        if self.activation_type == 'linearspline_slope_constrained':
            self.linearspline = LinearSplineSlopeConstrained
            

    def init_activation_list(self, activation_specs, **kwargs):
        """
        Initialize list of activation modules (linearspline or standard).

        Args:
            activation_specs (list):
                list of 2-tuples (mode[str], num_activations[int]);
                mode can be 'conv' (convolutional) or 'fc' (fully-connected);
        Returns:
            activations (nn.ModuleList)
        """
        assert isinstance(activation_specs, list), \
            f'activation_specs type: {type(activation_specs)}'

        if self.using_splines:
            activations = nn.ModuleList()
            for mode, num_activations in activation_specs:
                activations.append(self.linearspline(mode=mode,
                                    num_activations=num_activations,
                                    size=self.spline_size,
                                    range_=self.spline_range,
                                    init=self.spline_init,
                                    smin = self.slope_min,
                                    smax = self.slope_max,
                                    slope_constrained=self.slope_constrained,
                                    ))

        else:
            activations = self.init_standard_activations(activation_specs)

        return activations
    

    def init_activation(self, activation_specs, **kwargs):
        """
        Initialize a single activation module (linearspline or standard).
        Args:
            activation_specs (tuple):
                2-tuple (mode[str], num_activations[int]);
                mode can be 'conv' (convolutional) or 'fc' (fully-connected)
        Returns:
            activation (nn.Module)
        """
        assert isinstance(activation_specs, tuple), \
            f'activation_specs type: {type(activation_specs)}'
        activation = self.init_activation_list([activation_specs], **kwargs)[0]
        return activation
    

    def init_standard_activations(self, activation_specs, **kwargs):
        """
        Initialize standard activation modules.

        Args:
            activation_specs :
                list of pairs (mode, num_channels/neurons);

        Returns:
            activations (nn.ModuleList)
        """
        activations = nn.ModuleList()

        if self.activation_type == 'relu':
            for i in range(len(activation_specs)):
                activations.append(nn.ReLU())

        elif self.activation_type == 'elu':
            for i in range(len(activation_specs)):
                activations.append(nn.ELU())

        elif self.activation_type == 'prelu':
            for _, num_activations in activation_specs:
                activations.append(LipschitzPReLU(num_parameters=num_activations, init=self.prelu_init))

        elif self.activation_type == 'absolute_value':
            for i in range(len(activation_specs)):
                activations.append(AbsoluteValue())

        elif self.activation_type == 'groupsort':
            for _, num_activations in activation_specs:
                activations.append(GroupSort(num_units=num_activations // self.groupsort_groupsize, axis=1))

        elif self.activation_type == 'householder':
            for mode, num_activations in activation_specs:
                activations.append(HouseHolder(mode, num_activations)) 

        else:
            raise ValueError(f'{self.activation_type} '
                             'is not a valid parameter...')

        return activations
    

    @property
    def modules_linearspline(self):
        """
        Yields all deepspline modules in the network.
        """
        for module in self.modules():
            if isinstance(module, self.linearspline):
                yield module
                

    @property
    def using_splines(self):
        """
        True if using linearspline activations.
        """
        return (self.linearspline is not None)
    

    @property
    def device(self):
        """
        Get the network's device (torch.device).

        Returns the device of the first found parameter.
        """
        return next(self.parameters()).device
    

    def initialization(self, init_type):
        """
        Initializes the network weights with 'He', 'Xavier', or a
        custom gaussian initialization.
        """

        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose2d):
                if init_type == 'He_uniform':
                    nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_out', nonlinearity='relu')
                elif init_type == 'He_normal':
                    nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_out', nonlinearity='relu')
                elif init_type == 'Xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif init_type == 'Xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                elif init_type == 'orthonormal':
                    nn.init.orthogonal_(module.weight)
                elif init_type == 'gan':
                    module.weight.data.normal_(0, 0.02)
                    module.bias.data.zero_()
                elif init_type == 'identity':
                    if isinstance(module, nn.Conv2d):
                        # initialize weights close to identity with some small additional noise
                        num_channels = module.weight.shape[0:2]
                        random_weights = torch.randn(num_channels) * 0.0075
                        if num_channels[0] == 1 or num_channels[1] == 1:
                            random_weights = random_weights - torch.mean(random_weights) + 1 / math.sqrt(max(num_channels))
                            random_weights = random_weights / torch.norm(random_weights)
                        else:
                            random_weights = random_weights - torch.mean(random_weights, dim=0) + 1 / max(num_channels)
                            random_weights = random_weights / torch.linalg.norm(random_weights, 2)
                        identity_kernel = torch.zeros(module.weight.shape)
                        identity_kernel[:,:,1,1] = random_weights
                    if isinstance(module, nn.Linear):
                        raise ValueError('Identity init is not compatible with fully connected')
                else:
                    raise ValueError('init_type {} is invalid.'.format(init_type))
            

    def get_num_params(self):
        """
        Returns the total number of network parameters.
        """
        num_params = 0
        for param in self.parameters():
            num_params += torch.numel(param)

        return num_params


    ##########################################################################
    # linearspline: regularization and sparsification
    
    def TV2(self):
        """
        Computes the sum of the TV(2) (second-order total-variation)
        semi-norm of all linearspline activations in the network.

        Returns:
            tv2 (0d Tensor):
                tv2 = sum(TV(2))
        """
        tv2 = Tensor([0.]).to(self.device)

        for module in self.modules():
            if isinstance(module, LinearSpline):
                module_tv2 = module.totalVariation(mode='additive')
                tv2 = tv2 + module_tv2.norm(p=1)

        return tv2[0]  # 1-tap 1d tensor -> 0d tensor