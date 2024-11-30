import torch
import torch.nn as nn
from architectures.base_model_2 import BaseModel
from layers.lipschitzlinear import LipschitzLinear
from projections.fc_projections import identity, bjorck_orthonormalize_fc, layerwise_orthogonal_fc

### create the slope constrained neural network!
class SimpleFCSlopeConstrained(BaseModel):
    """simple architecture for a fully-connected network based classifier"""
    def __init__(self, network_parameters,bias=0, **params):
        
        super().__init__(**params)

        modules = nn.ModuleList()
        # print("network parameters are:"); print(network_parameters)
        if network_parameters['projection'] == 'no_projection':
            print("no orthonormalisation. projection= identity")
            projection = identity
        elif network_parameters['projection'] == 'orthonormalize':
            print("Orthonormalisation will take place.")
            if 'bjorck_iter' in network_parameters:
                print("Bjoerck and Bowie orthonormalisation will take place")
                def proj(weights, lipschitz_goal):
                    return bjorck_orthonormalize_fc(weights, lipschitz_goal,
                                                    beta=0.5, iters=network_parameters['bjorck_iter'])
                projection = proj
            elif 'LOT' in network_parameters:
                print("LOT orthonormalization will take place")
                def proj_2(weights, lipschitz_goal):
                    return layerwise_orthogonal_fc(weights,lipschitz_goal, 
                                                beta = 0.5, iters = network_parameters['LOT']['LOT_iter'])
                projection = proj_2
            else:
                print("Bjorck and Bowie orthonormalisation will take place")
                projection = bjorck_orthonormalize_fc
        else:
            raise ValueError('Projection type is not valid')

        layer_sizes = network_parameters['layer_sizes']


        for i in range(len(layer_sizes)-1):
            # print(f"i is: {i} and layer_sizes[i] is: {layer_sizes[i+1]}")
            modules.append(LipschitzLinear(1, projection, layer_sizes[i], layer_sizes[i+1],bias))# 1 corresponds to lipschitz constant 1
            modules.append(self.init_activation(('fc', layer_sizes[i+1]))) ### activation should be i


        # modules.append(LipschitzLinear(1, projection, layer_sizes[-2], layer_sizes[-1]))
        # modules.append(nn.Sigmoid())
        self.initialization(init_type=network_parameters['weight_initialization'])
        self.num_params = self.get_num_params()

        self.layers = nn.Sequential(*modules)
        

    def forward(self, x):
        """ """
        ### apply layers and sigmoid function
        x= self.layers(x)
        #x = torch.sigmoid(x)
        return x
    # def forward(self, x):
    #     """Forward pass with intermediate outputs printed."""
    #     print("Input:\n", x)
    #     for i, layer in enumerate(self.layers):
    #         x = layer(x)
    #         print(f"-----After layer {i} ({layer.__class__.__name__}):\n{x}")
    #     return x

### This was the old code. Not quite sure if this is correct
# class SimpleFC(BaseModel):
#     """simple architecture for a fully-connected network"""
#     def __init__(self, network_parameters, **params):
        
#         super().__init__(**params)

#         modules = nn.ModuleList()

#         if network_parameters['projection'] == 'no_projection':
#             projection = identity
#         elif network_parameters['projection'] == 'orthonormalize':
#             if 'bjorck_iter' in network_parameters:
#                 def proj(weights, lipschitz_goal):
#                     return bjorck_orthonormalize_fc(weights, lipschitz_goal, beta=0.5, iters=network_parameters['bjorck_iter'])
#                 projection = proj
#             else:
#                 projection = bjorck_orthonormalize_fc
#         else:
#             raise ValueError('Projection type is not valid')

#         layer_sizes = network_parameters['layer_sizes']


#         for i in range(len(layer_sizes)-2):
#             modules.append(LipschitzLinear(1, projection, layer_sizes[i], layer_sizes[i+1]))
#             modules.append(self.init_activation(('fc', layer_sizes[i+1])))


#         modules.append(LipschitzLinear(1, projection, layer_sizes[-2], layer_sizes[-1]))

#         self.initialization(init_type=network_parameters['weight_initialization'])
#         self.num_params = self.get_num_params()

#         self.layers = nn.Sequential(*modules)
        

#     def forward(self, x):
#         """ """
#         return self.layers(x)