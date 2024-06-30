import numpy as np
import torch 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from architectures.base_model import BaseModel
from layers.lipschitzlinear import LipschitzLinear
from projections.fc_projections import identity, bjorck_orthonormalize_fc, layerwise_orthogonal_fc


class SimpleFCClassification(BaseModel):
    """simple architecture for a fully-connected network"""
    def __init__(self, network_parameters, **params):
        
        super().__init__(**params)

        modules = nn.ModuleList()

        if network_parameters['projection'] == 'no_projection':
            print("no orthonormalisation. projection= identity")
            projection = identity
        elif network_parameters['projection'] == 'orthonormalize':
            print("Orthonormalisation will take place.")
            if 'bjorck_iter' in network_parameters:
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
                projection = bjorck_orthonormalize_fc
        else:
            raise ValueError('Projection type is not valid')

        layer_sizes = network_parameters['layer_sizes']


        for i in range(len(layer_sizes)-2):
            modules.append(LipschitzLinear(1, projection, layer_sizes[i], layer_sizes[i+1]))# 1 corresponds to lipschitz constant 1
            modules.append(self.init_activation(('fc', layer_sizes[i+1])))


        modules.append(LipschitzLinear(1, projection, layer_sizes[-2], layer_sizes[-1]))
        modules.append(nn.Sigmoid())
        self.initialization(init_type=network_parameters['weight_initialization'])
        self.num_params = self.get_num_params()

        self.layers = nn.Sequential(*modules)
        

    def forward(self, x):
        """ """
        ### apply layers and sigmoid function
        x= self.layers(x)
        #x = torch.sigmoid(x)
        return x