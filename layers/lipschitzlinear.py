import torch
from torch.nn import Linear
import torch.nn.functional as F

class LipschitzLinear(Linear):
    def __init__(self, lipschitz: float, projection, in_features: int, 
                    out_features: int, bias: bool = True):
        '''
        1. lipschitz= I believe it is lipschitz constant.
        2. projection= A function to constraint the lipschitz constant of the weight matrix
        '''
        
        super().__init__(in_features, out_features, bias)
        
        self.lipschitz = lipschitz
        self.projection = projection
        
    def forward(self, x):
        lipschitz_weight = self.projection(self.weight, self.lipschitz)
        # print("debugging, lipschitz_weight's dtype is:", lipschitz_weight.dtype)
        # print("debugging, self.bias's dtype is:", self.bias.dtype)
        # print("debigging, x.dtype:",x.dtype)
        return F.linear(x.float(), lipschitz_weight, self.bias)