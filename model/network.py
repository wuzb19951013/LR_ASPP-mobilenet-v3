"""
    end to end network

Author: Zhengwei Li
Date  : 2018/12/24
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.T_Net import T_mv2_unet


T_net = T_mv2_unet


class net(nn.Module):
    '''
		end to end net 
    '''

    def __init__(self):

        super(net, self).__init__()

        self.t_net = T_net()



    def forward(self, input):

    	# trimap
        trimap = self.t_net(input)
        return trimap







