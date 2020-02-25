import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
# from models.layers import ConvOffset2D
import torch.nn.functional as F
import torchvision.models as models
import numpy
# from modules.resnet import *
from torch.nn import Parameter
from modules.Self_Att import *
from modules.transformer import TransformerEncoder
from modules.multihead_attention import MultiheadAttention
from modules.resnet_7_7_512 import resnet18
# from modules.resnet import resnet18

class VTF_CNNs_pre_freeze(nn.Module):
    def __init__(self, hyp_params):

        super(VTF_CNNs_pre_freeze, self).__init__()
        self.cnn_rgb=resnet18(pretrained=True)
        self.cnn_gelsight=resnet18(pretrained=True)
        # self.self_attn=Self_Attn(1024,activation="softmax")
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1=nn.Linear(1024, 64)
        # self.fc2 = nn.Linear(64, 2)
        # self.fc3 = nn.Linear(128, 3)

    def forward(self, rgb,gel):

        rgb=self.cnn_rgb(rgb)
        gel=self.cnn_gelsight(gel)
        # rgb=F.softmax(rgb,dim=1)
        # gel=F.softmax(gel,dim=1)
        # print(rgb.shape, gel.shape)
        x_fusion = torch.zeros(rgb.size(0), rgb.size(1)+gel.size(1),
                                     rgb.size(2)*gel.size(2),rgb.size(3)*rgb.size(3)).cuda()
        # print(x_fusion.shape)
        for i in range(x_fusion.size(2)):
            for j in range(x_fusion.size(3)):
                x_fusion[:, :,i, j] = torch.cat((rgb[:,:, i // rgb.size(2),
                                                           j // rgb.size(3)],
                                                           gel[:, :,i % gel.size(2),
                                                           j % gel.size(3)]), dim=1)
        return x_fusion


class VTF_SA(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a VTF_SA model.
        """
        super(VTF_SA, self).__init__()
        # self.cnn_rgb=resnet18(pretrained=True)
        # self.cnn_gelsight=resnet18(pretrained=True)
        self.self_attn=Self_Attn(1024,activation="softmax")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1=nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 2)
        # self.fc3 = nn.Linear(128, 3)

    def forward(self, x_fusion):

        out, attention = self.self_attn(x_fusion)
        x = self.avgpool(out)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # x=torch.cat((x_rgb1,x_gel0_diff,x_gel1_diff),dim=-1)#,x_rgb1
        output=self.fc2(self.fc1(x))
        return output

