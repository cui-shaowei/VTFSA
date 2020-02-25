import os
import torch
from torch import nn

from libs.model._utils import load_state_dict_from_url

from libs.model.resnet import resnet18, model_urls


class TsinghuaModel(nn.Module):
    def __init__(self, pretrained=True, lstm_hidden_size=256, lstm_layers=2, lstm_dropout=0.1, num_classes=2):
        """

        :param pretrained:
        :param lstm_hidden_size:
        :param lstm_layers:
        :param num_classes:
        """
        super(TsinghuaModel, self).__init__()
        # self.backbone_1 = resnet18()
        self.backbone_2 = resnet18()

        if pretrained:
            if os.path.isfile(pretrained):
                state_dict = torch.load(pretrained)
            else:
                state_dict = load_state_dict_from_url(model_urls["resnet18"],
                                                      progress=True)
            # self.backbone_1.load_state_dict(state_dict)
            self.backbone_2.load_state_dict(state_dict)

        self.lstm = nn.LSTM(input_size=16, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            dropout=lstm_dropout,
                            batch_first=True,)
        self.fc1 = nn.Linear(in_features=512 + lstm_hidden_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, imgs, tactile):
        """

        :param imgs: A list with the order front_0, front_1, left_0, left_1
        :param tactile:
        :return:
        """
        # before_feature = self.backbone_1(imgs[0])
        current_feature = self.backbone_2(imgs[1])

        tactile = tactile.float()
        tactile_feautre, (h_n, h_c) = self.lstm(tactile, None)

        x = torch.cat((current_feature, tactile_feautre[:, -1, :]), dim=-1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x


