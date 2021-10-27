import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GradCam(nn.Module):
    def __init__(self, network, target_layer):
        super(GradCam, self).__init__()
        self.gradients = None

        self.network = network
        self.features_bert = self.network.bert
        self.features_conv1 = self.network.conv1
        # self.features_pool = nn.MaxPool1d(2)

        self.target_conv = self.network.cnn[:target_layer]
        self.remain_conv = self.network.cnn[target_layer:]
        self.classifier = self.network.classifier

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, masks):
        x = self.features_bert(x, masks)[0]
        x = x.unsqueeze(1)
        x = self.features_conv1(x)
        x = x.squeeze(3)
        x = self.target_conv(x)
        '''
            이 layer와 최종 노드 gradient를 계산
            gradient가 역전파 과정에서 계산되고 버려지는데 hook을 이용해 잡아둠.

        '''
        h = x.register_hook(self.activations_hook)
        # x = self.features_pool(x)
        x = self.remain_conv(x)

        x = x.view(-1, 128 * 31)
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x, masks):
        x = self.features_bert(x, masks)[0]
        x = x.unsqueeze(1)
        x = self.features_conv1(x)
        x = x.squeeze(3)
        x = self.target_conv(x)
        return x