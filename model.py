from resnet18 import resnet18
from resnet50 import resnet50, resnet50H, resnet50HH
import torch.nn as nn
import torch
import numpy as np


class ClsNet(nn.Module):
    def __init__(self, opt):
        super(ClsNet, self).__init__()
        self.opt = opt
        pretrained_dict = torch.load('resnet50-19c8e357.pth')#'resnet50-19c8e357.pth'
        self.net3 = resnet50(pretrained=False)
        model_dict3 = self.net3.state_dict()
        pretrained_dict3 = {k: v for k, v in pretrained_dict.items() if k in model_dict3}
        pretrained_dict3.pop('classifier.weight', None)
        pretrained_dict3.pop('classifier.bias', None)
        model_dict3.update(pretrained_dict3)
        self.net3.load_state_dict(model_dict3)

        self.criterionCls = torch.nn.CrossEntropyLoss()

    def forward(self, img1):
        if isinstance(img1, np.ndarray): #Lagt til for å endre hvis det er np.array i stedet for pytorch tensor object
            img1 = torch.from_numpy(img1).float()
            img1 = img1.permute(0, 3, 1, 2)
        out = self.net3(img1)
        return out

    def calculate_loss(self, img1, Y):
        all_out = self.forward(img1)
        all_img_loss = self.criterionCls(all_out, Y)
        return all_img_loss

    def calculate_acc(self, img1):
        all_out = self.forward(img1)
        allimg_pre = torch.softmax(all_out, dim=1)
        allimg_pre = allimg_pre.cpu().data.numpy()
        allimg_prob = allimg_pre[0, 1]
        allimg_pre = np.argmax(allimg_pre, axis=1)
        return allimg_pre, allimg_prob