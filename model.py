import torch.nn as nn
import torch.nn.functional as F
import pdb


class Deconv(nn.Module):
    def __init__(self):
        super(Deconv, self).__init__()
        self.pred5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.pred4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.pred3 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.pred2 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.pred1 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        pred5 = self.pred5(x[4])
        pred4 = self.pred4(F.upsample_bilinear(pred5, scale_factor=2) + x[3])
        pred3 = self.pred3(F.upsample_bilinear(pred4, scale_factor=2) + x[2])
        pred2 = self.pred2(F.upsample_bilinear(pred3, scale_factor=2) + x[1])
        pred1 = self.pred1(F.upsample_bilinear(pred2, scale_factor=2) + x[0])
        return [pred1, pred2, pred3, pred4, pred5]