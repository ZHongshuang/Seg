import gc
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.autograd import Variable
import torchvision
from dataset import MyTestData
from model import Deconv
from vgg import Vgg16
from tensorboardX import SummaryWriter
import numpy as np
from datetime import datetime
import os
import glob
import pdb
import argparse
from PIL import Image
from os.path import expanduser
home = expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='%s/data/datasets/saliency_Dataset/ECSSD'%home)  # training dataset
parser.add_argument('--output_dir', default='%s/data/datasets/saliency_Dataset/ECSSD/Web'%home)  # training dataset
parser.add_argument('--para_dir', default='parameters')  # training dataset
parser.add_argument('--b', type=int, default=1)  # batch size
opt = parser.parse_args()
print(opt)


def main():
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)
    bsize = opt.b
    # models
    feature = Vgg16(pretrained=True)
    feature.cuda()
    feature.eval()
    feature.load_state_dict(torch.load('%s/feature.pth'%opt.para_dir))

    deconv = Deconv()
    deconv.cuda()
    deconv.eval()
    deconv.load_state_dict(torch.load('%s/deconv.pth'%opt.para_dir))
    loader = torch.utils.data.DataLoader(
        MyTestData(opt.input_dir),
        batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)
    for ib, (data, img_name, img_size) in enumerate(loader):
        print ib
        inputs = Variable(data).cuda()
        feats = feature(inputs)
        outputs = deconv(feats)
        outputs = F.sigmoid(outputs)
        outputs = outputs.data.cpu().squeeze(1).numpy()
        for msk in outputs:
            msk = (msk * 255).astype(np.uint8)
            msk = Image.fromarray(msk)
            msk = msk.resize((img_size[0][0], img_size[1][0]))
            msk.save('%s/%s.png' % (opt.output_dir, img_name[0]), 'PNG')


if __name__ == "__main__":
    main()

