import glob
import logging
from pathlib import Path
from typing import List

import imageio
import torch
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def execute(image_paths: List[Path]) -> torch.tensor:
    net = resnet50(num_classes=120)
    ckpt_file_path = './resnet_bes_150t.pt'
    net.load_state_dict(torch.load(ckpt_file_path, map_location=torch.device('cpu')))
    with torch.no_grad():

        images = [np.array(imageio.v2.imread(image_path)) for image_path in image_paths]

        transforms_test = transforms.Compose(
            [
                transforms.ToPILImage(),
                #transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        images = transforms_test(images)
        images = [np.array(images(i))/ 255 for i in images]
        images = np.stack(images, axis=0)
        images = torch.tensor(np.transpose(images, (0, 3, 1, 2))).to('cpu').float()
        net(images)
        return net(images)


if __name__ == '__main__':
    files_list = glob.glob("D:/datasets/Similarity test/02e81d8623/*png")
    execute(files_list)
