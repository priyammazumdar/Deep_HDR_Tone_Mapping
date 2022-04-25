import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import glob
import cv2
from PIL import Image

class HDRDataset(Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        self.ldr_images = glob.glob(os.path.join(filepath, "LDR_in/*.jpg"))
        print(self.ldr_images[0])
        self.hdr_images = glob.glob(os.path.join(filepath, "HDR_gt/*.hdr"))

    def __len__(self):
        return len(self.ldr_images)

    def __getitem__(self, idx):
        ldr = Image.open(self.ldr_images[idx])
        hdr = cv2.imread(self.hdr_images[idx], cv2.IMREAD_ANYDEPTH)
        ### PRECOMPUTED NORMALIZE ###
        normalize = transforms.Normalize(mean=[0.3805, 0.3663, 0.3367],
                                         std=[0.3665, 0.3626, 0.3590])
        ldr_transform = transforms.Compose([transforms.ToTensor(),
                                            normalize])
        ldr = ldr_transform(ldr)
        hdr = torch.tensor(hdr).permute((2,0,1))
        if torch.rand(1) > 0.5:
            ldr = TF.hflip(ldr)
            hdr = TF.hflip(hdr)
        if torch.rand(1) > 0.5:
            ldr = TF.vflip(ldr)
            hdr = TF.vflip(hdr)

        return ldr, hdr