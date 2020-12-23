import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import numpy as np

class MyDataset(Dataset):
    def __init__(self, path_clean, path_noise, target_size = (80, 80), transform=None):
        self.path_clean = path_clean
        self.path_noise = path_noise
        self.angle_array = [90, -90, 180, -180, 270, -270]
        self.target_size = target_size
        self.transform = transform
        self.pil2tensor = transforms.ToTensor()
        
    def __getitem__(self, index):
        clean = Image.open(self.path_clean[index])
        noise = Image.open(self.path_noise[index])
        
        i, j, h, w = transforms.RandomCrop.get_params(clean, output_size=self.target_size)
        
        clean = TF.crop(clean, i, j, h, w)
        noise = TF.crop(noise, i, j, h, w)
        
        if random.random() > 0.5:
            clean = TF.hflip(clean)
            noise = TF.hflip(noise)
        
        if random.random() > 0.5:
            angle = np.random.choice(self.angle_array, 1)
            clean = TF.rotate(clean, angle)
            noise = TF.rotate(noise, angle)
        
        clean = np.array(clean)/255.
        clean = torch.Tensor(clean)
        noise = np.array(noise)/255.
        noise = torch.Tensor(noise)
        
        
        return clean.permute(2,0,1), noise.permute(2,0,1)
    
    def __len__(self):
        return len(self.path_clean)