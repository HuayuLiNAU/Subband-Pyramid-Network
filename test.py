import torch
import torchvision.transforms as transforms
import time
import torch.optim.lr_scheduler
import os
from PIL import Image
import numpy as np
from skimage import io
from models import SPANET
from dataset import MyDataset
from utils import *
import torch.nn as nn

PATH = './check_points/SPANET.pth'
out_path = './check_points/predict/'
model = SPANET()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = nn.DataParallel(model, [0])
model.load_state_dict(torch.load(PATH, map_location='cuda:0'))
model.eval()
model = model.to(device)

root = './validation/Noisy'
images = os.listdir(root)

pil2tensor = transforms.Compose([transforms.ToTensor()])

tensor2pil = transforms.ToPILImage()

with torch.no_grad():
    for image in images:
        img = Image.open('%s/%s'%(root,image))
        im_array = np.array(img)
        im_input = pil2tensor(img)
        im_input = im_input.view(-1,3,im_array.shape[0],im_array.shape[1])
        
        im_input = im_input.to(device)
        start_time = time.time()
        clear = torch.clamp(model(im_input), 0., 1.)
        
        clear = clear.cpu()
        im_h = clear.data[0].numpy().astype(np.float32)
        im_h = np.clip(im_h, 0., 1.)
        im_h = im_h.transpose(1,2,0)
        io.imsave(out_path+image,im_h)
        
        elapsed_time = time.time() - start_time
        print(elapsed_time)

clean = './validation/GT/'
PSNR, SSIM = psnr(clean, out_path)
print("The average PSNR is ----> ", round(PSNR,4), " SSIM is ----> ", round(SSIM,4))