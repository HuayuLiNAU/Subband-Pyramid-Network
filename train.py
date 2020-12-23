import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import time
from glob import glob
from torch.autograd import Variable
import torch.optim.lr_scheduler
from torch.optim import lr_scheduler
import torch.nn.functional as F
from models import SPANET
from dataset import MyDataset
from utils import *
import os
import torch.nn as nn


def main():
    PATH = './check_points/SPANET1.pth'

    best = 0

    batch_size = 32
    num_steps = int(2.5 * 1e4)
    paths_clean = glob('./SIDD/gt/*')
    paths_noise = glob('./SIDD/noise/*')

    test_clean = glob('./validation/GT/*')
    test_noise = glob('./validation/Noisy/*')

    train_dataset = MyDataset(path_clean=paths_clean, path_noise=paths_noise, target_size=(64, 64), transform=None)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dataset = MyDataset(path_clean=test_clean, path_noise=test_noise, target_size=(256, 256), transform=None)
    test_dataset_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

    model = SPANET()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    summary(model, input_size=(3, 64, 64))

    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.0)

    scheduler = lr_scheduler.StepLR(optimizer, int(1*1e4), gamma=0.5)

    start_time = time.time()

    for step in range(num_steps):
        data_time = time.time()

        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        clean, noise = next(iter(train_dataset_loader))

        clean = Variable(clean.to(device))
        noise = Variable(noise.to(device))

        outputs = model(noise)
        loss = F.l1_loss(outputs, clean)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_train = torch.clamp(outputs, 0., 1.)
        psnr_train = batch_PSNR(out_train, clean, 1.)

        data_time = time.time() - data_time
        total_time = time.time() - start_time

        print("[step:%d/%d]   time:%.4f/%.4f    loss:%.4f / %.4f" % (
        step, num_steps, data_time, total_time, loss, psnr_train))

        if (step + 1) % 5000 == 0:
            # if (step+1) % 100 == 0:
            model.eval()
            psnr_test = 0
            for i, (clean_test, noise_test) in enumerate(test_dataset_loader):
                clean_test = Variable(clean_test.to(device))
                noise_test = Variable(noise_test.to(device))
                with torch.no_grad():
                    rec_test = model(noise_test)
                    psnr_temp = batch_PSNR(rec_test, clean_test, 1.)
                    print("[step:%d/%d]   psnr: %.4f" % (i, len(test_dataset_loader), psnr_temp))
                    psnr_test += psnr_temp
            psnr_test /= len(test_dataset_loader)
            if step > 22000:
                if psnr_test > best:
                    best = psnr_test
                    torch.save(model.state_dict(), './check_points/best/SPANET_{}.pth'.format(best))
            print("train steps at: :%d   psnr: %.4f" % (step, psnr_test))
            torch.save(model.state_dict(), PATH)
            print('check points saved')

        scheduler.step()

    torch.save(model.state_dict(), PATH)
    print('check points saved')


if __name__ == '__main__':
    main()




















