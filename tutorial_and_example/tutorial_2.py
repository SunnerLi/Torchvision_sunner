from torch.autograd import Variable
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2

"""
    This tutorial shows how to use dataloader to load the batch image
"""

def main():
    dataset = sunnerData.ImageDataset(
        root_list = ['./waiting_for_you_dataset/real_world', './waiting_for_you_dataset/wait'],
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        ]) 
    )

    # --------------------------------------------------------------
    # 1. Cast the container as iterator
    # --------------------------------------------------------------
    loader = sunnerData.ImageLoader(dataset, batch_size=32, shuffle=False, num_workers = 2)
    loader_iter = iter(loader)

    # load the next batch
    image_tensor = loader_iter.next()
    batch_img = image_tensor[0]

    # post-process
    batch_img = sunnertransforms.tensor2Numpy(batch_img, transform = transforms.Compose([
        sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
    ]))
    cv2.imshow('show', batch_img[1])
    cv2.waitKey()
    
    # --------------------------------------------------------------
    # 2. Use container
    # --------------------------------------------------------------
    loader = sunnerData.ImageLoader(dataset, batch_size=32, shuffle=False, num_workers = 2)
    for real_tensor, wait_tensor in loader:
        batch_img = sunnertransforms.tensor2Numpy(wait_tensor, transform = transforms.Compose([
            sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
        ]))
        cv2.imshow('show', batch_img[1])
        cv2.waitKey()
        break

if __name__ == '__main__':
    main()    