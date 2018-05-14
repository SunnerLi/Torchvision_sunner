import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2

"""
    This tutorial shows how to define the dataset object with flexibility
"""

if __name__ == '__main__':
    # -------------------------------------------------------------------------------------------
    # 1. A single list which contains multiple images
    # -------------------------------------------------------------------------------------------
    dataset = sunnerData.ImageDataset(
        root_list = ['./waiting_for_you_dataset/real_world/a_1.jpg', './waiting_for_you_dataset/wait/a_1.jpg'],
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        ]) 
    )
    loader = sunnerData.ImageLoader(dataset, batch_size=32, shuffle=False, num_workers = 2)
    loader_iter = iter(loader)
    image_tensor = loader_iter.next()
    batch_img = image_tensor[0]
    batch_img = sunnertransforms.tensor2Numpy(batch_img, transform = transforms.Compose([
        sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
    ]))
    cv2.imshow('show', batch_img[0])
    cv2.waitKey()


    # -------------------------------------------------------------------------------------------
    # 2. A single list which contains multiple folders
    # -------------------------------------------------------------------------------------------
    print('\n'*5)    
    dataset = sunnerData.ImageDataset(
        root_list = ['./waiting_for_you_dataset/real_world', './waiting_for_you_dataset/wait'],
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        ]) 
    )
    loader = sunnerData.ImageLoader(dataset, batch_size=32, shuffle=False, num_workers = 2)
    loader_iter = iter(loader)
    image_tensor = loader_iter.next()
    batch_img = image_tensor[0]
    batch_img = sunnertransforms.tensor2Numpy(batch_img, transform = transforms.Compose([
        sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
    ]))
    cv2.imshow('show', batch_img[0])
    cv2.waitKey()


    # -------------------------------------------------------------------------------------------
    # 3. A list of list which contains multiple list, and the sub-lists contain multiple images
    # -------------------------------------------------------------------------------------------
    print('\n'*5)    
    dataset = sunnerData.ImageDataset(
        root_list = [['./waiting_for_you_dataset/real_world/a_1.jpg', './waiting_for_you_dataset/wait/a_1.jpg'],
            ['./waiting_for_you_dataset/real_world/a_3.jpg', './waiting_for_you_dataset/wait/b_1.jpg']],
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        ]) 
    )
    loader = sunnerData.ImageLoader(dataset, batch_size=32, shuffle=False, num_workers = 2)
    loader_iter = iter(loader)
    image_tensor = loader_iter.next()
    batch_img = image_tensor[0]
    batch_img = sunnertransforms.tensor2Numpy(batch_img, transform = transforms.Compose([
        sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
    ]))
    cv2.imshow('show', batch_img[-1])
    cv2.waitKey()


    # -------------------------------------------------------------------------------------------
    # 4. A list of list which contains multiple list, and the sub-lists contain multiple images
    # -------------------------------------------------------------------------------------------
    print('\n'*5)    
    dataset = sunnerData.ImageDataset(
        root_list = [['./waiting_for_you_dataset/real_world/a_1.jpg', './waiting_for_you_dataset/wait/a_1.jpg'],
            './waiting_for_you_dataset/wait', './waiting_for_you_dataset/real_world'],
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        ]) 
    )
    loader = sunnerData.ImageLoader(dataset, batch_size=32, shuffle=True, num_workers = 2)
    loader_iter = iter(loader)
    image_tensor = loader_iter.next()
    batch_img = image_tensor[1]
    batch_img = sunnertransforms.tensor2Numpy(batch_img, transform = transforms.Compose([
        sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
    ]))
    cv2.imshow('show', batch_img[-1])
    cv2.waitKey()