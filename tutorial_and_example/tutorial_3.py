from torch.autograd import Variable
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2

"""
    This tutorail shows how to split the testing data and remain to the disk
    You can also read the record from disk and use in another process
"""

if __name__ == '__main__':
    # -------------------------------------------------------------------------------------------
    # 1. First use the dataset object
    # -------------------------------------------------------------------------------------------
    # arrange the train and test dataset
    train_dataset = sunnerData.ImageDataset(
        root_list = ['./waiting_for_you_dataset/real_world', './waiting_for_you_dataset/wait'],
        split_ratio = 0.1,
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        ])
    )
    test_dataset = sunnerData.ImageDataset(
        root_list = train_dataset.getSplit(),
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        ])
    )
    loader = sunnerData.ImageLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 2)
    for real_tensor, wait_tensor in loader:
        batch_img = sunnertransforms.tensor2Numpy(wait_tensor, transform = transforms.Compose([
            sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
        ]))
        cv2.imshow('show', batch_img[1])
        cv2.waitKey()
        break
        
    # save to disk
    train_dataset.save(remain_save_path = 'waiting_remain.pkl', split_save_path = 'waiting_split.pkl')

    # -------------------------------------------------------------------------------------------
    # 2. Second use the dataset object
    # -------------------------------------------------------------------------------------------
    # Load the record .pkl file
    print('\n'*3)
    train_dataset = sunnerData.ImageDataset(
        root_list = 'waiting_remain.pkl',
        split_ratio = 0.1,
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        ])
    )
    test_dataset = sunnerData.ImageDataset(
        root_list = 'waiting_split.pkl',
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        ]) 
    )
    loader = sunnerData.ImageLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 2)
    for real_tensor, wait_tensor in loader:
        batch_img = sunnertransforms.tensor2Numpy(wait_tensor, transform = transforms.Compose([
            sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
        ]))
        cv2.imshow('show', batch_img[1])
        cv2.waitKey()
        break