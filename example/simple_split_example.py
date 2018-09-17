import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2

"""
    This example shows how to do the train-test-split
    And show the 1st test image

    Author  : SunnerLi
    Date    : 2018/09/15
"""

def main():
    # Split the dataset
    train_dataset = sunnerData.ImageDataset(
        root = [
            ['/home/sunner/Music/waiting_for_you_dataset/wait'], 
            ['/home/sunner/Music/waiting_for_you_dataset/real_world']
        ], transform = None, split_ratio = 0.1, save_file = True
    )
    del train_dataset
    test_dataset = sunnerData.ImageDataset(
        file_name = '.split.pkl',
        transform = transforms.Compose([
            sunnertransforms.Resize((160, 320)),
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize(),
        ])
    )

    # Create the data loader
    loader = sunnerData.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 2)

    # Use upper wrapper to assign particular iteration
    loader = sunnerData.IterationLoader(loader, max_iter = 1)

    # Show!
    for batch_img, _ in loader:
        batch_img = sunnertransforms.asImg(batch_img, size = (160, 320))
        cv2.imshow('show_window', batch_img[0][:, :, ::-1])
        cv2.waitKey(0)

if __name__ == '__main__':
    main()
