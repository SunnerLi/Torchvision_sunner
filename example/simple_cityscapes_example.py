import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
import os

"""
    This example shows how to deal with categorical images toward CitysSapes dataset
    You should download the dataset by your own
    In this example, the pallete file will generate automatically

    Author  : SunnerLi
    Date    : 2018/09/13
"""

# You should revise the path of image and label to the corresponding position
img_folder = [
    '/home/sunner/Dataset/CityScape/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png', 
    '/home/sunner/Dataset/CityScape/leftImg8bit/train/aachen/aachen_000001_000019_leftImg8bit.png'
]
tag_folder = [
    '/home/sunner/Dataset/CityScape/gtFine_trainvaltest/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png',
    '/home/sunner/Dataset/CityScape/gtFine_trainvaltest/gtFine/train/aachen/aachen_000001_000019_gtFine_color.png'
]

def main():
    # Define the loader to generate the pallete object
    loader = sunnerData.DataLoader(sunnerData.ImageDataset(
        root = [
            tag_folder
        ],
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
        ]), save_file = False                               # Don't save the record file, be careful!
        ), batch_size = 2, shuffle = False, num_workers = 2
    )
    pallete = sunnertransforms.getCategoricalMapping(loader, path = 'pallete.json')[0]
    del loader

    # Define the actual loader
    loader = sunnerData.DataLoader(sunnerData.ImageDataset(
        root = [
            img_folder,
            tag_folder
        ],
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
            sunnertransforms.ToFloat(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Resize((512, 1024)),
            sunnertransforms.Normalize(),
        ])), batch_size = 32, shuffle = False, num_workers = 2
    )

    # Define the reverse operator
    goto_op = sunnertransforms.CategoricalTranspose(pallete = pallete, direction = sunnertransforms.COLOR2ONEHOT)
    back_op = sunnertransforms.CategoricalTranspose(pallete = pallete, direction = sunnertransforms.ONEHOT2COLOR)

    # Show!
    for _, batch_index in loader:
        batch_img = back_op(goto_op(batch_index))
        batch_img = sunnertransforms.asImg(batch_img, size = (512, 1024))
        cv2.imshow('show_window', batch_img[0][:, :, ::-1])
        cv2.waitKey(0)
        break

if __name__ == '__main__':
    main()
