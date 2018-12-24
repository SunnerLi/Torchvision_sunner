import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
import os

"""
    This example shows how to deal with categorical images with self-defined pallete
    You should define your own pallete file by yourself before running this code
    Moreover, we use multi-loader to customize the each image domain

    Author  : SunnerLi
    Date    : 2018/09/13
"""

def main():
    # Get the pallete object
    pallete = sunnertransforms.getCategoricalMapping(path = "ear-pen-pallete.json")[0]

    # Create the dataset
    img_dataset = sunnerData.ImageDataset(
        root = [
            ['/home/sunner/Music/Ear-Pen-master/generate/train/img'], 
        ],
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
            sunnertransforms.ToFloat(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Resize((260, 195)),
            sunnertransforms.Normalize(),
        ]), save_file = False
    )
    tag_dataset = sunnerData.ImageDataset(
        root = [
            ['/home/sunner/Music/Ear-Pen-master/generate/train/tag']
        ],
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
            sunnertransforms.ToFloat(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Resize((260, 195)),
            sunnertransforms.Normalize(),
            sunnertransforms.CategoricalTranspose(pallete = pallete, direction = sunnertransforms.COLOR2INDEX, index_default = 0)
        ])
    )

    # Create the loader
    loader = sunnerData.MultiLoader([img_dataset, tag_dataset], batch_size = 1, shuffle = False, num_workers = 2)

    # Define the reverse operator
    back_op = sunnertransforms.CategoricalTranspose(pallete = pallete, direction = sunnertransforms.INDEX2COLOR, index_default = 0)

    # Show!
    for (_, batch_tag) in loader:
        batch_tag = back_op(batch_tag)
        batch_tag = sunnertransforms.asImg(batch_tag, size = (260, 195))
        cv2.imshow('show_window', batch_tag[0][:, :, ::-1])
        cv2.waitKey(0)
        break

if __name__ == '__main__':
    main()
