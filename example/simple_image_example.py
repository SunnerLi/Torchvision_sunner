import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2

"""
    This example shows the very simple way to use this package
    The loader will load the image for 1 iteration

    Author  : SunnerLi
    Date    : 2018/09/13
"""

def main():
    # Create the fundemental data loader
    loader = sunnerData.DataLoader(sunnerData.ImageDataset(
        root = [
            ['/home/sunner/Music/waiting_for_you_dataset/wait'], 
            ['/home/sunner/Music/waiting_for_you_dataset/real_world']
        ],
        transform = transforms.Compose([
            sunnertransforms.ToTensor(),
            sunnertransforms.ToFloat(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Resize((160, 320)),
            sunnertransforms.Normalize(),
        ])), batch_size=32, shuffle=False, num_workers = 2
    )

    # Use upper wrapper to assign particular iteration
    loader = sunnerData.IterationLoader(loader, max_iter = 1)

    # Show!
    for batch_tensor, _ in loader:
        batch_img = sunnertransforms.asImg(batch_tensor, size = (160, 320))
        cv2.imshow('show_window', batch_img[0][:, :, ::-1])
        cv2.waitKey(0)

        # Or show multiple image in one line
        sunnertransforms.show(batch_tensor[:10], row=2, column=5)

if __name__ == '__main__':
    main()
