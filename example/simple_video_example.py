import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2

"""
    This example shows how to use VideoDataset to deal with multiple videos

    Author  : SunnerLi
    Date    : 2018/09/09
"""

def main():
    # Define op first
    transform_op = transforms.Compose([
        sunnertransforms.ToTensor(),
        sunnertransforms.ToFloat(),
        sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        sunnertransforms.Normalize(),
    ])

    # Define loader
    loader = sunnerData.DataLoader(
        sunnerData.VideoDataset(
            root = [
                ['/home/sunner/Dataset/flower/A'], 
                ['/home/sunner/Dataset/flower/B']
            ], transform = transform_op, T = 20
        ), batch_size=2, shuffle=False, num_workers = 2
    )

    # Looping
    for _, seq in loader:
        seq = [_.squeeze(1) for _ in torch.chunk(seq, seq.size(1), dim = 1)]    # BTCHW -> T * BCHW
        for i in range(10):                                                     # repeat for 10 times
            for img in seq:
                batch_img = sunnertransforms.asImg(img, size = (320, 640))
                cv2.imshow('show_window', batch_img[0][:, :, ::-1])
                cv2.waitKey(10)
        break

if __name__ == '__main__':
    main()
