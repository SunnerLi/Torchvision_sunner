import torchvision.transforms as transforms
import torchvision_sunner.transforms as suntransforms
import torchvision_sunner.data as sunData
import numpy as np
import torch
import cv2

"""
    This example show how to deal with two image folder and show it!
"""

if __name__ == '__main__':
    # Define dataset & loader
    dataset = sunData.ImageDataset(
        root_list = ['./waiting_for_you_dataset/real_world', './waiting_for_you_dataset/wait'],
        use_cv = False,
        sample_method = sunData.OVER_SAMPLING,
        transform = transforms.Compose([
            suntransforms.Rescale((160, 320), use_cv = False),
            suntransforms.ToTensor(),
            suntransforms.Transpose(suntransforms.BHWC2BCHW),
            suntransforms.Normalize([127., 127., 127.], [127., 127., 127.])
        ]) 
    )
    loader = sunData.ImageLoader(dataset, batch_size=32, shuffle=False, num_workers = 2)

    # Work
    for batch_img1, batch_img2 in loader:

        # Transfer tensor into numpy object
        # Result size: [batch_size, image_height, image_width, image_channel]
        batch_img1 = suntransforms.tensor2Numpy(batch_img1, transform = transforms.Compose([
            suntransforms.UnNormalize([127., 127., 127.], [127., 127., 127.]),
            suntransforms.Transpose(suntransforms.BCHW2BHWC), 
        ]))
        batch_img2 = suntransforms.tensor2Numpy(batch_img2, transform = transforms.Compose([
            suntransforms.UnNormalize([127., 127., 127.], [127., 127., 127.]),
            suntransforms.Transpose(suntransforms.BCHW2BHWC),
        ]))
        
        # Show
        batch_img = batch_img2[0].astype(np.uint8)
        batch_img = cv2.cvtColor(batch_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('show_window', batch_img)
        cv2.waitKey()

        # Done
        break