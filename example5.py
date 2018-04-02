from torch.autograd import Variable
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2

if __name__ == '__main__':
    # Define dataset & loader
    # sunnerData.quiet()
    # sunnertransforms.quiet()
    train_dataset = sunnerData.ImageDataset(
        root_list = ['./waiting_for_you_dataset/real_world', './waiting_for_you_dataset/wait'],
        split_ratio = 0.1,
        transform = transforms.Compose([
            sunnertransforms.Rescale((160, 320)),
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize([127., 127., 127.], [127., 127., 127.])
        ]) 
    )
    test_dataset = sunnerData.ImageDataset(
        root_list = train_dataset.getSplit(),
        transform = transforms.Compose([
            sunnertransforms.Rescale((160, 320)),
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize([127., 127., 127.], [127., 127., 127.])
        ]) 
    )
    loader = sunnerData.ImageLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 2)

    # Work
    batch_num = loader.getIterNumber()  # !    
    for real_img, wait_img in loader:
        batch_img = Variable(wait_img)
        batch_img = sunnertransforms.tensor2Numpy(batch_img, transform = transforms.Compose([
            sunnertransforms.UnNormalize([127., 127., 127.], [127., 127., 127.]),
            sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
        ]))

        # Show
        batch_img = batch_img[0].astype(np.uint8)
        # cv2.namedWindow('show_window', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('show_window', batch_img)
        # cv2.waitKey(1000)
        break