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
    dataset = sunnerData.ImageDataset(
        root_list = ['./waiting_for_you_dataset/wait/a_0.jpg', './waiting_for_you_dataset/wait/a_1.jpg'],        
        transform = transforms.Compose([
            sunnertransforms.Rescale((160, 320)),
            sunnertransforms.ToTensor(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize([127., 127., 127.], [127., 127., 127.])
        ]) 
    )
    loader = sunnerData.ImageLoader(dataset, batch_size=32, shuffle=False, num_workers = 2)
    loader_iter = iter(loader)

    # Work
    batch_num = loader.getIterNumber()  # !    
    for i in range(batch_num):
        # Set the input tensor
        # Result size: [batch_size, image_channel, image_height, image_width]
        # ex. [32, 3, 160, 320]
        data_tuple = loader_iter.next()
        batch_img1 = torch.stack(data_tuple[0], 0)
        batch_img1 = sunnertransforms.tensor2Numpy(batch_img1, transform = transforms.Compose([
            sunnertransforms.UnNormalize([127., 127., 127.], [127., 127., 127.]),
            sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
        ]))

        # Show
        batch_img = batch_img1[0].astype(np.uint8)
        cv2.imshow('show_window', batch_img)
        cv2.waitKey()