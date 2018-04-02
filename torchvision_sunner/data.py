import torchvision.transforms as transforms
import torch.utils.data as Data
import numpy as np
import torch
import glob
import cv2
import os

UNDER_SAMPLING = 0
OVER_SAMPLING = 1

# Constant
verbose = True

def quiet():
    global verbose
    verbose = False

class ImageDataset(Data.Dataset):
    def __init__(self, root_list, use_cv = True, sample_method = UNDER_SAMPLING, transform = None, test_ratio = 0.0):
        global verbose
        import glob
        import os
        self.root_list = root_list
        self.folder_list = []
        self.use_cv = use_cv
        self.sample_method = sample_method
        self.transform = transform
        self.test_ratio = test_ratio
        if not isinstance(root_list, (list, int)) and not isinstance(root_list, (tuple, int)):
            raise Exception('The type of 1st parameter should be tuple or list')
        channel_format_desc = "cv" if use_cv else "skimage"

        """
        1. [img1, img2]
        2. [folder1, folder2]
        3. [[img1, img2], [img1, img2]]
        4. [[img1, img2], folder1]
        """
        origin_img_list = []
        for root in root_list:
            if type(root) == str:
                if os.path.exists(root):
                    # ----------------------------------------------------------
                    # This function accept the user to key for two form:
                    # 1. The name of folder
                    # 2. The list of images
                    # ----------------------------------------------------------
                    if os.path.isdir(root):
                        img_list = glob.glob(os.path.join(root, '*'))
                        img_list = sorted(img_list)
                        if len(origin_img_list) > 0:
                            self.folder_list.append(origin_img_list)
                            if verbose:
                                print("[ ImageDataset ] path: %40s \t image number: %d \t channel format: %s" 
                                    % ('self-Defined', len(origin_img_list), channel_format_desc)
                                )
                            origin_img_list = []
                        self.folder_list.append(img_list)
                        if verbose:
                            print("[ ImageDataset ] path: %40s \t image number: %d \t channel format: %s" 
                                % (root, len(img_list), channel_format_desc)
                            )
                    else:
                        origin_img_list.append(root)                        
                else:
                    raise Exception("root folder or image not found...")

            # Check the image is exist toward given image list
            elif type(root) == list:
                for name in root:
                    if not os.path.exists(name):
                        raise Exception("Image %s not found..." % (name))
                if verbose:
                    print("[ ImageDataset ] path: %40s \t image number: %d \t channel format: %s" 
                        % ('self-Defined', len(root), channel_format_desc)
                    )
                self.folder_list.append(root)

        # If there is only one image list, append it!
        if len(origin_img_list) > 0:
            self.folder_list.append(origin_img_list)
            if verbose:
                print("[ ImageDataset ] path: %40s \t image number: %d \t channel format: %s" 
                    % ('self-Defined', len(origin_img_list), channel_format_desc)
                )
            origin_img_list = []

        self.setImgNumber()
        if self.sample_method == OVER_SAMPLING:
            self.fill()
            self.img_num = max(self.img_num_list)
        else:
            self.img_num = min(self.img_num_list)

    def setImgNumber(self):
        self.img_num_list = []
        for i in range(len(self.folder_list)):
            self.img_num_list.append(len(self.folder_list[i]))

    def fill(self):
        for i in range(len(self.folder_list)):
            if max(self.img_num_list) > len(self.folder_list[i]):
                random_idx = np.random.randint(low=0, 
                    high=len(self.folder_list[i]), 
                    size=max(self.img_num_list) - len(self.folder_list[i]))
                for j in range(len(random_idx)):
                    self.folder_list[i].append(self.folder_list[i][random_idx[j]])
        
    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        return_list = []
        if self.use_cv:
            import cv2
            for i in range(len(self.folder_list)):
                img = cv2.imread(self.folder_list[i][idx])
                if self.transform:
                    img = self.transform(img)
                return_list.append(img)
            return return_list
        else:
            from skimage import io
            for i in range(len(self.folder_list)):
                img = io.imread(self.folder_list[i][idx])
                if self.transform:
                    img = self.transform(img)
                return_list.append(img)
            return return_list

    def quiet(self):
        global verbose
        verbose = False

class ImageLoader(Data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers = 1):
        super(ImageLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers)
        self.dataset = dataset
        self.iter_num = self.getIterNumber()

    def getIterNumber(self):       
        return round(self.dataset.img_num / self.batch_size) + 1
        # return round(len(self.dataset.data_tensor) / self.batch_size)

    def getImageNumber(self):
        return self.dataset.img_num