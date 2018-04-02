import torchvision.transforms as transforms
import torch.utils.data as Data
import numpy as np
import random
import torch
import math
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
    def __init__(self, root_list, use_cv = True, sample_method = UNDER_SAMPLING, transform = None, split_ratio = 0.0):
        global verbose
        import glob
        import os
        self.root_list = root_list
        self.folder_list = []
        self.use_cv = use_cv
        self.sample_method = sample_method
        self.transform = transform
        self.split_ratio = split_ratio
        if not isinstance(root_list, (list, int)) and not isinstance(root_list, (tuple, int)):
            raise Exception('The type of 1st parameter should be tuple or list')
        channel_format_desc = "cv" if use_cv else "skimage"

        # -----------------------------------------------------------------------------------------------
        #   This function allow five types of root_list object:
        #   1. The list of images,                            e.g: [img1, img2]
        #   2. The list of folders,                           e.g: [folder1, folder2]
        #   3. The list of list which contain images,         e.g: [[img1, img2], [img1, img2]]
        #   4. The combination of folder and list of image,   e.g: [[img1, img2], folder1]
        #
        #
        #   This part with deal with various situation, and form the new_root_list and folder_list
        # 
        #   * new_root_list   : The list object which record the root of each dataset
        #   * self.folder_list: The list object which contain the list of images
        #
        # -----------------------------------------------------------------------------------------------
        origin_img_list = []
        new_root_list = []
        for root in root_list:
            if type(root) == str:
                if os.path.exists(root):
                    # -----------------------------------------------------------------------------------
                    # This function accept the user to key for two form:
                    # 1. The name of folder
                    # 2. The list of images
                    # -----------------------------------------------------------------------------------
                    if os.path.isdir(root):
                        img_list = glob.glob(os.path.join(root, '*'))
                        img_list = sorted(img_list)
                        if len(origin_img_list) > 0:
                            self.folder_list.append(origin_img_list)
                            new_root_list.append('self')
                            origin_img_list = []
                        self.folder_list.append(img_list)
                        new_root_list.append(root)                     
                    else:
                        origin_img_list.append(root)
                else:
                    raise Exception("root folder or image not found...")

            # Check the image is exist toward given image list
            elif type(root) == list:
                for name in root:
                    if not os.path.exists(name):
                        raise Exception("Image %s not found..." % (name))
                self.folder_list.append(root)
                new_root_list.append('self')

        # If there is only one image list, append it!
        if len(origin_img_list) > 0:
            self.folder_list.append(origin_img_list)
            new_root_list.append('self')
            origin_img_list = []

        # ----------------------------------------------------------------------
        # Split as train and test
        """
            [[a_1, a_2, a_3], [b_1, b_2]]
        """
        # ----------------------------------------------------------------------
        def generateIndexList(a, size):
            result = set()
            while len(result) != size:
                result.add(random.randint(0, len(a)))
            return list(result)

        self.setImgNumber()
        if self.split_ratio:
            train_folder_list = list(self.folder_list)
            self.test_folder_list = []
            if len(set(self.img_num_list)) == 1:
                # Determine the choice index list
                test_img_num = math.floor(self.img_num_list[0] * self.split_ratio)
                choice_index_list = generateIndexList(range(self.img_num_list[0]), size = test_img_num)

                # Generate the test list and remove from train list
                for i, img_list in enumerate(self.folder_list):
                    pick_list = []
                    for idx in choice_index_list:
                        pick_list.append(img_list[idx])
                    self.test_folder_list.append(pick_list)
                for i in range(len(self.test_folder_list)):
                    for test_img_name in self.test_folder_list[i]:
                        train_folder_list[i].remove(test_img_name)
            else:
                for i, img_list in enumerate(self.folder_list):
                    pick_list = []
                    test_img_num = math.floor(len(img_list) * self.split_ratio)
                    choice_index_list = generateIndexList(range(len(img_list)), size = test_img_num)
                    for idx in choice_index_list:
                        pick_list.append(img_list[idx])
                    self.test_folder_list.append(pick_list)
                for i in range(len(self.test_folder_list)):
                    for test_img_name in self.test_folder_list[i]:
                        train_folder_list[i].remove(test_img_name)
            self.folder_list = train_folder_list
        
        # Print the Information
        for i, root in enumerate(new_root_list):
            split_or_not = 'Split' if self.split_ratio != 0.0 else 'Remain'
            if type(root) == str and verbose:
                print("[ ImageDataset ] image number: %d \t split type: %6s \t channel format: %s \t path: %s" 
                    % (len(self.folder_list[i]), split_or_not, channel_format_desc, root)
                )
            elif type(root) == list and verbose:
                print("[ ImageDataset ] image number: %d \t split type: %6s \t channel format: %s \t path: %s" 
                    % (len(self.folder_list[i]), split_or_not, channel_format_desc, 'Self-Define')
                )

        # Adjust the image number
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

    def getSplit(self):
        return self.test_folder_list

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

    def getImageNumber(self):
        return self.dataset.img_num