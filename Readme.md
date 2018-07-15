# torchvision_sunner

### The flexible extension of torchvision toward multiple image space 

[![Packagist](https://img.shields.io/badge/Version-18.5.14-yellow.svg)]()
[![Packagist](https://img.shields.io/badge/Pytorch-0.3.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Torchvision-0.2.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()
[![Packagist](https://img.shields.io/badge/OpenCV-3.1.0-brightgreen.svg)]()
[![Packagist](https://img.shields.io/badge/skImage-0.13.1-green.svg)]()

![](https://github.com/SunnerLi/Torchvision_sunner/blob/master/logo.png)

Update
---
The Torchvision_sunner package has moved to [gitlab](https://gitlab.com/SunnerLi/Torchvision_sunner) already! The github only supply the stable version. If you want to use the latest version, you should visit [gitlab](https://gitlab.com/SunnerLi/Torchvision_sunner) main page!      

Motivation
---
In pytorch, the common dataset can be load in an easy way. It also provides the `TensorDataset` to form the dataset. However, if we want to custom our unique image folder, or we want to load the muultiple image, the original methods cannot complete this work. In this package, you can load multiple images in an easy way!    

Install
---
1. download `torchvision_sunner` folder
2. put it in your current folder
3. import library and done!

Dataset
---
The examples will use waiting-for-you dataset as example. You can find the dataset [here](https://www.dropbox.com/s/cbuwbrehgglebhp/waiting_for_you_dataset.zip?dl=0). Download and extract it before you run the example. You can use your own data too! Just give the path of the folder, the module will load the image automatically!         


Usage
---
Import library first
```python
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
```
And start for your flexible data processing! The more detail can be refer to [wiki](https://github.com/SunnerLi/Torchvision_sunner/wiki).     

Tutorial
---
Since this project has already moved to gitlab, we provide the clear usage in the wiki page in both github and gitlab! Also, we will keep updating the containing in gitlab. You can visit the newest wiki [here](https://gitlab.com/SunnerLi/Torchvision_sunner/wikis/home). Unfortunately, we didn't have full time to write readthedocs. In the future, we will expect this plan after the stars of encouragement.    

Notice
---
* This package provides two backend image processing library working: opencv and skimage. Since the opencv can show the continuous image easily, the default library we use is opencv. On the contrary, the installation of opencv is tedious. You can choose skimage to become the backend library while it can be easily installed. 
* `tensor2Numpy` is function, and it just deals with single batch image. The detailed usage can be referred in example script.    
*  This project doesn't provides PyPI installation approach.    