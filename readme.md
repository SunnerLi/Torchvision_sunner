# Torchvision_sunner

[![Packagist](https://img.shields.io/badge/Version-19.4.15-yellow.svg)]()
[![Packagist](https://img.shields.io/badge/Pytorch-0.4.1-red.svg)]()
[![Packagist](https://img.shields.io/badge/Torchvision-0.2.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()
[![Packagist](https://img.shields.io/badge/skImage-0.13.1-green.svg)]()

![](https://i.imgur.com/2NGgNQc.png)

Motivation
---
In pytorch, the common dataset can be load in an easy way. It also provides the ``TensorDataset`` to form the dataset. However, if we want to custom our unique image folder, or we want to load the muultiple image, the original methods cannot complete this work. In this package, you can **load multiple images or video** in an easy way!

Install & Usage
---
Since the version ``18.9.15``, we provide the detail manual. You can refer to the [book](https://torchvision-sunner-book.readthedocs.io/en/latest/) to check the detail (including tutorial of the toolkit)! Or you can copy the following command to install. 

```
git clone https://github.com/SunnerLi/Torchvision_sunner.git && cd Torchvision_sunner/ && git fetch origin master && cd .. && mv Torchvision_sunner/torchvision_sunner/ . && rm -rf Torchvision_sunner/
```

Requirement
---
Since the ``18.9.15``, we raise the limitation of some package. You should check these requirement and update to the newest version as well! We also provide the old version, and you can check for others branch.    
* Pytorch: We don't support the version which under ``0.4.1``
* OpenCV: We don't adopt OpenCV as the back-end now because the installation of OpenCV is time-consuming. But you can still use OpenCV to do the further development. Check for tutorial for detail!