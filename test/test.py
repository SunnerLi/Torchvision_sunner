# -------------------------------------------------------------------
#   Test code to check the function of whole torchvisiov_sunner wrapper
#   This code will check the function of example code and tutorial automatically
#   You don't need to use this code while using this wrapper
#
#   Author  : SunnerLi
#   Date    : 5/17/2018
# -------------------------------------------------------------------
import subprocess
import os

# Record the phenomenon of original folder
folder_list = os.listdir('./')

# Check if the dataset is downloaded first!!!
if not os.path.exists('./waiting_for_you_dataset'):
    raise Exception('You should download the dataset before you start the testing...')

# Copy the wrapper here
cmd = ['cp', '-r', '../torchvision_sunner', '.']
subprocess.call(cmd)

# Copy the example code here and test it!
import glob
code_list = sorted(glob.glob('../tutorial_and_example/*.py'))
for code_path in code_list:
    cmd = ['cp', code_path, '.']
    subprocess.call(cmd)
    cmd = ['python3', code_path.split('/')[-1]]
    subprocess.call(cmd)
    cmd = ['rm', code_path.split('/')[-1]]
    subprocess.call(cmd)

# Pass the test and remove the copied files
print('-' * 50)
print('\t Pass the testing !')
print('-' * 50)
for file_name in os.listdir('./'):
    if file_name not in folder_list:
        cmd = ['rm', '-r', file_name]
        subprocess.call(cmd)