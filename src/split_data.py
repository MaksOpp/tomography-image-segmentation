#!/usr/bin/env python
# coding: utf-8

# In[95]:


import glob
import os
import math
import shutil

from pathlib import Path

INPUT_FILES_PATH = 'merged/'
OUTPUT_FILES_PATH = 'final_data/'


# In[91]:


def train_val_test_split(subsets_proportion_percentage):
    try: 
        if (sum(subsets_proportion_percentage) != 100):
            raise Exception('The sum of the percentages of the subsets does not equal 100%')
        if (len(subsets_proportion_percentage) != 3):
            raise Exception('Invalid proportions (remember about 3 values - train, validation and test percentage values)')
            
        files = list(set([Path(x).stem.split('i')[0] for x in glob.glob(INPUT_FILES_PATH + 'images/*')]))   
        length_to_split = [x / 100 * len(files) for x in subsets_proportion_percentage]
        val_frac, val_whole = math.modf(length_to_split[1])
        test_frac, test_whole = math.modf(length_to_split[2])
        train_count, val_count, test_count = [int(math.ceil(length_to_split[0] + val_frac + test_frac)), int(val_whole), int(test_whole)]
        
        random.seed(21)
        random.shuffle(files)
        
        return files[:train_count], files[train_count:train_count+val_count], files[train_count+val_count:]
    except Exception as error:
        print('Caught this error: ' + repr(error))  


# In[92]:


subsets_proportion_percentage = [70, 15, 15]
train_files, val_files, test_files = train_val_test_split(subsets_proportion_percentage)

print(train_files)
print(val_files)
print(test_files)


# In[121]:


def copy_files(files, images_path, labels_path):
    os.makedirs(images_path)
    os.makedirs(labels_path)

    for file in files: 
        for file_path in glob.glob(INPUT_FILES_PATH + 'labels/' + file + 'i*'):
            shutil.copy2(file_path, labels_path) 
        for file_path in glob.glob(INPUT_FILES_PATH + 'images/' + file + 'i*'):
            shutil.copy2(file_path, images_path)   

def fill_split_subfolders():
    if os.path.exists(OUTPUT_FILES_PATH):
        shutil.rmtree(OUTPUT_FILES_PATH)
        
    copy_files(train_files, OUTPUT_FILES_PATH+'train/images', OUTPUT_FILES_PATH+'train/labels')
    copy_files(val_files, OUTPUT_FILES_PATH+'val/images', OUTPUT_FILES_PATH+'val/labels')
    copy_files(test_files, OUTPUT_FILES_PATH+'test/images', OUTPUT_FILES_PATH+'test/labels')


fill_split_subfolders()


# In[ ]:




