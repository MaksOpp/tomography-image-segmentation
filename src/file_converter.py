import os
import glob
import pydicom
import imageio
import shutil
import natsort 
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image

import yaml_config
config = yaml_config.getContentFromFile('../config/file_converter.yaml')

LITS_PATHS_FILE = config['lits_paths']
PG_PATHS_FILE = config['pg_paths']
ROTATE_PATHS_FILE = config['rotate_paths']
MERGE_PATHS_FILE = config['merge_paths']

MIN_IMG_BOUND = config['min_img_bound']
MAX_IMG_BOUND = config['max_img_bound']

class NiiFileConverter:
    @staticmethod
    def set_bounds(img, MIN_BOUND, MAX_BOUND):
        image = np.clip(img, MIN_BOUND, MAX_BOUND)
        return image
    
    @staticmethod
    def normalize(img):
        image = (img - np.mean(img)) / np.std(img)
        image = (
            (image - np.min(image)) / (np.max(image) - np.min(image)))
        return image

    @staticmethod
    def read_nii_file(nii_file_path):
        img = nib.load(nii_file_path)
        img_fdata = img.get_fdata()
        #img_fdata = intensity_normalization.normalize.zscore.zscore_normalize(img, mask=None)
        img_fdata = NiiFileConverter.set_bounds(img_fdata, MIN_IMG_BOUND, MAX_IMG_BOUND)
        img_fdata = NiiFileConverter.normalize(img_fdata)
        return img_fdata

    @staticmethod
    def save_as_png(nii_file_path, dest):
        fdata = NiiFileConverter.read_nii_file(nii_file_path)
        (x,y,z) = fdata.shape
        for k in range(z):
            slice = fdata[:,:,k]
            imageio.imwrite(os.path.join(dest,'{}.png'.format(k)),slice)


class DicomFileConverter:
    @staticmethod
    def read_dicom_file(dicom_file_path):
        ds = pydicom.read_file(dicom_file_path)
        return ds

    @staticmethod
    def save_as_png(dicom_file_path, dest, filename):
        ds = DicomFileConverter.read_dicom_file(dicom_file_path)
        img = ds.pixel_array.astype(float)
        img = (np.maximum(img,0)/img.max())*255
        img = np.uint8(img)
        imageio.imwrite(os.path.join(dest,'{}.png'.format(filename)),img)

class LitsDbConverter:
    def __init__(self, path_list):
        self.path_list = path_list

    def save_as_png(self):
        lines = self.path_list
        for line in lines:
            ext, source, dest = line.split()

            if not os.path.exists(dest):
                os.makedirs(dest)

            if(ext == 'nii.gz'):
                self.save_nii_files(source, dest)

    def save_nii_files(self, source, dest):
        files = glob.glob(f'{source}/*')
        for file in files:
            filename = os.path.splitext(os.path.splitext(file)[0])[0].split('\\')[-1]
            dest_path = dest + filename
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
                NiiFileConverter.save_as_png(file, dest_path)


class PgDbConverter:
    def __init__(self, path_list):
        self.path_list = path_list

    def save_as_png(self):
        self.save_type_as_png('*_P.nii.gz', 'pg_p')
        self.save_type_as_png('*_V.nii.gz', 'pg_v')
        self.save_type_as_png('*_V+P.nii.gz', 'pg_v_p')
        self.save_type_as_png('*_V+P+Vesicle.nii.gz', 'pg_v_p')
        self.save_type_as_png('*_V+P+Vesicle.nii.gz', 'pg_v_p_vesicle')
        self.save_type_as_png('*_T.nii.gz', 'tumors')

    def save_type_as_png(self, ext, dest_subfolder):
        lines = self.path_list
        for line in lines:
            images_path, labels_path, dest = line.split()

            type_dest = f'{dest}{dest_subfolder}\\'

            if not os.path.exists(type_dest):
                os.makedirs(type_dest)

            type_paths = Path(labels_path).rglob(ext)
            
            for path in type_paths:
                examination_num, series_num, ext = path.name.split('_')

                self.save_nii_file(path, type_dest, examination_num)
                self.save_dicom_files(images_path, type_dest, examination_num, series_num)

    
    def save_nii_file(self, path, type_dest, examination_num):
        destPath = f'{type_dest}labels\{examination_num}\\'

        if not os.path.exists(destPath):
            os.makedirs(destPath)
            NiiFileConverter.save_as_png(path, destPath)


    def save_dicom_files(self, images_path, type_dest, examination_num, series_num):
        selected_image_directory = f'{images_path}\\{examination_num}\\DICOMS\\STU00001\\SER000{series_num}'
        files = glob.glob(f'{selected_image_directory}/*')
        for file in files:
            filename = os.path.splitext(os.path.splitext(file)[0])[0].split('\\')[-1]

            directory_path = f'{type_dest}images\{examination_num}\\'

            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            if not os.path.exists(f'{directory_path}\\{filename}\\.png'):
                filename = os.path.splitext(os.path.splitext(file)[0])[0].split('\\')[-1]
                DicomFileConverter.save_as_png(file, directory_path, filename)

class ImageRotator:
    def __init__(self, path):
        self.path = path

    def save_rotated(self):
        lines = open(self.path, "r").readlines()
        for line in lines:
            source, dest = line.split()
            
            if not os.path.exists(dest):
                os.makedirs(dest)
                
            self.rotate(source, dest)
            
    def rotate(self, source, dest):    
        directories = glob.glob(f'{source}/*')
        for directory in directories:
            files = glob.glob(f'{directory}/*')
            for file in files:
                img = Image.open(file)
                img = img.rotate(90) 
                os.remove(file)
                img.save(file) 

class DirectoryMerger:
    def __init__(self, path):
        self.path = path
    def save_merged(self):
        lines = open(self.path, "r").readlines()
        for line in lines:
            source, dest = line.split()
            
            if not os.path.exists(dest):
                os.makedirs(dest)
                
            self.merge(source, dest)
    def merge(self, source, dest):
        directories = natsort.natsorted(glob.glob(f'{source}/*'))
        caseNumber = 0
        for directory in directories:
            files = natsort.natsorted(glob.glob(f'{directory}/*'))
            imageNumber = 0
            for file in files:
                print(file)
                shutil.copy2(file, dest+'p'+str(caseNumber)+'i'+str(imageNumber)+'.png')
                imageNumber = imageNumber + 1
            caseNumber = caseNumber + 1
        

def main():
    lits_converter = LitsDbConverter(LITS_PATHS_FILE)
    pg_converter = PgDbConverter(PG_PATHS_FILE)
    
    lits_converter.save_as_png()
    pg_converter.save_as_png()
    
    image_rotator = ImageRotator(ROTATE_PATHS_FILE)
    image_rotator.save_rotated()
    
    directory_merger = DirectoryMerger(MERGE_PATHS_FILE)
    directory_merger.save_merged()
    
if __name__ == "__main__":
    main()