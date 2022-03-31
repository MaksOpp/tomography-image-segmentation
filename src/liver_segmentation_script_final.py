import nibabel as nib
import numpy as np
import tensorflow as tf
from termcolor import colored

def f1(y_true, y_pred):
    return 1

def save_liver_segmentation(liver_image_path, model_path):
    print("Application started! Loading data...")
    min_img_bound = -1000
    max_img_bound = 3000

    img = nib.load(liver_image_path)
    img_fdata = img.get_fdata()
    img_fdata = np.clip(img_fdata, min_img_bound, max_img_bound)

    img_fdata = (img_fdata - np.mean(img_fdata)) / np.std(img_fdata)
    img_fdata = (
        (img_fdata - np.min(img_fdata)) / (np.max(img_fdata) - np.min(img_fdata)))

    (x,y,z) = img_fdata.shape
    data = []
    for k in range(z):
        slice = img_fdata[:,:,k]
        slice = slice.astype(np.float64) / slice.max()
        slice = 255 * slice
        slice = slice.astype(np.uint8)
        slice = np.rot90(slice)
        data.append(slice)
    print(colored("Data loaded!\n", "green"))
    print("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'f1':f1})
    print(colored("Model loaded!\n", "green"))
    print("Predicting liver mask...")
    result_data = [np.rot90(model.predict(x.reshape(1,512,512,1)).reshape(512,512),3) for x in data]

    result = np.array(result_data).swapaxes(0, 2).swapaxes(0,1)

    (x_dim,y_dim,i_dim) = result.shape
    data = []
    result_image = img_fdata.copy()
    for x in range(x_dim):
        for y in range(y_dim):
            for i in range(i_dim):
                if result[x][y][i] > 0.5:
                    result_image[x][y][i] = 1

    new_image = nib.Nifti1Image(result_image, affine=img.affine)
    nib.save(new_image, '/content/result_tumor_new.nii.gz')  
    print(colored("Mask prepared and exported to ", "green"), colored("result_tumor_new.nii.gz", "green"))
