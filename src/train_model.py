import sys, os
import random
import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import glob

import yaml_config
config = yaml_config.takeConfig('../config/train_model.yaml')

os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_visible_devices']
img_path = config['path']['img']
mask_path = config['path']['mask']
validation_img_path = config['path']['val_img']
validation_mask_path = config['path']['val_mask']

NO_OF_EPOCHS = config['epochs']
BATCH_SIZE = config['batch_size']
NO_OF_TRAINING_IMAGES = len(os.listdir(img_path))
NO_OF_VAL_IMAGES = len(os.listdir(validation_img_path))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def data_gen(img_folder, mask_folder, batch_size):
  c = 0
  n = os.listdir(img_folder) #List of training images
  random.shuffle(n)
  
  while (True):
    img = np.zeros((batch_size, 512, 512, 1)).astype('float')
    mask = np.zeros((batch_size, 512, 512, 1)).astype('float')

    for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 
        
      img1 = plt.imread(img_folder+'/'+n[i])
      mask1 = plt.imread(mask_folder+'/'+n[i])

      img1 = cv2.resize(img1, (512, 512)) 
      mask1 = cv2.resize(mask1, (512, 512))

      img1 = np.asarray(img1)
      mask1 = np.asarray(mask1)
        
      img1 = np.expand_dims(img1, axis=2)
      mask1 = np.expand_dims(mask1, axis=2)

      img[i-c] = img1
      mask[i-c] = mask1
      
    c+=batch_size
    if(c+batch_size>=len(os.listdir(img_folder))):
      c=0
      random.shuffle(n)
        
    yield img, mask

train_gen = data_gen(img_path, mask_path, BATCH_SIZE)
val_gen = data_gen(validation_img_path, validation_mask_path, BATCH_SIZE)

#Build the model
inputs = tf.keras.layers.Input((512, 512, 1))
s = tf.keras.layers.Lambda(lambda x: x / 256)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = tf.keras.backend.sum(y_true, axis=[1,2,3]) + tf.keras.backend.sum(y_pred, axis=[1,2,3])
    return tf.keras.backend.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_loss(y_true, y_pred):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.math.sigmoid(y_pred)
  numerator = 2 * tf.reduce_sum(y_true * y_pred)
  denominator = tf.reduce_sum(y_true + y_pred)

  return 1 - numerator / denominator

model.compile(optimizer = 'adam', loss = dice_loss, metrics = ['accuracy', dice_coef])

retVal = model.fit(train_gen, 
                   epochs=NO_OF_EPOCHS, 
                   steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                   validation_data=val_gen, 
                   validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE))