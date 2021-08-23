#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import glob


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# In[2]:


framObjTrain = {'img' : [],
           'mask' : []
          }



def LoadData( frameObj = None, imgPath = None, maskPath = None, shape = 256):
    imgNames = os.listdir(imgPath)
    
    imgAddr = imgPath + '/'
    maskAddr = maskPath + '/'
    
    for i in range (len(imgNames)):
        img = plt.imread(imgAddr + imgNames[i])
        mask = plt.imread(maskAddr + imgNames[i])
        
        img = cv2.resize(img, (shape, shape)) 
        mask = cv2.resize(mask, (shape, shape))
        
        frameObj['img'].append(img)
        frameObj['mask'].append(mask)
        
    return frameObj


# In[3]:


#img_path = './/raid//kamkac//hepatic//images//hepaticvessel_002'
#mask_path = './/raid//kamkac//hepatic//labels//hepaticvessel_002'
# img_path = './/raid//kamkac//liver//images//liver_0'
# mask_path = './/raid//kamkac//liver//labels//liver_0'
img_path = './/raid//kamkac//merged//images'
mask_path = './/raid//kamkac//merged//labels'

framObjTrain = LoadData( framObjTrain, imgPath = img_path
                        , maskPath = mask_path
                         , shape = 512)


# In[4]:


plt.figure(figsize = (10, 7))
plt.subplot(1,2,1)
plt.imshow(framObjTrain['img'][55])
plt.title('Image')
plt.subplot(1,2,2)
plt.imshow(framObjTrain['mask'][55])
plt.title('Mask')
plt.show()

# In[6]:

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

prefix_path = "checkpoints\\"
if not os.path.exists(prefix_path):
    os.makedirs(prefix_path)

checkpoint_path = prefix_path + "{epoch:04d}.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)
list_of_files = glob.glob(prefix_path + "*.hdf5")
latest = max(list_of_files, key=os.path.getctime)
exists = latest and os.path.isfile(latest)

if(exists):
    model.load_weights(latest)
    print("Weights from checkpoint file loaded")


model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, mode='max', save_freq="epoch")
callbacks_list = [checkpoint]
model.summary()


# In[14]:

initial_epoch = 0
if(exists):
    initial_epoch = int(latest.replace(prefix_path, "").replace(".hdf5", ""))
    print("Start from epoch", initial_epoch)

retVal = model.fit(np.array(framObjTrain['img']), np.array(framObjTrain['mask']), epochs = 10, verbose = 1, validation_split = 0.1, callbacks=callbacks_list, initial_epoch=initial_epoch)

plt.plot(retVal.history['accuracy'])
plt.plot(retVal.history['val_accuracy'])
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.savefig('accvsepochs.png')

plt.plot(retVal.history['loss'])
plt.plot(retVal.history['val_loss'])
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.savefig('loss.png')



# In[8]:


plt.plot(retVal.history['loss'], label = 'training_loss')
plt.plot(retVal.history['accuracy'], label = 'training_accuracy')
plt.legend()
plt.grid(True)


# In[9]:

import uuid
def predict16 (valMap, model, shape = 512):
    ## getting and proccessing val data
    img = valMap['img'][0:16]
    mask = valMap['mask'][0:16]
    #mask = mask[0:16]
    
    imgProc = img [0:16]
    imgProc = np.array(img)
    
    predictions = model.predict(imgProc)
  

    return predictions, imgProc, mask


def Plotter(img, predMask, groundTruth):
    name = uuid.uuid4()
    plt.figure(figsize=(9,9))
    
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title(' image')
    
    plt.subplot(1,3,2)
    plt.imshow(predMask)
    plt.title('Predicted mask')
    
    plt.subplot(1,3,3)
    plt.imshow(groundTruth)
    plt.title('Actual mask')
    plt.savefig(name+'.png')



# In[10]:


sixteenPrediction, actuals, masks = predict16(framObjTrain, model)
Plotter(actuals[1], sixteenPrediction[1][:,:,0], masks[1])


# In[11]:


Plotter(actuals[2], sixteenPrediction[2][:,:,0], masks[2])


# In[12]:


Plotter(actuals[3], sixteenPrediction[3][:,:,0], masks[3])


# In[13]:


model.save('Segmentor.h5')


# In[ ]:




