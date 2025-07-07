import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
import os
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
import tensorflow as tf
import random

# Set all random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# For GPU determinism (if using GPU)
tf.config.experimental.enable_op_determinism()

num_class=7
img_rows,img_cols=48,48 #image size
batch_size=16 #decrease from 32

train_data=r'C:\Users\nourm\OneDrive\Desktop\PROJECTS\UNITY MODEL PROJECT\FER-2013 (USED)\train'

validation_data=r'C:\Users\nourm\OneDrive\Desktop\PROJECTS\UNITY MODEL PROJECT\FER-2013 (USED)\test'

#image data generator section

#here we generated some images 

train_dataGen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,           # Increase from 10 to 15
    shear_range=0.10,            # Increase from 0.1 to 0.2 back to 0.15
    zoom_range=0.10,             # Increase from 0.1 to 0.2 back to 0.15 back to 0.1
    width_shift_range=0.10,      # Increase from 0.1 to 0.2 back to 0.15
    height_shift_range=0.10,     # Increase from 0.1 to 0.2 back to 0.15
    horizontal_flip=True
    #brightness_range=[0.8, 1.2], # ADD: brightness variation- remove for now
    #channel_shift_range=0.1      # ADD: slight channel shift remove for now 
)


validation_dataGen=ImageDataGenerator(rescale=1./255)

#give to model to train-train parameters

train_generator = train_dataGen.flow_from_directory(train_data,
color_mode='grayscale',
 target_size=(img_rows,img_cols) , 
 batch_size=batch_size,
   class_mode='categorical', 
   shuffle=True
   )

#validation parameters
validation_generator = validation_dataGen.flow_from_directory(validation_data,
color_mode='grayscale',
 target_size=(img_rows,img_cols) , 
 batch_size=batch_size,
   class_mode='categorical', 
   shuffle=True
   )

model= Sequential()

#change, block 1-4, increase dropout from 0.2 to 0.3 0.2
#change, block 5-6, decrease dropout from 0.5 to 0.4 0.3


#Block 1 - had 2 layer of convutional 2d here
#32 neurens in 1st layer
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal', input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))  #set threshhold value for every neuron
model.add(BatchNormalization()) #technique for neural network- stabalise /improve performance
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal')) #32 neurons in 2nd layer
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))  
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


#Block 2 -no input size , 64 neurons
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')) #32 neurens in 1st layer
model.add(Activation('elu'))  #set threshhold value for every neuron
model.add(BatchNormalization()) #technique for neural network- stabalise /improve performance
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')) #32 neurons in 2nd layer
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))  
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))



#Block 3 -no input size , 128 neurons
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization()) #technique for neural network- stabalise /improve performance
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))  #32 neurons in 2nd layer
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))  
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))



#Block4 -no input size , 256 neurons
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal')) #32 neurens in 1st layer
model.add(Activation('elu')) #set threshhold value for every neuron
model.add(BatchNormalization()) #technique for neural network- stabalise /improve performance
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal')) #32 neurons in 2nd layer
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))  
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


#Block 5 -flatten!  - Two fully connected layers with 64 neurons
model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Block-6  -Two fully connected layers with 64 neurons
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))



# Block-7

model.add(Dense(num_class,kernel_initializer='he_normal'))

model.add(Activation('softmax'))

print(model.summary())


from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(r'C:\Users\nourm\OneDrive\Desktop\PROJECTS\UNITY MODEL PROJECT\Emotion_little_vgg.h5',
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True,
                            verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                         min_delta=0,
                         patience=7, #increase from 3
                         verbose=1,
                         restore_best_weights=True
                         )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                            factor=0.5, #change from 0.2
                            patience=5, #increase from 3
                            verbose=1,
                            min_delta=0.0001)

callbacks = [earlystop,reduce_lr]

optimizer = Adam(learning_rate=0.0001)  # Reduced from 0.001

model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy', 'precision', 'recall'])

nb_train_samples = train_generator.samples
nb_validation_samples = validation_generator.samples
epochs=50 #how many times to train same image, model will go 25 times in every folder #increase from 25


history=model.fit(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)

model.save('Emotion_model_14.h5')