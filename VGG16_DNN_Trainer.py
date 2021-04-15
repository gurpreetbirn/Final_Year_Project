from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator ##preprocess images to be able to rotate,flip and shift for more accurate learning
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

NumberOfEmotions = 5 ## 5 kinds of emotion angry sad happy neutral and surprise
imageSize = 48
imageRow,imageColumn = imageSize,imageSize ## this are the target size of the images that will be getting trained
batch_size = 32 ## 32 imgages at one time for training

##both these are used for testing 
trainingDatabaseFilePath = '/Users/gurpr/Desktop/ce301_birn_g/python_desktop_app/train_database'\
                 ##define the training data set directory
validationDatabaseFilePath = '/Users/gurpr/Desktop/ce301_birn_g/python_desktop_app/validation-database'\
                      ##define the validation data set directory
##validation needed to test the trained data set to ensure the accuracy of the DNN


imageGenForTraining = ImageDataGenerator(##this will generate atered verions of current pictures for better accuracy by flipping and rotating images in the dataset
					rescale=1./255,##rescale/normalise each pixel so it is scaled from 0 - 1
					rotation_range=30,##rotate image 30 degrees to the left and right
					shear_range=0.3,##30%
					zoom_range=0.3,##30%
					width_shift_range=0.4,##40%
					height_shift_range=0.4,##40%
					horizontal_flip=True,##image is flipped horizontally
					fill_mode='nearest')##if image is shifted to the right the pixels to the left are lost, so this fills with nearest pixels to the right 

imageValidationNormalising = ImageDataGenerator(rescale=1./255)##image only being normailsed no new image being generated 

##will train imageGenForTraining variable
GenTrainingFromImages = imageGenForTraining.flow_from_directory( ## this line is where the datasets are accessed from files
					trainingDatabaseFilePath,
					color_mode='grayscale',##dont need colour to classify emotions if it was animals for example then fur colour might be needed to ditinguish diffferent types
					target_size=(imageRow,imageColumn),
					batch_size=batch_size,
					class_mode='categorical',##this is where the class is determined out of the 5 emotions
					shuffle=True) ##data is shuffled to prevent cheating etc

##has to be the same as GenTrainingFromImages
GenTrainingValidation = imageValidationNormalising.flow_from_directory(
							validationDatabaseFilePath,
							color_mode='grayscale',
							target_size=(imageRow,imageColumn),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)

#define convolultional neural net

DNNmodel = Sequential()

# Block-1
##

DNNmodel.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(imageRow,imageColumn,1)))##32,3,3
DNNmodel.add(Activation('elu'))##elu instead of relu? why is this, what is the difference
DNNmodel.add(BatchNormalization())
DNNmodel.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(imageRow,imageColumn,1)))##32,3,3
DNNmodel.add(Activation('elu'))
DNNmodel.add(BatchNormalization())
DNNmodel.add(MaxPooling2D(pool_size=(2,2)))
DNNmodel.add(Dropout(0.2))

# Block-2 
##no input shape needed for this layer as block 1 layer will do this
DNNmodel.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))##64,3,3
DNNmodel.add(Activation('elu'))
DNNmodel.add(BatchNormalization())
DNNmodel.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
DNNmodel.add(Activation('elu'))
DNNmodel.add(BatchNormalization())
DNNmodel.add(MaxPooling2D(pool_size=(2,2)))
DNNmodel.add(Dropout(0.2))##this means 20% of the time this neuron will turn off, only uses 80%

# Block-3

DNNmodel.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
DNNmodel.add(Activation('elu'))
DNNmodel.add(BatchNormalization())
DNNmodel.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
DNNmodel.add(Activation('elu'))
DNNmodel.add(BatchNormalization())
DNNmodel.add(MaxPooling2D(pool_size=(2,2)))
DNNmodel.add(Dropout(0.2))

# Block-4 

DNNmodel.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
DNNmodel.add(Activation('elu'))
DNNmodel.add(BatchNormalization())
DNNmodel.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
DNNmodel.add(Activation('elu'))
DNNmodel.add(BatchNormalization())
DNNmodel.add(MaxPooling2D(pool_size=(2,2)))
DNNmodel.add(Dropout(0.2))

##CNN is completed


# Block-5

DNNmodel.add(Flatten())##no matrix needed so this makes it 1 dimentional
DNNmodel.add(Dense(64,kernel_initializer='he_normal'))##fully connected  layer 64 neurons
DNNmodel.add(Activation('elu'))
DNNmodel.add(BatchNormalization())
DNNmodel.add(Dropout(0.5))##50% of neurons will be activated

# Block-6

DNNmodel.add(Dense(64,kernel_initializer='he_normal'))##fully connected  layer 64 neurons
DNNmodel.add(Activation('elu'))
DNNmodel.add(BatchNormalization())
DNNmodel.add(Dropout(0.5))

# Block-7

##look up whart the kernal)initializer does properly
DNNmodel.add(Dense(NumberOfEmotions,kernel_initializer='he_normal'))##numclasses , we only need 5 neurons in this spot as there are only 5 emotions that we are looking for
DNNmodel.add(Activation('softmax'))

print(DNNmodel.summary())##to see the architecture of the model

from keras.optimizers import RMSprop,SGD,Adam ## 3 different optimisers, not all three will be used
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
##following comments are for the line above
##ModelCheckpoint will save the model after every epoch, saves the best model
##EarlyStopping, this will stop the training when the validation times will no longer improve
##ReduceLROnPlateau this will reduce the learning rate when there is no imporvement seen for a number of epochs

##trainedEmotionModel.h5 is the name of the model and will be saved in the current working directory
modelCheckpoint = ModelCheckpoint('trainedEmotionModel.h5',
                             monitor='val_loss',##need to see if this is decreasing, why?
                             mode='min',
                             save_best_only=True,
                             verbose=1)
##read about this is keras documentation 
early_stopping = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=9,##this will determine how long the system will wait before moving to the net epoch
                          verbose=1,
                          restore_best_weights=True
                          )

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,##decrease learning rate by facotor of 0.2
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [early_stopping,modelCheckpoint,reduceLROnPlat]

##compile model and finish training

DNNmodel.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

trainingSet = 24176
validationSet = 3006
epochs=25

##training
history=DNNmodel.fit_generator(
                GenTrainingFromImages,
                steps_per_epoch=trainingSet//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=GenTrainingValidation,
                validation_steps=validationSet//batch_size)




