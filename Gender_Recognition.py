# Gender recognition using CNN & ResNet50 (Transfer learning) 2021


# import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import glob
import random
from keras.optimizers import Adam
from keras.models import Sequential,load_model,Model
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,BatchNormalization,Flatten,Input,Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns


# config
path = "gender_dataset_face"
img_height = 96
img_width = 96
batch_size = 64
images = []
labels = []


# load images in the dataset
image_files = [f for f in glob.glob(path + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)


# images and labels
for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_width,img_height))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    images.append(np.array(image))

    label = img.split(os.path.sep)[-2] # gender_dataset_face\woman\img
    if label == "woman":
        label = 1
    else:
        label = 0
    labels.append([label])


# pre-processing
images = np.array(images, dtype="float") / 255.0
labels = np.array(labels,dtype=np.uint64)


# split data into training & testing
(x_train, x_test, y_train, y_test) = train_test_split(images, labels,test_size=0.3, random_state=42)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
augmentation = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# ==============================================================
# CNN model
model_g = Sequential()
#1st layer
model_g.add(Conv2D(64,(3,3),padding = 'same',input_shape = (img_width,img_height,3)))
model_g.add(Activation('relu'))
model_g.add(MaxPooling2D(pool_size = (2,2)))
model_g.add(Dropout(0.25))
#2nd layer
model_g.add(Conv2D(128,(5,5),padding = 'same'))
model_g.add(Activation('relu'))
model_g.add(MaxPooling2D(pool_size = (2,2)))
model_g.add(Dropout (0.25))
#3rd layer
model_g.add(Conv2D(512,(3,3),padding = 'same'))
model_g.add(Activation('relu'))
model_g.add(MaxPooling2D(pool_size = (2,2)))
model_g.add(Dropout (0.25))
#4th layer
model_g.add(Conv2D(512,(3,3), padding='same'))
model_g.add(Activation('relu'))
model_g.add(MaxPooling2D(pool_size=(2, 2)))
model_g.add(Dropout(0.25))
model_g.add(Flatten())
#Fully connected 1st layer
model_g.add(Dense(256))
model_g.add(Activation('relu'))
model_g.add(Dropout(0.25))
# Fully connected layer 2nd layer
model_g.add(Dense(512))
model_g.add(BatchNormalization())
model_g.add(Activation('relu'))
model_g.add(Dropout(0.25))
model_g.add(Dense(2, activation='sigmoid'))
model_g.summary()


# compile the model
model_g.compile(loss="binary_crossentropy", optimizer = Adam(lr=0.001), metrics=["accuracy"])
checkpoint = ModelCheckpoint("./model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss',min_delta=0, patience=3, verbose=1, restore_best_weights=True)
reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',factor=0.2, patience=3, verbose=1, min_delta=0.0001)
callbacks_list = [early_stopping,checkpoint,reduce_learningrate]


# train the model
history_g = model_g.fit_generator(augmentation.flow(x_train, y_train, batch_size=batch_size), validation_data=(x_test,y_test), steps_per_epoch=len(x_train) // batch_size, epochs=50, verbose=1, callbacks=callbacks_list)


# prediction & evaluation
results_g = model_g.evaluate(x_test,y_test)
preds_g = model_g.predict(x_test)
print(' Model accuracy {}%'.format(round(results_g[1]*100, 2)))


# visualise training and testing accuracy & loss
def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

plot_results(history_g)


# saving model
model_g.save("./models/Gender_CNN_model.h5")

# ==============================================================
# ResNet50 model
resnet_model = Sequential()
pretrained_model= ResNet50(include_top=False, input_shape=(img_width,img_height,3), pooling='avg',classes=2, weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(2, activation='sigmoid'))
resnet_model.summary()


# compile the model
resnet_model.compile(optimizer = Adam(lr=0.001), loss='binary_crossentropy',metrics=['accuracy'])
checkpoint = ModelCheckpoint("./model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss',min_delta=0, patience=3, verbose=1, restore_best_weights=True)
reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',factor=0.2, patience=3, verbose=1, min_delta=0.0001)
callbacks_list = [early_stopping,checkpoint,reduce_learningrate]


# train the model
history_g_res = resnet_model.fit(x_train, y_train,validation_data=(x_test, y_test), epochs=50,callbacks=callbacks_list)

 
# prediction & evaluation
results = resnet_model.evaluate(x_test,y_test)
preds = resnet_model.predict(x_test)
print(' Model accuracy {}%'.format(round(results[1]*100, 2)))


# visualise training and testing accuracy & loss
def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(24, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

plot_results(history_g_res)


# saving model
resnet_model.save("./models/Gender_ResNet50_model.h5")

# ==============================================================
#Test ResNet50 model
my_Res_model = load_model('./models/Gender_ResNet50_model.h5', compile=False)
predictions_Res = my_Res_model.predict(x_test)
y_pred_Res = (predictions_Res>= 0.5).astype(int)[:,0]
y_test_p = y_test.astype(int)[:,0]
print ("Accuracy = ", metrics.accuracy_score(y_test_p, y_pred_Res)) # Accuracy = 0.8311688311688312
print( "Actual value :" , y_test_p[8], "/ Predicted value :" , y_pred_Res[8])
cm=confusion_matrix(y_test_p, y_pred_Res)  
sns.heatmap(cm, annot=True)


#Test CNN model
my_CNN_model = load_model('./models/Gender_CNN_model.h5', compile=False)
predictions_CNN = my_CNN_model.predict(x_test)
y_pred_CNN = (predictions_CNN>= 0.5).astype(int)[:,0]
y_test_p = y_test.astype(int)[:,0]
print ("Accuracy = ", metrics.accuracy_score(y_test_p, y_pred_CNN)) # Accuracy = 0.9292929292929293
print( "Actual value :" , y_test_p[8], "/ Predicted value :" , y_pred_CNN[8])
cm=confusion_matrix(y_test_p, y_pred_CNN)  
sns.heatmap(cm, annot=True)