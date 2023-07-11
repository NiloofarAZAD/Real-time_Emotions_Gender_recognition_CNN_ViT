# Face Emotions recognition using CNN 2021
# Dataset : Fer 2013


# import libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Activation
import os
from matplotlib import pyplot as plt
import numpy as np
import random
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from sklearn import metrics
import seaborn as sns


# config
img_height = 48
img_width = 48
batch_size = 128


# load dataset
train_data_directory = './data/train/'
validation_data_directory = './data/test/'


# visualize images
def plot_exp(expression):
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 12))
    for i in range(1, 10, 1):
        plt.subplot(3, 3, i)
        img = load_img(train_data_directory + expression+"/" + os.listdir(train_data_directory + expression)[i], target_size=(img_width,img_height))
        plt.imshow(img)
    plt.show()
plot_exp('happy')


# build data generator
train_data_gen = ImageDataGenerator()
validation_data_gen = ImageDataGenerator()

train_set = train_data_gen.flow_from_directory(train_data_directory, target_size=(img_width,img_height), color_mode="grayscale", batch_size=batch_size, class_mode='categorical', shuffle=True)

test_set = validation_data_gen.flow_from_directory(validation_data_directory, target_size=(img_width,img_height), color_mode="grayscale", batch_size=batch_size, class_mode='categorical', shuffle=False)


class_labels = ['Angry', 'Disgust', 'Fear','Happy', 'Natural', 'Sad', 'Surprise']
img, label = train_set.__next__()
print(img.shape)


# CNN Model
model_1 = Sequential()
#1st layer
model_1.add(Conv2D(64,(3,3),padding = 'same',input_shape = (img_width,img_height,1)))
model_1.add(Activation('relu'))
model_1.add(MaxPooling2D(pool_size = (2,2)))
model_1.add(Dropout(0.25))
#2nd layer
model_1.add(Conv2D(128,(5,5),padding = 'same'))
model_1.add(Activation('relu'))
model_1.add(MaxPooling2D(pool_size = (2,2)))
model_1.add(Dropout (0.25))
#3rd layer
model_1.add(Conv2D(512,(3,3),padding = 'same'))
model_1.add(Activation('relu'))
model_1.add(MaxPooling2D(pool_size = (2,2)))
model_1.add(Dropout (0.25))
#4th layer
model_1.add(Conv2D(512,(3,3), padding='same'))
model_1.add(Activation('relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Dropout(0.25))
model_1.add(Flatten())
#Fully connected 1st layer
model_1.add(Dense(256))
model_1.add(Activation('relu'))
model_1.add(Dropout(0.25))
# Fully connected layer 2nd layer
model_1.add(Dense(512))
model_1.add(BatchNormalization())
model_1.add(Activation('relu'))
model_1.add(Dropout(0.25))
model_1.add(Dense(7, activation='softmax'))
model_1.summary()


# compile the model
checkpoint = ModelCheckpoint("./models/FER_CNN_model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss',min_delta=0, patience=3, verbose=1, restore_best_weights=True)
reduce_learningrate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)
callbacks_list = [early_stopping,checkpoint,reduce_learningrate]
model_1.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
history_1 = model_1.fit_generator(generator=train_set, steps_per_epoch=train_set.n//train_set.batch_size, epochs=50,
                                  validation_data=test_set, validation_steps=test_set.n//test_set.batch_size, callbacks=callbacks_list)


# prediction & evaluation
results = model_1.evaluate(test_set)
preds = model_1.predict(test_set)
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

plot_results(history_1)


# saving model
model_1.save("./models/FER_CNN_model.h5")


# Test the model
my_model = load_model('./models/FER_CNN_model.h5', compile=False)
test_img, test_lbl = test_set.__next__()
predictions = my_model.predict(test_img)
predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_lbl, axis=1)
print("Accuracy = ", metrics.accuracy_score(test_labels, predictions))


# confusion matrix
cm = confusion_matrix(test_labels, predictions)
sns.heatmap(cm, annot=True)


# prediction on random images
class_labels = ['Angry', 'Disgust', 'Fear','Happy', 'Neutral', 'Sad', 'Surprise']
n = random.randint(0, test_img.shape[0] - 1)
image = test_img[n]
orig_labl = class_labels[test_labels[n]]
pred_labl = class_labels[predictions[n]]
plt.imshow(image[:, :, 0], cmap='gray')
plt.title("Original label is:"+orig_labl+" Predicted is: " + pred_labl)
plt.show()
