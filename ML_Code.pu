# Part 1 : Importing Libraries 
import numpy as np
import matplotlib.pyplot as plt 
import os
import cv2
from PIL import Image
import tensorflow as tf
import keras

# Part 2 : Initializing and Preparing data

# define all paths and directories to refer to images
image_dir = '\\Users\\Haadi\\Documents\\UCUSCIMATL2\\Project\\cell_images\\' # change for each user
parasitized_cells_dir = image_dir + 'Parasitized\\'
uninfected_cells_dir = image_dir + 'Uninfected\\'

# initalize dataset and label lists which will be used to train the model 
dataset = []
label = []

size_x = 64
size_y = 64

parasitized_cells=os.listdir(parasitized_cells_dir)

for i in parasitized_cells: #iterate through the parasitized cells folder 
    image = cv2.imread(parasitized_cells_dir+ i) #read image 
    image = Image.fromarray(image, 'RGB') #convert image into a numpy array
    image = image.resize((size_x, size_y)) #resize the image to 64 by 64
    dataset.append(np.array(image)) #save image array to dataset array
    label.append(0) #assign label of 0 to image
        
# repeat process for uninfected cells folder, assigning 1 to the images
uninfected_cells=os.listdir(uninfected_cells_dir)

for i in uninfected_cells:
    image = cv2.imread(uninfected_cells_dir + i)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((size_x, size_y))
    dataset.append(np.array(image))
    label.append(1)
    
# Part 3 : Plotting the sample images

def PlotRandomImages(n,path,size_x,size_y):
    """
    This function plots a number n of random images from a folder in a location given by the 
    argument path, that has to be in string form. The images are resizes to size_x and size_y
    before being plotted.
    """
    data = os.listdir(path) # creates a list with the names of the entries in the directory given by path
    sample_images= np.random.choice (data,n) # selects n random entries from the list
    fig, axs = plt.subplots(1,n,figsize=(15,3)) # creates a row of n columns to plot the images
    for i in range(0, n): # for loop that repeats the plotting process for each of the n images
        image = cv2.imread(path + sample_images[i]) # reads each of the n images
        image = Image.fromarray(image, 'RGB')  # convers the array to an image which contains RGB values
        axs[i].imshow(image) 
        plt.suptitle(t='Sample of ' + str(n) + ' random images retrieved from '+ path, size=15)
    plt.show()

PlotRandomImages(n=4, path=uninfected_cells_dir, size_x=64, size_y=64)
PlotRandomImages(n=4, path=parasitized_cells_dir, size_x=64, size_y=64)
    
# Part 4:  
    
# Below the model is defined with four convolutional section and a fifth section to define output

model = keras.Sequential(
    [
    keras.layers.Convolution2D(32, (3, 3), padding = 'same', input_shape = (size_x, size_y, 3), kernel_initializer='he_normal', activation = 'relu', data_format = 'channels_last'),  
	keras.layers.MaxPooling2D(pool_size = (2, 2), data_format="channels_last"), 
    keras.layers.BatchNormalization(axis = -1),
    keras.layers.Dropout(0.2),
    keras.layers.Convolution2D(32, (3, 3), padding = 'same', kernel_initializer='he_normal', activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2, 2), data_format="channels_last"),
    keras.layers.BatchNormalization(axis = -1),
    keras.layers.Dropout(0.2),
    keras.layers.Convolution2D(32, (3, 3), padding = 'same', kernel_initializer='he_normal', activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2, 2), data_format="channels_last"),
    keras.layers.BatchNormalization(axis = -1),
    keras.layers.Dropout(0.2),
    keras.layers.Convolution2D(32, (3, 3), padding = 'same', kernel_initializer='he_normal', activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2, 2), data_format="channels_last"),
    keras.layers.BatchNormalization(axis = -1),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(activation = 'relu', units=512),
    keras.layers.BatchNormalization(axis = -1),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(activation = 'relu', units=256),
    keras.layers.BatchNormalization(axis = -1),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(activation = 'sigmoid', units=2),
    ]
    )
    
#here the model is compiled using the adam optimizer and a binary crossentropy function which values each training cycle via the metric of accuracy
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(model.summary())

# Part 5: Model Training 

from sklearn.model_selection import train_test_split

#the training and test datasets are defined with a 85:15 percentage split 
x_train, x_test, y_train, y_test = train_test_split(dataset,keras.utils.to_categorical(np.array(label)), train_size=0.85, random_state=3)

#an early stopping variable is created to interrupt model when no significant improvement is made after 5 consistent epochs
earlystop = [keras.callbacks.EarlyStopping(patience = 5, monitor='val_loss')]

#training data is fit into model and assigned to a variable as to record its history
ht = model.fit(np.array(x_train), 
                         y_train, 
                         verbose = 1,     #displays a progress bar for each epoch
                         epochs = 50,      #changed to 5 from 50 for testing purposes.
                         batch_size = size_x, # 64 samples per batch of computation
                         validation_split = 0.1, # 10% of the training data to be used as validation data
                         shuffle = False, #data not shuffled between each epoch
                         callbacks= earlystop #inital earlystop callback passed
                     )

# Part 6: Model Asessment

#net accuracy of model evaluated and printed as a percentage
print("Net_Accuracy: {:.2f}%".format(model.evaluate(np.array(x_test), np.array(y_test))[1]*100))    
    
#model accuracy and validation accuracy functions plotted 
plt.plot(ht.history['accuracy'], label='$Train \ Accuracy$' )
plt.plot(ht.history['val_accuracy'], label='$Validation \ Accuracy$')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

#model losses and validation losses functions plotted  
plt.plot(ht.history['loss'], label='$Training \ Loss$')
plt.plot(ht.history['val_loss'], label='$Validation \ Loss$')
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.legend()
plt.show()hjgu
