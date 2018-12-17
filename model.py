import os
import cv2
import numpy as np
import sklearn
import random
import utils
from keras import layers
from keras.models import Sequential
import matplotlib
import keras
#matplotlib.use("agg")
import matplotlib.pyplot as plt
from keras import backend as K

IMAGE_MEAN, IMAGE_STD = utils.load_image_stats()

def normalize_img(image):
    #return (image - 0.5)*2
    return (image - IMAGE_MEAN) / (IMAGE_STD+1e-8)

def denormalize_img(image):
    return (image * IMAGE_STD) - IMAGE_MEAN

def generator(samples, batch_size=1, augment_data=True):
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            y_train = []
            for batch_sample in batch_samples:
                name = batch_sample[0].split('/')[-1]
                correction = 0
                if augment_data and random.random() > 0.75: # Augment left / right camera
                    if random.random() > 0.5: # left
                        name = batch_sample[1].split('/')[-1]
                        correction = 0.2
                    else: # right
                        name = batch_sample[2].split('/')[-1]
                        correction = -0.2
                
                filepath = os.path.join("/opt/test/drive2", "IMG", name)
                assert os.path.isfile(filepath), "Is not path: {}".format(filepath)
                center_image = utils.read_img(filepath)
                center_angle = float(batch_sample[3]) # Steering
                center_angle += correction
                throttle = float(batch_sample[4])
                brake = float(batch_sample[5])
                speed = float(batch_sample[6])
                if augment_data and random.random() > 0.5:
                    center_image = np.fliplr(center_image)
                    center_angle = - center_angle
                
                images.append(center_image)
                y_train.append([center_angle, throttle])
            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(y_train)
            y_train = y_train[:, 0]
            #X_train = (X_train - mean) / std
            # Reshape from BGR to RGB
            #print(X_train.shape, IMAGE_MEAN.shape)
            X_train = normalize_img(X_train)
            #X_train = crop_images(X_train)
            #print(X_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)


def add_conv2d(model, filters, max_pool=False):
    model.add(layers.Conv2D(filters, 3, padding="same", activation="relu", data_format="channels_last"))
    if max_pool:
        model.add(layers.MaxPool2D(2, strides=[2,2]))
    model.add(layers.BatchNormalization())

def create_model():
    ch, row, col = 3, 160, 320  # Trimmed image format
    top_crop = row * 8 // 20
    print(top_crop)
    model = Sequential()
    local = False
    # Preprocess incoming data, centered around zero with small standard deviation 
    #model.add(layers.Dropout(0, input_shape=[160, 320, 3]))
    model.add(layers.Cropping2D(cropping=((50,20), (0,0)), input_shape=[row, col, ch]))
    #model.add(layers.Lambda(lambda x: normalize_data(x)))
    if not local:
        add_conv2d(model, 32, True)
        add_conv2d(model, 32)
        add_conv2d(model, 32)
        add_conv2d(model, 32, True)
        add_conv2d(model, 64)
        add_conv2d(model, 64)
        add_conv2d(model, 64, True)
        add_conv2d(model, 128)
        add_conv2d(model, 128)
        add_conv2d(model, 128, True)
        add_conv2d(model, 256)
        model.add(layers.MaxPool2D(2, strides=[1,2]))
        add_conv2d(model, 256)
        add_conv2d(model, 256, True)
        add_conv2d(model, 512, True)

    #model.add(... finish defining the rest of your model architecture here ...)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
    model.summary()
    return model
    
    
if __name__ == "__main__":
    samples = utils.read_csv_file('data/driving_log.csv')
    samples += utils.read_csv_file("/opt/test/drive1/driving_log.csv")
    samples += utils.read_csv_file("/opt/test/drive2/driving_log.csv")
    samples += utils.read_csv_file("/opt/test/drive3/driving_log.csv")
    samples += utils.read_csv_file("/opt/test/drive4/driving_log.csv")

    samples = samples[1:] # remove header line

    
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.1)


    #for i in generator(train_samples):
    #    print(i[0].shape)

    batch_size = 32
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size, augment_data=False)
    #imgs = next(train_generator)[0]
    #print(imgs.min(), imgs.max)
    #imgs2 = imgs - imgs.min(axis=0)
    #imgs2 = imgs2 / imgs2.max(axis=0)
    #plt.imshow(imgs2[0])
    #plt.show()
    model = create_model()
    #imgs = model.predict(imgs)
    #plt.imshow(denormalize_img( imgs[0]))
    #plt.show()
    model.load_weights("my_model_weights.h5")
    """
    model.fit_generator(train_generator, 
                        samples_per_epoch=len(train_samples),
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples),
                        nb_epoch=3)

    """
    print("Train samples:", len(train_samples))
    #If the above code throw exceptions, try 

    model.fit_generator(train_generator, steps_per_epoch= len(train_samples)//batch_size,
    validation_data=validation_generator, validation_steps=len(validation_samples)//batch_size, epochs=40, verbose = 1)
    model.save("my_model.h5")
    model.save_weights("my_model_weights.h5")