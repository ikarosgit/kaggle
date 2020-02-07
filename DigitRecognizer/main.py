import os
import argparse

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import numpy as np

def load_data(args):

    train_path = os.path.join(args.data_dir, "train.csv")
    test_path = os.path.join(args.data_dir, "test.csv")
    assert os.path.exists(train_path), train_path
    assert os.path.exists(test_path), test_path
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_labels = train_df.iloc[:, 0].values.astype("int32")
    train_images = train_df.iloc[:, 1:].values.astype("float32")
    test_images = test_df.values.astype("float32")

    train_labels = np_utils.to_categorical(train_labels)
    
    return train_images, train_labels, test_images

def preprocess(train_images, test_images):
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)
    train_images /= 255.0
    test_images /= 255.0
    return train_images, test_images

class Model:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.create()

    def create(self):
        self.network = Sequential()

        self.network.add(Conv2D(
            filters=32, kernel_size=(5, 5), padding="Same",
            activation="relu", input_shape=self.input_shape))
        self.network.add(Conv2D(
            filters=32, kernel_size=(5, 5), padding="Same",
            activation="relu"))

        self.network.add(MaxPool2D(pool_size=(2, 2)))
        self.network.add(Dropout(0.25))

        self.network.add(Conv2D(
            filters=64, kernel_size=(3, 3), padding="Same",
            activation="relu"))
        self.network.add(Conv2D(
            filters=64, kernel_size=(3, 3), padding="Same",
            activation="relu"))

        self.network.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.network.add(Dropout(0.25))

        self.network.add(Flatten())
        self.network.add(Dense(256, activation="relu"))
        self.network.add(Dropout(0.5))
        self.network.add(Dense(self.num_classes, activation="softmax"))

def write_predictions(preds, out_file):
    df = pd.DataFrame({"ImageId": list(range(1, len(preds)+1)), "Label": preds})
    df.to_csv(out_file, index=False, header=True)

def main(args):

    train_images, train_labels, test_images = load_data(args)
    print(train_images.shape, train_labels.shape, test_images.shape)

    train_images, test_images = preprocess(train_images, test_images)
   
    train_images, valid_images, train_labels, valid_labels = train_test_split(
        train_images, train_labels, test_size=0.1, random_state=args.random_seed)

    input_shape = train_images.shape[1:]
    num_classes = train_labels.shape[1]

    model = Model(input_shape, num_classes)

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.network.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"])
   
    learning_rate_reduction = ReduceLROnPlateau(
        monitor="val_accuracy",
        patience=3,
        verbose=1,
        factor=0.5,
        min_lr=0.00001)

    data_generator = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False)

    data_generator.fit(train_images)

    history = model.network.fit_generator(
            data_generator.flow(train_images, train_labels, batch_size=args.batch_size),
            epochs=args.num_epochs,
            validation_data=(valid_images, valid_labels),
            verbose=2,
            steps_per_epoch=train_images.shape[0] // args.batch_size,
            callbacks=[learning_rate_reduction])

    predictions = model.network.predict_classes(test_images, verbose=0)

    write_predictions(predictions, args.out_file)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/ikaros/hdd/data/kaggle/DigitRecognizer")
    parser.add_argument("--out_file", type=str, default="output.csv")
    parser.add_argument("--random_seed", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=86)
    parser.add_argument("--num_epochs", type=int, default=10)
    
    args = parser.parse_args()
    main(args)
