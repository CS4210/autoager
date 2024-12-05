import tensorflow as tf
import keras
import pandas as pd
import os

# ---Model Architecture-------
# Predict a car's age (continuous value) => regression 
input_shape = (224, 224, 3) # image size and RGB (3 channels)
# Define LeNet architecture as the base model
model = keras.models.Sequential([
    keras.layers.Input(shape=input_shape),
    keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu', padding='same'),
    keras.layers.AvgPool2D(pool_size=2, strides=2),
    keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu'),
    keras.layers.AvgPool2D(pool_size=2, strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    # Output layer for regression (might range from 0 to 100 years old or beyond)
    keras.layers.Dense(1, activation="linear")]) # linear activation for regression task

# Compile the model with an appropriate loss function for regression (MSE)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae']) 
#---------------


# Machine Learning Engineer: Read and train data using 'model.fit()' 
'''train_df = pd.read_csv("CSV/Car_Train.csv")

train_data = keras.utils.image_dataset_from_directory(
    image_size=(224, 224),
    # batch_size=32,
)
'''