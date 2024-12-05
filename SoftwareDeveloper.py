import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class DataLoader:
    def __init__(self, image_dir, csv_path, max_samples=100):
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.max_samples = max_samples

    def load_data(self):
        data = pd.read_csv(self.csv_path)
        # Only use a subset for testing
        data = data.head(self.max_samples)
        
        images, labels = [], []
        for _, row in data.iterrows():
            img_path = os.path.join(self.image_dir, row['Pictures'])
            if os.path.exists(img_path):
                image = cv2.imread(img_path)
                image = cv2.resize(image, (224, 224))  # Resize to 224x224
                images.append(image)
                # Convert year to the car's age
                labels.append(2024 - row['Year'])  # Assuming the current year is 2024
        
        images = np.array(images)
        labels = np.array(labels)
        
        # Normalize image data
        images = images / 255.0  # Normalize pixel values to [0, 1]
        
        return images, labels


def build_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1)  # Output layer (age prediction)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


# Directories
train_dir = "Resized Train Images"
test_dir = "Resized Test Images"
train_csv = "CSV/Car_Train.csv"
test_csv = "CSV/Car_Test.csv"

# Load data (only 100 samples for testing)
train_loader = DataLoader(train_dir, train_csv, max_samples=100)
X_train, y_train = train_loader.load_data()

test_loader = DataLoader(test_dir, test_csv, max_samples=100)
X_test, y_test = test_loader.load_data()

# Split the train set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build and train the model
model = build_model()

# Fit the model on the training data
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=16)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
