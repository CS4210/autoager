import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from PIL import UnidentifiedImageError
import numpy as np

# Define paths
accident_images_path = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\Accident Images"
processed_images_path = r"C:\Users\solar\OneDrive\Documents\GitHub\autoager\Processed_Accident_Images"

# Create the processed images folder if it doesn’t already exist
os.makedirs(processed_images_path, exist_ok=True)

# Define image dimensions
img_width, img_height = 224, 224

# Function to preprocess and save an image
def preprocess_and_save_image(image_path, save_dir):
    try:
        # Load, resize, and normalize the image
        img = load_img(image_path, target_size=(img_width, img_height))
        img_array = img_to_array(img) / 255.0  # Normalize to 0-1 range

        # Convert the array back to an image and save
        processed_img = array_to_img(img_array)
        image_name = os.path.basename(image_path)  # Get the original filename
        processed_img.save(os.path.join(save_dir, f"processed_{image_name}"))
        print(f"Saved processed image: {image_name}")

    except UnidentifiedImageError:
        print(f"Skipped: '{image_path}' is not a valid image.")
    except Exception as e:
        print(f"Error processing '{image_path}': {e}")

# Process and save each image in the Accident Images folder
for image_file in os.listdir(accident_images_path):
    image_path = os.path.join(accident_images_path, image_file)

    # Check if the path is a file and has a valid image extension
    if os.path.isfile(image_path) and image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        preprocess_and_save_image(image_path, processed_images_path)
