####################################################################
############ Import libraries ######################################
####################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

####################################################################
############# Explore classes ######################################
####################################################################
dataset_dir = "C:\\Users\\ander\\GeoAI_NGEN27\\Project_Soil_CNN\\Dataset\\Train"
# List and print labels (name of each folder in directory)
LABELS = os.listdir(dataset_dir)
print(LABELS)
# plot class distributions of whole dataset
# Count the number of picturs in each class (folder) 
counts = {}
for l in LABELS:
    counts[l] = len(os.listdir(os.path.join(dataset_dir, l)))
# Plot a diagrame showing the distribution of each class
plt.figure(figsize=(12, 6))
# Use barplot fo visualisation 
plt.bar(range(len(counts)), list(counts.values()), align="center")
plt.xticks(range(len(counts)), list(counts.keys()), fontsize=12, rotation=40)
plt.xlabel("class label", fontsize=13)
plt.ylabel("class size", fontsize=13)
plt.title("Soil Class Distribution", fontsize=15)
plt.tight_layout()
plt.show()

#####################################################################
################ Resize ##########################################
#####################################################################
"""
This section resizes images in the directory with subfolders of the image
classes. To not get wierd patterns in the resized images with interpolation,
a padding of only zeros are added to the image if needed.

The code was created with the help of:
https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec 

For future note: If the images to be padded and resized are of the same
size, modules such as Pillow can be used. However in this case the images
to be resized differ alot in original size and therefore a dynamic approach
is needed. 
"""

def resize_with_padding(image, target_size):
    # old_size is in (height, width) format
    old_size = image.shape[:2]  
    # Calculate the ratio between the target size and old size
    # To use so the image is unchanged inside the padding 
    ratio = float(target_size) / max(old_size)
    # Scale to new size by multiplying the old widht and height
    # with the ratio between old and target size. Dim must be integers 
    new_size = tuple([int(x * ratio) for x in old_size])
    # Resize the image with the new size
    # cv2 rezise takes (width, height) instead of (height, width)
    resized_image = cv2.resize(image, (new_size[1], new_size[0]))
    # Create a new image with the target size and fill it with a padding color 
    # 0,0,0 is black for every chanel. Speicfy datatype.  
    new_image = np.full((target_size, target_size, 3), (0, 0, 0), dtype=np.uint8)
    # Place the resized_image at the center of new_image. Each part in the list
    # gives the horizontal and vertical pixeldistance in order to centre the image
    new_image[(target_size - new_size[0]) // 2 : (target_size - new_size[0]) // 2 + new_size[0],
              (target_size - new_size[1]) // 2 : (target_size - new_size[1]) // 2 + new_size[1]] = resized_image

    return new_image

def resize_images_in_directory(directory, target_size):
    # Loop trough the directory and the subdirectorys
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            # Different files was used, bulletproof to not miss any
            if file.endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                file_path = os.path.join(subdir, file)
                image = cv2.imread(file_path)
                # Resize the image with padding
                resized_image = resize_with_padding(image, target_size[0])
                cv2.imwrite(file_path, resized_image)
                print(f"Resized and saved: {file_path}")
# Target size = 256 to keep the whole image and relations intact
resize_images_in_directory(dataset_dir, target_size=(256, 256))

#####################################################################
################# Affine transformation #############################
#####################################################################
"""
This section duplicates images and applies an affine transformation.
This is done to increase the data input and even out differences in 
datasizes between clases. 

The code was created with the help of:
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator 
"""
# Set affine parametres
datagen = ImageDataGenerator(
    rotation_range=90,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="constant",
    cval = 0
)

# Duplicate every picture with affine transformation
def augment_images(directory):
    # List all images in directory
    image_files = [f for f in os.listdir(directory) if f.endswith((".png", ".jpg", ".jpeg"))]
    # Count how many images the folder contatins 
    num_images = len(image_files)
    # Duplicated only if folder contains less than 300 images 
    if num_images < 300:
        print(f"Folder: {directory} has fewer than 300 images. Performing affine transformation to double the number of images...")
        # Iterate trough all images 
        for image_file in image_files:
            # Build image path
            image_path = os.path.join(directory, image_file)
            # Load image 
            image = load_img(image_path)
            # Convert to array
            x = img_to_array(image)
            # Array needs a batch dimension to be compatible with affine
            x = x.reshape((1,) + x.shape)
            
            # Generate a new image for every image with the affine parametres
            for batch in datagen.flow(x, batch_size=1, save_to_dir=directory, save_prefix="aug2", save_format="jpg"):
                break 
        print(f"Affine transformation complete for folder: {directory}.")
    elif num_images >= 500:
        print(f"Folder: {directory} has more than 500 images.")


# Iterate trough dataset directory (image folders)
for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    if os.path.isdir(folder_path):
        # Use affine function
        augment_images(folder_path)

print("Augmentation completed.")




