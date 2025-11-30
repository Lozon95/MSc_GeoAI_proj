import os
import shutil
import random
import numpy as np
import pandas as pd
import PIL, cv2
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import visualkeras
import json

#####################################################################
################ Load data ##########################################
#####################################################################
dataset_dir = "C:\\Users\\ander\\GeoAI_NGEN27\\Project_Soil_CNN\\Dataset\\Train"
image_size = (256, 256)  
batch_size = 32
def load_data():
    # Initialize empty lists to store images and their corresponding labels
    images = []
    labels = []

    # Loop through each class folder
    class_folders = os.listdir(dataset_dir)
    class_folders.sort()  # Sort to ensure consistent class ordering
    num_classes = len(class_folders)

    for class_index, class_folder in enumerate(class_folders):
        class_path = os.path.join(dataset_dir, class_folder)
        
        # Loop through images in the class folder
        for image_filename in os.listdir(class_path):
            image_path = os.path.join(class_path, image_filename)
            
            # Load and preprocess the image
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, image_size)
                image = image.astype("float32") / 255.0  # Normalize pixel values to [0, 1]
            
                # Append the image and label to the respective lists
                images.append(image)
                labels.append(class_index)
            else:
                print(f"Image is not loaded {image_path}")

    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split the dataset into training and val sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.3, random_state=42)
    # Split again the val data to validation and test data in half (which makes them 15% of the entire data)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    # Convert labels to one-hot encoding, in other words from strings to vectors
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    # Gets problem when computing evalutation metrics and confusion matrix
    # when converting y_test to one-hot encoding. Error says y_pred and y_test
    # have different number of labels. Remove and it works. 
    #y_test = to_categorical(y_test, num_classes=num_classes)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Save test and train data in variables 
X_train, X_val, X_test, y_train, y_val, y_test = load_data()

#####################################################################
################ Load model #########################################
#####################################################################
# Load model
model = load_model("Transfer_Learning_vgg16_SOIL_optimized_final.keras")

# Save model architecture as image
plot_model(model, to_file="model_architecture_tuned.png", show_shapes=True, show_layer_names=True)
# Another model image to evaluate 
visualkeras.layered_view(model, to_file="model_architecture_2_tuned.png", legend=True).show()

################################################################
######################## Validation ############################
################################################################
# Load training history for the model
with open("TLVS_2_training_history.json", "r") as f:
    history = json.load(f)

# Retrieve the training loss and accuracy from the history object
train_loss = history["loss"]
val_loss = history["val_loss"]
train_acc = history["categorical_accuracy"]
val_acc = history["val_categorical_accuracy"]
# Plot the training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
# Plot the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
# Show the plots
plt.tight_layout()
plt.show()

#####################################################################
######## Prediction and evaluation ##################################
#####################################################################
# Predict test data
prediction = model.predict(X_test) 
# Convert the probabilites to class labels, I used np.round for
# binary classification with ANN but since this task is multi class
# we need to use argmax to find the class with the largest predicted
# probability. Use axis = 1 to acces columns in the array
# https://machinelearningmastery.com/argmax-in-machine-learning/ 
pred_labels = np.argmax(prediction, axis=1) 
# Calculate precision, accuracy, recall and f1
# Use weighted average since there is a slight inbalance in number
# of examples in every class.
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
precision = precision_score(y_test, pred_labels, average="weighted")
accuracy = accuracy_score(y_test, pred_labels)
recall = recall_score(y_test, pred_labels, average="weighted")
f1 = f1_score(y_test, pred_labels, average="weighted")
# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

####################################################################
############ Confusion Matrix ######################################
####################################################################
# Compute the confusion matrix
confusion = confusion_matrix(y_test, pred_labels)
# Create a confusion matrix for report
plt.figure(figsize=(8, 6))  
# Add fmt = d to format the label in the heatmap
sns.heatmap(confusion, annot=True, fmt="d", cmap="Reds")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
# Adjust plot borders to the labels
plt.tight_layout()
# List class names i ascending order to match with confusion matrix
class_labels = sorted(os.listdir(dataset_dir))
# Adjust labels to be placed at centre height of confusion matrix cells (+0.5)
# and rotate for easier intepretation and better visual looks
plt.xticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=45, ha="right")
plt.yticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=0)
plt.tight_layout()
plt.show()

####################################################################
############ Classify 5 images #####################################
####################################################################
# Load images in the test folder in same size as modell been trained on
def load_and_prepare_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    # Since normalized during training, normalize also this
    image = image.astype("float32") / 255.0 
    image = np.expand_dims(image, axis=0)
    return image

# Predictions funciton
def predict_image_class(model, image_path):
    image = load_and_prepare_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

# Search path to test folder
dataset_dir = "C:\\Users\\ander\\GeoAI_NGEN27\\Project_Soil_CNN\\Dataset\\test"

# Modell classes 
class_labels = ["Alluvial soil", "Black Soil", "Clay soil", "Red soil"]

# For storing predictions
predictions = {}

# Same procedure as in affine file
for subdir, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(subdir, file)
            true_class = os.path.basename(subdir)
            predicted_class = predict_image_class(model, file_path)
            # Save the prediction
            predictions[file_path] = (true_class, class_labels[predicted_class])

# Plotting template to fit all images in the same plot
plt.figure(figsize=(15, 10))

# Select 25 random images for plotting
sample_images = np.random.choice(list(predictions.keys()), 25, replace=False)

# Iterate through the sample images and plot them
for i, file_path in enumerate(sample_images):
    img = cv2.imread(file_path)
    # Images in blue, convert
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Retrieve predicted and true class
    true_class, predicted_class = predictions[file_path]
    plt.subplot(5, 5, i+1)
    plt.imshow(img)
    plt.title(f"True class: {true_class}\nPredicted class: {predicted_class}")
    plt.axis('off')

plt.tight_layout()
plt.show()

