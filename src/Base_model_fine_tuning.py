####################################################################
############ Import libraries ######################################
####################################################################
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from keras_tuner import HyperModel, RandomSearch
####################################################################
############# Explore classes ######################################
####################################################################
""" 
The code in this section was written by Mr Oucheikh for exercises
in the GeoAI course. The code is slightly adjusted by the author 
of this python script. 
"""
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


# Define image size and batch size
image_size = (256, 256)  
batch_size = 32
#####################################################################
################ Load data ##########################################
#####################################################################
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
                # Images prints in BLUE!! Change to RGB. 
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

    # Convert labels to one-hot encoding, in other words from strings to vector
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    # Gets problem when computing evalutation metrics and confusion matrix
    # when converting y_test to one-hot encoding. Error says y_pred and y_test
    # have different number of labels. Remove and it works. 
    #y_test = to_categorical(y_test, num_classes=num_classes)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Save test and train data in variables 
X_train, X_val, X_test, y_train, y_val, y_test = load_data()

################################################################
######################### Build model ##########################
################################################################
# Calculate class weight since dataset is slightly skewed with more samples for Alluvial soil (Tip from Mr Oucheikh!)
# Since y_train is one hot encoded and the y argument needs the original label number, we need to 
# transform these: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html 
# Since one hot encoded is a binary form av the class label where the class label is equal to the indices where 1 is 
# located ammong the zeros in the list, we can use keras argmax and enter axis=1 to retrieve that indice. 
# https://stackoverflow.com/questions/47435526/what-is-the-meaning-of-axis-1-in-keras-argmax 
# Enter "balanced" to assign a balanced weight for each class. 
class_weights = compute_class_weight("balanced", classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
# Save class weigths as a dictionary with class as key
class_weights = dict(enumerate(class_weights))

# Define the hypermodell. This code was created with keras guide as a template
# https://www.tensorflow.org/tutorials/keras/keras_tuner 
def build_hypermodel(hp):
    # Use VGG16 as base model (Tip from MR Oucheikh!): https://keras.io/api/applications/vgg/ 
    # The VGG16 is trained on imagenet if choosen, found this github repository as template:
    # https://github.com/trzy/VGG16/blob/master/VGG16/__main__.py 
    # Do not include the three top layers since the model has some overfit 
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    # When using the model the weights is updated. Since VGG16 is pre trained, "freeze" the layer
    # so it cannot be trained and the weights does not change during training.
    # https://www.tensorflow.org/tutorials/images/transfer_learning 
    # https://stackoverflow.com/questions/49112941/keras-freezing-a-model-and-then-adding-trainable-layers
    base_model.trainable = False  
    # Build trainable model
    model = Sequential([
        # VGG16 base model
        base_model,
        # Add flatten layer to flatten output and prepare for fully connected layers
        # Flatten reforms the input to a one-dimensional array which is needed for 
        # conectivity compatiblity. 
        # https://www.geeksforgeeks.org/what-is-the-role-of-flatten-in-keras/
        Flatten(),
        # Dense layers require one dimensional input and thus do not use
        # spatial information. This layers capture abstract details in the 
        # information provided during the use of spatial convolutional layers.
        # https://wandb.ai/ayush-thakur/keras-dense/reports/Keras-Dense-Layer-How-to-Use-It-Correctly--Vmlldzo0MjAzNDY1
        Dense(
            # Tune the number of units in the first dense layer 
            # https://www.tensorflow.org/tutorials/keras/keras_tuner 
            units=hp.Int("units", min_value=128, max_value=512, step=64),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # Tip from Mr Oucheikh on overfit! 
        # Normalize before output so mean output stays close to 0 and SD close to 1:
        # https://keras.io/api/layers/normalization_layers/batch_normalization/
        BatchNormalization(),
        # Add dropout layer to drop out unused neurons in the Dense layers
        # https://www.baeldung.com/cs/ml-relu-dropout-layers
        # Test different dropout rates: https://www.tensorflow.org/tutorials/keras/keras_tuner 
        Dropout(hp.Float("dropout", min_value=0.3, max_value=0.7, step=0.1)),
        # Four classes 
        Dense(4, activation="softmax")])
    # Minimize the loss function: https://keras.io/api/optimizers/rmsprop/ 
    # https://www.youtube.com/watch?v=JhQqquVeCE0 
    # Try different learning rates: https://www.tensorflow.org/tutorials/keras/keras_tuner 
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-3, sampling="LOG")
    # During building of architecture, RMSprop showed most promesing results
    optimizer = RMSprop(learning_rate=learning_rate)
    # Since our classification is categorical with multiple classes
    # use categorical crossentropy instead of binary as we did in ANN. 
    # This compares two probability matrices to increase accuracy
    # https://www.geeksforgeeks.org/categorical-cross-entropy-in-multi-class-classification/ 
    # The resulting metric is categorical accuracy, not accuracy
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


# Create tuner: https://keras.io/keras_tuner/getting_started/ 
# Use random to save some processing time. (Code was running 18 hours before changing to random)
tuner = RandomSearch(
    # Call the modell building 
    build_hypermodel,
    # Maximize validation accuracy 
    objective="val_categorical_accuracy",
    # Test 10 different combinations
    max_trials=10,
    # Test each combination once 
    executions_per_trial=1,
    directory="CNN_soil_GeoAI_Proj",
    project_name="CNN_soil_transfer_learning_tuner_VGG16")
# Since many epochs is to be used, early stopping is used to minimze the number of epochs
# and avoiding overfit at the end of the epochs. The best weights is restored for every epoch:
# https://keras.io/api/callbacks/early_stopping/
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
# Search for best combination: https://keras.io/keras_tuner/getting_started/ 
# Use maximum 20 epochs for every trial. Maybee I should have used more since using early stopping but the computer
# is warning about overheating cores. Therefore a limit is needed. 
tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[early_stopping])
# Retrieve the best model 
best_model = tuner.get_best_models(num_models=1)[0]

# Fine tuning the VGG16 base model to achieve slightly higher accruacy 
# ONLY FIRST FOUR LAYERS IN ORDER TO NOT OVERFIT!!!
# Chapter "Fine tuning" in keras guide for transfer learning:
# https://www.tensorflow.org/guide/keras/transfer_learning
for layer in best_model.layers[0].layers[-4:]:  
    layer.trainable = True

# Re-compile the model with very low learning rate
# Watch out for overfit during the fine tuning, keras guide is very clear about this
# https://www.tensorflow.org/guide/keras/transfer_learning 
optimizer = RMSprop(learning_rate=0.000001)  
best_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])

# Low number of epochs and early stop to prevent overfit! 
history_finetune = best_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[early_stopping])

# Could not access the training history in the saved model, 
# save separaelty to be able to print later in process if needed
import json
with open("Transfer_Learning_vgg16_SOIL_optimized_final_history.json", "w") as f:
    json.dump(history_finetune.history, f)

# Save best model
best_model.save("Transfer_Learning_vgg16_SOIL_optimized_final.keras")

################################################################
######################## Validation ##########################
################################################################
# Retrieve the training loss and accuracy from the history object
train_loss = history_finetune.history["loss"]
val_loss = history_finetune.history["val_loss"]
train_acc = history_finetune.history["categorical_accuracy"]
val_acc = history_finetune.history["val_categorical_accuracy"]
# Plot the training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training and Validation Loss")
plt.legend()
# Plot the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Training and Validation Accuracy")
plt.legend()
# Show the plots
plt.tight_layout()
plt.show()

#####################################################################
######## Prediction and evaluation ##################################
#####################################################################
# Save prediction in variable
prediction = best_model.predict(X_test) 
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
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")


















