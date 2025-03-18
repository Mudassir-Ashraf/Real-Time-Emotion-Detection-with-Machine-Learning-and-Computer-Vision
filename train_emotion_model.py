import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Set paths to the train and test directories
train_dir = os.path.join('data', 'train')
test_dir = os.path.join('data', 'test')
"""
Define the file paths for the training and testing datasets.
The datasets are expected to be organized in subdirectories under 'data/train' and 'data/test'.
"""

# 2. Create an ImageDataGenerator for augmenting the training data
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,      # Rescale pixel values to the range [0, 1]
    rotation_range=10,      # Randomly rotate images in the range of 10 degrees
    width_shift_range=0.1,  # Randomly shift images horizontally by 10% of the width
    height_shift_range=0.1, # Randomly shift images vertically by 10% of the height
    zoom_range=0.1,         # Randomly zoom images by up to 10%
    horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode='nearest'     # Fill in missing pixels after transformation
)
"""
Create an instance of ImageDataGenerator for the training dataset.
This will perform real-time data augmentation to improve model generalization.
"""

# 3. Create an ImageDataGenerator for the test data (only rescale the pixel values)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
"""
Create an instance of ImageDataGenerator for the testing dataset.
In this case, only rescaling is performed since no augmentation is needed for validation.
"""

# 4. Load the images from the train directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),   # Resize images to 48x48 pixels
    batch_size=64,          # Number of images to yield from the generator per batch
    color_mode='grayscale',  # Convert the images to grayscale
    class_mode='categorical' # Use categorical labels for multi-class classification
)
"""
Load the training images using the flow_from_directory method.
This method will automatically label the images based on their respective subdirectory names.
"""

# 5. Load the images from the test directory
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),   # Resize images to 48x48 pixels
    batch_size=64,
    color_mode='grayscale',  # Convert the images to grayscale
    class_mode='categorical'  # Use categorical labels for multi-class classification
)
"""
Load the testing images in a similar manner as the training images.
The test generator will yield batches of test images and their corresponding labels.
"""

# 6. Define the CNN model for emotion detection
def create_emotion_model():
    model = Sequential()  # Initialize a Sequential model
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))  # First convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))  # First max pooling layer
    
    model.add(Conv2D(128, (3, 3), activation='relu'))  # Second convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Second max pooling layer

    model.add(Flatten())  # Flatten the output from the previous layer
    model.add(Dense(256, activation='relu'))  # Fully connected layer with 256 neurons
    model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
    model.add(Dense(7, activation='softmax'))  # Output layer for 7 classes (emotions)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    """
    Compile the model specifying the optimizer, loss function, and metrics to track during training.
    """
    return model

# 7. Create and compile the model
model = create_emotion_model()
"""
Instantiate the model by calling the create_emotion_model function.
The model is now ready to be trained.
"""

# 8. Train the model using the training generator
history = model.fit(
    train_generator,
    epochs=50,  # Train for 50 epochs
    validation_data=test_generator  # Validate using the test generator
)
"""
Train the model using the training data generator.
The training process will run for 50 epochs and evaluate the model's performance on the test dataset at the end of each epoch.
"""

# 9. Save the trained model for future use
model.save('emotion_detection_model.h5')
"""
Save the trained model to a file named 'emotion_detection_model.h5'.
This allows for later use without needing to retrain the model.
"""