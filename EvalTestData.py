from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Load the model
model = load_model(r'C:\Users\nourm\OneDrive\Desktop\PROJECTS\UNITY MODEL PROJECT\CodePython\Emotion_model_16.h5')

test_data_path = r'C:\Users\nourm\OneDrive\Desktop\PROJECTS\UNITY MODEL PROJECT\FER-2013 (USED)\test'

# STEP 1: Load the data using ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)  # Normalize images

input_height, input_width = model.input_shape[1:3]

test_data = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(input_height, input_width),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',  
    shuffle=False
)

print(f"Found {test_data.samples} test images belonging to {test_data.num_classes} classes")

results = model.evaluate(test_data)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")

# h=history.history
# h.keys

# import matplotlib as plt
# plt.plot(h["accuracy"],c="yellow")
# plt.show