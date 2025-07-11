import tensorflow as tf
import numpy as np
import time
import os

def test_gpu_setup():
    print("=" * 60)
    print("GPU Setup Test for Unity Model Project")
    print("=" * 60)
    
    # Check GPU availability
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
    print(f"GPU count: {len(tf.config.list_physical_devices('GPU'))}")
    
    if tf.config.list_physical_devices('GPU'):
        print("‚úÖ GPU is available for Unity ML models!")
        
        # Test GPU computation
        print("\nTesting GPU computation for model training...")
        
        # Simulate a small neural network training
        print("Creating a simple model for testing...")
        
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Generate dummy data (like game state data)
        x_train = tf.random.normal([1000, 784])
        y_train = tf.random.uniform([1000], 0, 10, dtype=tf.int32)
        
        print("Training model on GPU...")
        with tf.device('/GPU:0'):
            start_time = time.time()
            history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
            gpu_time = time.time() - start_time
            print(f"GPU training time: {gpu_time:.4f} seconds")
        
        print("Training model on CPU...")
        with tf.device('/CPU:0'):
            start_time = time.time()
            history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
            cpu_time = time.time() - start_time
            print(f"CPU training time: {cpu_time:.4f} seconds")
        
        print(f"GPU Speedup: {cpu_time/gpu_time:.2f}x")
        
        # GPU memory info
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            details = tf.config.experimental.get_device_details(gpu_devices[0])
            print(f"\nGPU Details: {details}")
        
        return True
    else:
        print("‚ùå No GPU available")
        return False

def check_project_structure():
    print("\n" + "=" * 60)
    print("Checking Unity Python Project Structure")
    print("=" * 60)
    
    current_files = os.listdir('.')
    print("Files in current directory:")
    for file in current_files:
        if file.endswith('.py'):
            print(f"Ì∞ç {file}")
        elif file.endswith('.ipynb'):
            print(f"Ì≥ì {file}")
        elif file.endswith('.txt'):
            print(f"Ì≥Ñ {file}")
        else:
            print(f"Ì≥Å {file}")

if __name__ == "__main__":
    check_project_structure()
    test_gpu_setup()
