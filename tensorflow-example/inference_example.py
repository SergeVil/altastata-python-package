import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from altastata import AltaStataTensorFlowDataset
import altastata_config
import keras

@keras.saving.register_keras_serializable(package="CustomLayers")
class EdgeDetectionLayer(tf.keras.layers.Layer):
    """Custom layer for edge detection using Sobel filters."""
    def __init__(self, **kwargs):
        super(EdgeDetectionLayer, self).__init__(**kwargs)
        # Initialize Sobel filters
        self.sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        self.sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        # Reshape for conv2d
        self.sobel_x = tf.reshape(self.sobel_x, [3, 3, 1, 1])
        self.sobel_y = tf.reshape(self.sobel_y, [3, 3, 1, 1])

    def call(self, inputs):
        # Split input into channels
        channels = tf.split(inputs, 3, axis=-1)
        edges = []
        
        for channel in channels:
            # Apply Sobel filters
            edge_x = tf.nn.conv2d(channel, self.sobel_x, strides=[1,1,1,1], padding='SAME')
            edge_y = tf.nn.conv2d(channel, self.sobel_y, strides=[1,1,1,1], padding='SAME')
            edge_mag = tf.sqrt(tf.square(edge_x) + tf.square(edge_y))
            edges.append(edge_mag)
        
        edge_features = tf.concat(edges, axis=-1)
        return tf.concat([inputs, edge_features], axis=-1)

    def get_config(self):
        config = super(EdgeDetectionLayer, self).get_config()
        return config

def get_file_extension(path):
    """Extract the file extension from a path, handling AltaStata paths."""
    # If the path has a version suffix (✹), extract just the file part
    if '✹' in str(path):
        # Extract the file part before the version marker
        file_part = str(path).split('✹')[0]
        # Get the extension
        _, ext = os.path.splitext(file_part)
        return ext.lower()
    else:
        # Standard path handling
        _, ext = os.path.splitext(str(path))
        return ext.lower()

def preprocess_image(image, label):
    """Preprocess images for inference."""
    # Convert to float32 and normalize
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Resize to 96x96
    image = tf.image.resize(image, [96, 96])
    
    return image, label

def visualize_predictions(images, predictions, class_names, max_images=9):
    """Visualize images with their predictions."""
    plt.figure(figsize=(15, 15))
    for i in range(min(max_images, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        pred_class = np.argmax(predictions[i])
        confidence = predictions[i][pred_class]
        plt.title(f'Pred: {class_names[pred_class]}\nConf: {confidence:.2f}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Create dataset for inference
    dataset = AltaStataTensorFlowDataset(
        "bob123_rsa",  # Using AltaStata account
        root_dir="tensorflow_test/data/images",
        file_pattern="*.png",
        preprocess_fn=preprocess_image
    )
    
    # Create a separate dataset instance for model loading with proper file pattern
    model_dataset = AltaStataTensorFlowDataset(
        "bob123_rsa",
        root_dir="tensorflow_test/model",
        file_pattern="*.keras",  # Only .keras files, excludes .provenance.txt
        require_files=False
    )
    
    # Load the trained model from the resolved file path
    print(f"Loading model from AltaStata...")
    model = model_dataset.load_model(model_dataset.file_paths[0])
    print("Model loaded successfully")
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Total files: {len(dataset)}")
    circle_count = sum(1 for path in dataset.file_paths if 'circle' in str(path))
    rectangle_count = sum(1 for path in dataset.file_paths if 'rectangle' in str(path))
    print(f"Circle images: {circle_count}")
    print(f"Rectangle images: {rectangle_count}\n")
    
    # Create inference dataset
    inference_ds = dataset.batch(4).prefetch(tf.data.AUTOTUNE)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(inference_ds)
    
    # Get original images for visualization
    original_images = []
    for image, _ in dataset:
        original_images.append(image.numpy())
    
    # Define class names
    class_names = ['rectangle', 'circle']
    
    # Visualize predictions
    print("\nVisualizing predictions...")
    visualize_predictions(original_images, predictions, class_names)
    
    # Print summary statistics
    print("\nPrediction Summary:")
    total_correct = 0
    for i, (path, pred) in enumerate(zip(dataset.file_paths, predictions)):
        pred_class = np.argmax(pred)
        true_class = 1 if 'circle' in str(path) else 0
        if pred_class == true_class:
            total_correct += 1
    
    accuracy = total_correct / len(dataset)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Total Images: {len(dataset)}")
    print(f"Correct Predictions: {total_correct}")

if __name__ == '__main__':
    main() 