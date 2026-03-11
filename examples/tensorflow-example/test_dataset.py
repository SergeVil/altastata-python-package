import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from altastata import AltaStataTensorFlowDataset
import altastata_config
import re

# Add function to extract correct file extension from AltaStata paths
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
    """Preprocess images for training."""
    def _preprocess(img, lbl):
        # Convert to float32 and normalize
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Resize to 96x96
        img = tf.image.resize(img, [96, 96])
        # Convert label to int32
        lbl = tf.cast(lbl, tf.int32)
        return img, lbl
    
    # Wrap the preprocessing function
    image, label = tf.py_function(
        _preprocess,
        [image, label],
        [tf.float32, tf.int32]
    )
    
    # Set shapes explicitly after preprocessing
    image.set_shape([96, 96, 3])
    label.set_shape([])
    
    return image, label

def test_dataset_basic():
    """Test basic dataset functionality."""
    print("\nTesting basic dataset functionality...")
    
    # Create dataset
    dataset = AltaStataTensorFlowDataset(
        "bob123_rsa",  # Using AltaStata account for testing
        root_dir="tensorflow_test/data/images",
        file_pattern="*.png",
        preprocess_fn=preprocess_image
    )
    
    # Test dataset length
    print(f"Dataset length: {len(dataset)}")
    
    # Test getting a single item
    image, label = next(iter(dataset))
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Total files: {len(dataset)}")
    circle_count = sum(1 for path in dataset.file_paths if 'circle' in str(path))
    rectangle_count = sum(1 for path in dataset.file_paths if 'rectangle' in str(path))
    print(f"Circle images: {circle_count}")
    print(f"Rectangle images: {rectangle_count}")
    
    return dataset

def test_dataset_preprocessing():
    """Test dataset preprocessing."""
    print("\nTesting dataset preprocessing...")
    
    # Create dataset with preprocessing
    dataset = AltaStataTensorFlowDataset(
        "bob123_rsa",  # Using AltaStata account for testing
        root_dir="tensorflow_test/data/images",
        file_pattern="*.png",
        preprocess_fn=preprocess_image
    )
    
    # Test preprocessing on a single item
    image, label = next(iter(dataset))
    print(f"Preprocessed image shape: {image.shape}")
    print(f"Preprocessed label: {label}")
    
    return dataset

def test_dataset_batching():
    """Test dataset batching."""
    print("\nTesting dataset batching...")
    
    dataset = AltaStataTensorFlowDataset(
        "bob123_rsa",  # Using AltaStata account for testing
        root_dir="tensorflow_test/data/images",
        file_pattern="*.png",
        preprocess_fn=preprocess_image
    )
    
    # Create batched dataset
    batched_dataset = dataset.batch(4)
    
    # Test getting a batch
    images, labels = next(iter(batched_dataset))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    
    return batched_dataset

def main():
    """Run all dataset tests."""
    print("Starting dataset tests...")
    
    # Run all tests
    test_dataset_basic()
    test_dataset_preprocessing()
    test_dataset_batching()
    
    print("\nAll tests completed!")

if __name__ == '__main__':
    main() 