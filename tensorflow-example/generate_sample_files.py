import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import tensorflow as tf
from pathlib import Path

def create_sample_images(output_dir, num_samples=100):
    """Create sample images of circles and rectangles."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Create a blank image
        img = Image.new('RGB', (128, 128), color='white')
        draw = ImageDraw.Draw(img)
        
        # Randomly choose between circle and rectangle
        if np.random.random() < 0.5:
            # Draw a circle
            x = np.random.randint(20, 108)
            y = np.random.randint(20, 108)
            radius = np.random.randint(10, 30)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill='blue', outline='black')
            label = 'circle'
        else:
            # Draw a rectangle
            x1 = np.random.randint(20, 88)
            y1 = np.random.randint(20, 88)
            width = np.random.randint(20, 40)
            height = np.random.randint(20, 40)
            draw.rectangle([x1, y1, x1+width, y1+height], 
                         fill='red', outline='black')
            label = 'rectangle'
        
        # Save the image
        img.save(os.path.join(output_dir, f'{label}_{i}.png'))

def create_sample_csv(output_dir, num_samples=100):
    """Create sample CSV files with random data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame with random data
    data = {
        'feature1': np.random.randn(num_samples),
        'feature2': np.random.randn(num_samples),
        'feature3': np.random.randn(num_samples),
        'label': np.random.randint(0, 2, num_samples)
    }
    df = pd.DataFrame(data)
    
    # Save as CSV
    df.to_csv(os.path.join(output_dir, 'sample_data.csv'), index=False)

def create_sample_numpy(output_dir, num_samples=100):
    """Create sample NumPy arrays."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create random arrays
    for i in range(num_samples):
        # Create a random array
        arr = np.random.randn(10, 10)
        
        # Save as .npy file
        np.save(os.path.join(output_dir, f'sample_array_{i}.npy'), arr)

def main():
    # Create output directories
    base_dir = 'data'
    image_dir = os.path.join(base_dir, 'images')
    csv_dir = os.path.join(base_dir, 'csv')
    numpy_dir = os.path.join(base_dir, 'numpy')
    models_dir = os.path.join(base_dir, 'models')
    
    # Create directories
    for dir_path in [image_dir, csv_dir, numpy_dir, models_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Generate sample files
    print("Generating sample images...")
    create_sample_images(image_dir)
    
    print("Generating sample CSV files...")
    create_sample_csv(csv_dir)
    
    print("Generating sample NumPy arrays...")
    create_sample_numpy(numpy_dir)
    
    print("\nSample files generated successfully!")
    print(f"Images: {len(list(Path(image_dir).glob('*.png')))}")
    print(f"CSV files: {len(list(Path(csv_dir).glob('*.csv')))}")
    print(f"NumPy files: {len(list(Path(numpy_dir).glob('*.npy')))}")

if __name__ == '__main__':
    main() 