import os
import numpy as np
from PIL import Image

def create_sample_image(path, size=(100, 100), class_type='circle'):
    """Create a sample image with either a circle or a rectangle"""
    # Create a blank image with white background
    img = Image.new('RGB', size, 'white')
    pixels = np.array(img)
    
    # Get center coordinates
    center_x, center_y = size[0] // 2, size[1] // 2
    
    if class_type == 'circle':
        # Draw a red circle
        radius = min(size) // 4
        y, x = np.ogrid[-center_y:size[1]-center_y, -center_x:size[0]-center_x]
        mask = x*x + y*y <= radius*radius
        pixels[mask] = [255, 0, 0]  # Red color
    else:
        # Draw a blue rectangle
        rect_size = min(size) // 2
        x_start = center_x - rect_size // 2
        x_end = center_x + rect_size // 2
        y_start = center_y - rect_size // 2
        y_end = center_y + rect_size // 2
        pixels[y_start:y_end, x_start:x_end] = [0, 0, 255]  # Blue color
    
    img = Image.fromarray(pixels)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def create_sample_csv(path, rows=10, cols=5):
    """Create a sample CSV file with random numbers"""
    data = np.random.randn(rows, cols)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ','.join([f'col_{i}' for i in range(cols)])
    np.savetxt(path, data, delimiter=',', header=header, comments='')

def create_sample_numpy(path, shape=(10, 5)):
    """Create a sample numpy array with random numbers"""
    data = np.random.randn(*shape)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)

def generate_sample_files():
    # Generate image files (circles and rectangles)
    for i in range(5):
        # Create circle images (class 0)
        create_sample_image(
            os.path.join("data", "images", f"circle_{i}.jpg"),
            class_type='circle'
        )
        # Create rectangle images (class 1)
        create_sample_image(
            os.path.join("data", "images", f"rectangle_{i}.jpg"),
            class_type='rectangle'
        )
    
    # Generate CSV files
    for i in range(5):
        create_sample_csv(os.path.join("data", "csv", f"sample_{i}.csv"))
    
    # Generate numpy files
    for i in range(5):
        create_sample_numpy(os.path.join("data", "numpy", f"sample_{i}.npy"))

if __name__ == "__main__":
    generate_sample_files()
    print("Sample files generated successfully!") 