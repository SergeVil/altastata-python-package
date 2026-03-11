import os
import numpy as np
from PIL import Image, ImageDraw

def create_sample_data():
    """Generate sample data files for testing."""
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create data directories
    data_dirs = {
        'numpy': os.path.join(script_dir, 'data/numpy'),
        'csv': os.path.join(script_dir, 'data/csv'),
        'images': os.path.join(script_dir, 'data/images')
    }
    
    for dir_path in data_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Generate numpy files
    for i in range(5):
        data = np.random.rand(10, 5)
        np.save(f"{data_dirs['numpy']}/sample_{i}.npy", data)

    # Generate CSV files
    for i in range(5):
        data = np.random.rand(11, 5)
        np.savetxt(f"{data_dirs['csv']}/sample_{i}.csv", data, delimiter=',')

    # Generate image files
    for i in range(5):
        # Create circle
        img = Image.new('RGB', (100, 100), 'white')
        draw = ImageDraw.Draw(img)
        draw.ellipse((20, 20, 80, 80), fill='black')
        img.save(f"{data_dirs['images']}/circle_{i}.png")

        # Create rectangle
        img = Image.new('RGB', (100, 100), 'white')
        draw = ImageDraw.Draw(img)
        draw.rectangle((20, 20, 80, 80), fill='black')
        img.save(f"{data_dirs['images']}/rectangle_{i}.png")

if __name__ == "__main__":
    create_sample_data()
    print("Sample files generated successfully!") 