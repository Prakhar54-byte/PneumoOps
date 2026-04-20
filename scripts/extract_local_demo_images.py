import os
import numpy as np
from PIL import Image

def extract_local_images():
    os.makedirs('demo_images', exist_ok=True)
    npz_path = os.path.expanduser('~/.medmnist/chestmnist.npz')
    data = np.load(npz_path)
    
    test_images = data['test_images']
    
    # Extract 3 images
    for i in range(3):
        img_array = test_images[i]
        img = Image.fromarray(img_array)
        img.save(f'demo_images/sample_xray_{i+1}.png')
        print(f"Saved demo_images/sample_xray_{i+1}.png")

if __name__ == '__main__':
    extract_local_images()
