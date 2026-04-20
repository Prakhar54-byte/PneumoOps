import os
import medmnist
from medmnist import INFO
from torchvision import transforms

def extract_images():
    os.makedirs('demo_images', exist_ok=True)
    info = INFO['chestmnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    # Download the dataset
    dataset = DataClass(split='test', download=True)
    
    # Extract 3 images
    for i in range(3):
        img, label = dataset[i]
        # label is a multi-label array, we can just save the image
        img.save(f'demo_images/sample_xray_{i+1}.png')
        print(f"Saved demo_images/sample_xray_{i+1}.png with label: {label}")

if __name__ == '__main__':
    extract_images()
