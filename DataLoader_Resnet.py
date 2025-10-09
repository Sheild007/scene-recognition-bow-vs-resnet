import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, class_to_idx, transform=None):
        """
        image_paths: list of image file paths
        labels: list of class names (strings)
        class_to_idx: dict mapping class name -> integer
        transform: torchvision transforms
        
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_name = self.labels[idx]
        label_idx = self.class_to_idx[label_name]  # convert string label to int

        # Load image
        image = Image.open(img_path).convert("RGB") #force 3-channels format

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_idx, dtype=torch.long)
