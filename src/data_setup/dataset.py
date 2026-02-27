from torch.utils.data import Dataset 
from PIL import Image
from colorama import Fore
import os

class AnimalsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.allowable_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.samples = []
        for cls_name in self.class_names:
            cls_folder = os.path.join(root_dir, cls_name)
            for file_name in os.listdir(cls_folder):
                if any(file_name.lower().endswith(ext) for ext in self.allowable_extensions):
                    file_path = os.path.join(cls_folder, file_name)
                    self.samples.append((file_path, self.class_to_idx[cls_name]))
        
        if len(self.samples) == 0:
            raise ValueError(
                f"{Fore.LIGHTRED_EX}No valid image files found in {Fore.WHITE}{root_dir}"
                f"{Fore.LIGHTRED_EX}. Please check the directory structure and supported file extensions "
                f"(.jpg, .jpeg, .png).{Fore.RESET}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label
