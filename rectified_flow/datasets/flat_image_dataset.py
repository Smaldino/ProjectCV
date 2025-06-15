from torch.utils.data import Dataset
from PIL import Image
import os
import glob

class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None, ext="jpg"):
        self.root = root
        self.transform = transform
        self.paths = sorted(glob.glob(os.path.join(root, f"*.{ext}")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img