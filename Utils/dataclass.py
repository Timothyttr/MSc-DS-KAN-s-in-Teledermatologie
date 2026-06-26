import os
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


class SkinLesionDataset(Dataset):
    def __init__(self, master_csv, img_dir, split_name='train', transform=None):
        self.df = pd.read_csv(master_csv)
        
        # filter only for train data
        self.df = self.df[self.df['split'] == split_name].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
        # strict 5 class
        self.label_map = {'MEL': 0, 'NV': 1, 'BCC': 2, 'BKL': 3, 'AKIEC': 4}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_filename = self.df.loc[idx, 'image_id']
        string_label = str(self.df.loc[idx, 'label']).upper()
        dataset_domain = self.df.loc[idx, 'domain']
        
        # so images can be kept in seperate folders
        img_path = os.path.join(self.img_dir, dataset_domain, img_filename)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Convert to tensor
        label_tensor = torch.tensor(self.label_map[string_label], dtype=torch.long)
        
        return image, label_tensor