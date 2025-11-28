import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from ram.transform import get_transform
from preprocessors.lion_preprocessors import ImageEvalProcessor

class RefCOCODataset(Dataset):
    def __init__(self, img_root: str, ann_path: str):
        """
        Args:
            img_root (str): Path to COCO train2014 images.
            ann_path (str): Path to the RefCOCO json file (e.g., testA.json).
        """
        super().__init__()
        self.img_root = img_root
        self.ann_path = ann_path
        
        self.processor = ImageEvalProcessor()
        self.ram_processor = get_transform()
        
        self.samples = self._load_annotations()

    def _load_annotations(self):
        """Loads RefCOCO annotations from the JSON file"""
        print(f"Loading RefCOCO annotations from {self.ann_path}...")
        with open(self.ann_path, "r") as f:
            data = json.load(f)
        
        samples = []
        
        for item in data:
            img_name = item['img_name']
            
            path_train = os.path.join(self.img_root, "train2014", img_name)
            path_val = os.path.join(self.img_root, "val2014", img_name)
            path_root = os.path.join(self.img_root, img_name)
            
            if os.path.exists(path_train):
                img_path = path_train
            elif os.path.exists(path_val):
                img_path = path_val
            elif os.path.exists(path_root):
                img_path = path_root
            else:
                # Skip if image not found
                print(f"Warning: Image not found {img_name}")
                continue 

            bbox = item['bbox'] 
            
            # Each sentence becomes a separate training exampl
            for sent_data in item['sentences']:
                text = sent_data['sent'] if isinstance(sent_data, dict) else sent_data
                
                samples.append({
                    "img_path": img_path,
                    "bbox": bbox,
                    "sentence": text
                })
                
        print(f"Loaded {len(samples)} samples.")
        return samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Gets the processed image, RAM image, sentence, bounding box, and image size for the given index"""
        sample = self.samples[idx]
        img_path = sample['img_path']
        sentence = sample['sentence']
        bbox = sample['bbox'] # [x, y, w, h]

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                w, h = img.size # Keep original dimensions to rescale predictions later
                proc_img = self.processor(img)
                ram_img = self.ram_processor(img)
        except Exception as e: # Skip problematic images
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        return proc_img, ram_img, sentence, torch.tensor(bbox), torch.tensor([w, h])