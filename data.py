import torch, cv2
import numpy as np
from PIL import Image
from patchify import patchify
from torch.utils.data import Dataset



# TRdataset class for transformer based training

class TRdataset(Dataset):
    def __init__(self, df, encoder, max_sequence=19, transformer=None): # max_sequence = 9
        self.encoder = encoder
        self.transformer = transformer
        self.max_sequence = max_sequence
        self.file_name = df['file_name'].tolist()
        self.plate_texts = df['plate_texts'].tolist()
    
    def __len__(self):
        return len(self.file_name)
    
    def __getitem__(self, x):
        image = cv2.imread("dataset/JPEGImages/" + self.file_name[x][:-3] + "jpg", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 112), interpolation=cv2.INTER_AREA)
        # patches = patchify(image=image, patch_size=(16, 16), step=16).reshape(-1, 16, 16)
        
        if self.transformer: 
            image = self.transformer(image=image)['image'] 
            patches = torch.tensor(patchify(image=image, patch_size=(16, 16), step=16).reshape(21, -1), dtype=torch.float32) / 255.0

        else:
            patches = patchify(image=image, patch_size=(16, 16), step=16).reshape(21, -1)
            patches = torch.tensor(patches, dtype=torch.float32) / 255.0
        
        text = torch.tensor([self.encoder[x] for x in self.plate_texts[x]], dtype=torch.long)
        target_lengths = len(text)
        pad = torch.zeros(size=(self.max_sequence - target_lengths, ))
        text = torch.cat([text, pad])
        return patches, text, self.max_sequence, target_lengths



# dataset class for the other architecture
class dataset(Dataset):
    def __init__(self, df, encoder, max_sequence=19, transformer=None): # max_sequence = 9
        self.encoder = encoder
        self.transformer = transformer
        self.max_sequence = max_sequence
        self.file_name = df['file_name'].tolist()
        self.plate_texts = df['plate_texts'].tolist()
    
    def __len__(self):
        return len(self.file_name)
    
    def __getitem__(self, x):
        image = np.array(Image.open("data/JPEGImages/" + self.file_name[x][:-3] + "jpg"))
        if self.transformer: 
            image = self.transformer(image=image)['image'] / 255.0
        else: 
            image = torch.tensor(image) / 255.0
        
        text = torch.tensor([self.encoder[x] for x in self.plate_texts[x]], dtype=torch.long)
        target_lengths = len(text)
        pad = torch.zeros(size=(self.max_sequence - target_lengths, ))
        text = torch.cat([text, pad])
        return image, text, self.max_sequence, target_lengths



















# # dataset class for the other architectures
# class dataset(Dataset):
#     def __init__(self, df, encoder, max_sequence=19, transformer=None): # max_sequence = 9
#         self.encoder = encoder
#         self.transformer = transformer
#         self.max_sequence = max_sequence
#         self.file_name = df['file_name'].tolist()
#         self.plate_texts = df['plate_texts'].tolist()
    
    
#     def rotate(self, image):
#         img1 = np.array(image.rotate(345))
#         img2 = np.array(image.rotate(15))
#         return np.concatenate([image, img1, img2], axis=-1)
    
    
#     def __len__(self):
#         return len(self.file_name)
    
#     def __getitem__(self, x):
#         image = Image.open("dataset/JPEGImages/" + self.file_name[x][:-3] + "jpg").convert("RGB")
#         image = self.rotate(image)
        
#         if self.transformer: 
#             image = self.transformer(image=image)['image'] / 255.0
#         else: 
#             image = torch.tensor(image) / 255.0
        
#         text = torch.tensor([self.encoder[x] for x in self.plate_texts[x]], dtype=torch.long)
#         target_lengths = len(text)
#         pad = torch.zeros(size=(self.max_sequence - target_lengths, ))
#         text = torch.cat([text, pad])
#         return image, text, self.max_sequence, target_lengths
