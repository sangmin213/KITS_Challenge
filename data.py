from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch
import numpy as np
import cv2
import pickle

class KidneyDataset(Dataset):
    def __init__(self, index="000", preprocess="clahe", logic="AND", transform=None):
        super().__init__()

        with open(f'./data/{preprocess}/input/imaging_{index}.pickle', 'rb') as f:
            self.input = pickle.load(f) # np array (600,512,512)
        with open(f'./data/{preprocess}/label/aggregated_{index}_{logic}.pickle', 'rb') as f:
            self.label = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        data = [ self.input[idx], self.label[idx] ]
        data[1] = transform_basic(data[1])
        if self.transform:
            data[0] = self.transform(data[0])
        else:
            data[0] = transform_basic(data[0])
        
        return data

def transform(input):
    input = cv2.resize(input, (256,256))
    transform1 = transforms.ToTensor()

    relu = nn.ReLU() # input 이미지의 검은색 부분이 음수임. 음수를 0으로 치환
    transform2 = transforms.Normalize((0.5,),(0.5,))

    tensor = transform1(input)
    tensor = transform2(relu(tensor))
    
    return tensor

def transform_basic(input):
    input = cv2.resize(input, (256,256))
    transform = transforms.ToTensor()
    tensor = transform(input)
    
    return tensor