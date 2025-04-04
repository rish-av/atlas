## to be designed based on the json file

import json
import torch
from torch.utils.data import Dataset
class JsonDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # Convert embedding lists (or arrays) into torch tensors
        sample['stack_trace_embedding'] = torch.tensor(sample['stack_trace_embedding'], dtype=torch.float32)
        sample['file_embeddings'] = torch.tensor(sample['file_embeddings'], dtype=torch.float32)
        sample['function_embeddings'] = torch.tensor(sample['function_embeddings'], dtype=torch.float32)
        sample['line_embeddings'] = torch.tensor(sample['line_embeddings'], dtype=torch.float32)
        return sample
