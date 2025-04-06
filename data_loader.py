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

from torch.nn.utils.rnn import pad_sequence
def custom_collate_fn(batch):
    collated = {}
    # Process each key in the dictionary. 
    # For keys that are tensors and might have variable first dimensions, we pad them.
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            # Check if the tensor is 2D (e.g., shape [n, 768]) and if the first dimension varies.
            shapes = [item[key].shape for item in batch]
            if len({s[0] for s in shapes}) > 1:
                # Pad the sequence of tensors along dimension 0
                collated[key] = pad_sequence([item[key] for item in batch], batch_first=True)
            else:
                collated[key] = torch.stack([item[key] for item in batch])
        else:
            # For non-tensor values (like indices), just put them in a list.
            collated[key] = [item[key] for item in batch]
    return collated