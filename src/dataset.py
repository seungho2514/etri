import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, root_dir, meta_csv_path, train, test_fold, duration, sr):
        self.root_dir = root_dir
        self.df = pd.read_csv(meta_csv_path)
        self.duration = duration
        self.sr = sr
        self.target_len = int(self.duration * self.sr)

        if train:
            self.df = self.df[self.df['fold'] != test_fold]

        else:
            self.df = self.df[self.df['fold'] == test_fold]

        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def process(self, path):
        # 1. Load audio
        wav, sr = torchaudio.load(path)

        # 2. Mono 
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # 3. Resample
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        # 4. Padding / Cutting
        current_len = wav.shape[-1]
        if current_len < self.target_len:
            wav = torch.nn.functional.pad(wav, (0, self.target_len - current_len))

        elif current_len > self.target_len:
            wav = wav[..., :self.target_len]

        return wav
        
class UrbanSoundDataset(BaseDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.root_dir, f"fold{row['fold']}", row['slice_file_name'])

        return {
            'wav': self.process(path),
            'label': row['classID']
        }

class ESCDataset(BaseDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = os.path.join(self.root_dir, 'audio', row['filename'])

        return {
            'wav': self.process(path),
            'label': row['target']
        }
    
def get_dataset(cfg):
    duration = cfg.get('duration', 5.0)
    
    is_train = cfg.get('train', True) 

    if cfg.name == 'urbansound':
        return UrbanSoundDataset(
            root_dir=cfg.path, 
            meta_csv_path=cfg.meta_csv, 
            train=is_train,
            test_fold=cfg.fold, 
            duration=duration, 
            sr=16000
        )
    
    elif cfg.name == 'esc50':
        return ESCDataset(
            root_dir=cfg.path, 
            meta_csv_path=cfg.meta_csv, 
            train=is_train,
            test_fold=cfg.fold, 
            duration=duration, 
            sr=16000
        )
    
    else:
        raise ValueError(f"Unknown dataset: {cfg.name}")