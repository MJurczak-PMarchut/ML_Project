import os
import pandas as pd
import numpy
import torch
from torch.utils.data import Dataset
import PreparePulses
from pathlib import Path


pulse_transform = lambda y: torch.tensor(y, dtype=torch.float)/63
cal_vol_transform = lambda y: torch.tensor(y, dtype=torch.float) / 1000


class PulseDataset(Dataset):
    def __init__(self, pulse_dir, transform=None, target_transform=None, max_len=1000):
        self.pulse_dir = Path(pulse_dir)
        self.transform = transform
        self.data = []
        self.target_transform = target_transform
        # load all data to dataset
        for path in self.pulse_dir.glob("*mv.csv"):
            pulse_data = PreparePulses.read_csv_data(path, max_len)
            filtered_pulses = PreparePulses.filter_pulse(pulse_data)
            val = path.name.split('mv')[0].replace('_', '.')
            for fp in filtered_pulses:
                self.data.append({'label': float(val), "pulse": fp})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pulse = self.data[idx]["pulse"]
        label = self.data[idx]["label"]
        if self.transform:
            pulse = self.transform(pulse)
        if self.target_transform:
            label = self.target_transform(label)
        return pulse, label
