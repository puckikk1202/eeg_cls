import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split

class EEGDataset(Dataset):
    def __init__(self, eeg_csv_files, emotion_csv_files, seq_len=240):
        self.data = []
        self.seq_len = seq_len
        self.channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        
        for csv_file in eeg_csv_files:
            data = pd.read_csv(csv_file)
            if not all(channel in data.columns for channel in self.channels):
                raise ValueError(f"Some channels are missing in the file {csv_file}")
                
            min_vals = data[self.channels].min()
            max_vals = data[self.channels].max()
            
            # Normalize to -1 to 1
            for channel in self.channels:
                data[channel] = 2 * (data[channel] - min_vals[channel]) / (max_vals[channel] - min_vals[channel]) - 1
            
            self.data.append(data)
        
        self.data = pd.concat(self.data, ignore_index=True)

        # Load emotion label data
        self.emotion_data = []
        for csv_file in emotion_csv_files:
            emotion_data = pd.read_csv(csv_file)
            emotion_data = emotion_data[emotion_data['Emotion'] != 'N/A']
            self.emotion_data.append(emotion_data)
        
        self.emotion_data = pd.concat(self.emotion_data, ignore_index=True)
        self.emotion_labels = {'Neutral': 0, 'Smile': 1, 'Sad': 2}

    def __len__(self):
        return len(self.data['Timestamp'].unique())

    def __getitem__(self, idx):
        while idx < self.__len__():
            timestamp = self.data['Timestamp'].unique()[idx]
            data_at_timestamp = self.data[self.data['Timestamp'] == timestamp]

            eeg_data = []
            for channel in self.channels:
                channel_data = data_at_timestamp[channel].values
                if len(channel_data) > self.seq_len:
                    channel_data = channel_data[:self.seq_len]
                elif len(channel_data) < self.seq_len:
                    padding = np.zeros(self.seq_len - len(channel_data))
                    channel_data = np.concatenate((channel_data, padding))
                eeg_data.append(channel_data)

            eeg_data = np.array(eeg_data)
            eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)

            # Find corresponding emotion label
            emotion_row = self.emotion_data[(self.emotion_data['Timestamp Start'] <= timestamp) & 
                                            (self.emotion_data['Timestamp End'] >= timestamp)]
            if not emotion_row.empty:
                emotion_label = emotion_row.iloc[0]['Emotion']
            else:
                idx += 1
                continue

            if emotion_label not in self.emotion_labels:
                idx += 1
                continue

            emotion_index = self.emotion_labels[emotion_label]
            emotion_one_hot = np.zeros(len(self.emotion_labels))
            emotion_one_hot[emotion_index] = 1
            emotion_tensor = torch.tensor(emotion_one_hot, dtype=torch.float32)

            return eeg_tensor, emotion_tensor
        

def create_dataloader_from_folders(eeg_folder_path, emotion_folder_path, batch_size=32, seq_len=240, test_split=0.2):
    eeg_csv_files = [os.path.join(eeg_folder_path, f) for f in os.listdir(eeg_folder_path) if f.endswith('.csv')]
    emotion_csv_files = [os.path.join(emotion_folder_path, f) for f in os.listdir(emotion_folder_path) if f.endswith('.csv')]
    dataset = EEGDataset(eeg_csv_files, emotion_csv_files, seq_len)
    
    # Split dataset into training and test sets
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    return train_loader, test_loader

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)
    eeg_data, emotion_labels = zip(*batch)
    eeg_data = torch.stack(eeg_data)
    emotion_labels = torch.stack(emotion_labels)
    return eeg_data, emotion_labels


# Usage example
eeg_folder_path = './dataset/eegset'  # Please replace with the actual folder containing EEG data
emotion_folder_path = './dataset/anno'  # Please replace with the actual folder containing emotion label data
train_loader, test_loader = create_dataloader_from_folders(eeg_folder_path, emotion_folder_path)

# Test dataloader
for batch in train_loader:
    eeg_data, emotion_labels = batch
    # print(eeg_data.shape, emotion_labels.shape)