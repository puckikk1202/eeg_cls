import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import wandb
from data.eegset import EEGDataset, create_dataloader_from_folders, custom_collate_fn

from sklearn.metrics import confusion_matrix
from model.resnet import ResNetMLP
import seaborn as sns
import matplotlib.pyplot as plt

wandb.init(project='eeg_cls')

class EEGTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, seq_len, num_layers=6, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.2, mlp_hidden_dim=1024, mlp_num_layers=3):
        super(EEGTransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = ResNetMLP(d_model * seq_len, mlp_hidden_dim, num_classes, mlp_num_layers)
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)  # (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2).contiguous().view(x.size(1), -1)  # (batch_size, d_model * seq_len)
        x = self.mlp(x)  # (batch_size, num_classes)
        return x

def train_transformer_model(train_loader, test_loader, model, criterion, optimizer, num_epochs=10):
    model.train()
    best_acc = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for eeg_data, emotion_labels in train_loader:

            eeg_data = eeg_data.cuda()
            emotion_labels = emotion_labels.cuda()  

            optimizer.zero_grad()
            outputs = model(eeg_data)
            loss = criterion(outputs, torch.argmax(emotion_labels, dim=1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(emotion_labels, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 验证模型
        model.eval()
        test_loss, test_acc, test_preds, test_labels, train_preds, train_labels = evaluate_model(test_loader, train_loader, model, criterion)

        cm = confusion_matrix(test_labels, test_preds)
        test_cm_fig = plot_confusion_matrix(cm, classes=['Neutral', 'Smile', 'Sad'])

        cm = confusion_matrix(train_labels, train_preds)
        train_cm_fig = plot_confusion_matrix(cm, classes=['Neutral', 'Smile', 'Sad'])
        
        # 打印和记录训练和测试的损失和准确率
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'train_confusion_matrix': wandb.Image(train_cm_fig),
            'test_confusion_matrix': wandb.Image(test_cm_fig),
        })

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), './model_outputs/best_model.pth')
            print(f'Best model saved with accuracy: {best_acc:.2f}%')

        model.train()
    
    torch.save(model.state_dict(), './model_outputs/egg_cls.pth')

def evaluate_model(test_dataloader, train_dataloader, model, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for eeg_data, emotion_labels in test_dataloader:

            eeg_data = eeg_data.cuda()
            emotion_labels = emotion_labels.cuda()

            outputs = model(eeg_data)
            loss = criterion(outputs, torch.argmax(emotion_labels, dim=1))
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(emotion_labels, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


        train_all_preds = []
        train_all_labels = []
        for eeg_data, emotion_labels in train_dataloader:

            eeg_data = eeg_data.cuda()
            emotion_labels = emotion_labels.cuda()

            outputs = model(eeg_data)
            # loss = criterion(outputs, torch.argmax(emotion_labels, dim=1))
            # running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(emotion_labels, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())
    
    loss = running_loss / len(test_dataloader)
    acc = 100 * correct / total
    return loss, acc, all_preds, all_labels, train_all_preds, train_all_labels

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.close(fig)  # Close the figure to prevent it from displaying in Jupyter notebooks
    return fig


input_dim = 14  
num_classes = 3  
seq_len = 240  
batch_size = 64
num_epochs = 100
learning_rate = 3e-5

model = EEGTransformerClassifier(input_dim=input_dim, num_classes=num_classes, seq_len=seq_len).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

eeg_folder_path = './dataset/eegset' 
emotion_folder_path = './dataset/anno'

train_loader, test_loader = create_dataloader_from_folders(eeg_folder_path, emotion_folder_path, batch_size=batch_size, seq_len=seq_len)
train_transformer_model(train_loader, test_loader, model, criterion, optimizer, num_epochs=num_epochs)