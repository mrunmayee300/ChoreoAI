import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np

# Define Dance Encoder (LSTM-based)
class DanceEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=256, num_layers=2):
        super(DanceEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Bidirectional output

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)  # Concatenate both directions
        return self.fc(hn)

# Define Text Encoder using DistilBERT
class TextEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super(TextEncoder, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return self.fc(outputs.last_hidden_state[:, 0, :])  # CLS token representation

# Contrastive Loss (Improved NT-Xent Loss with Stability)
def contrastive_loss(dance_emb, text_emb, temperature=0.5):
    dance_emb = nn.functional.normalize(dance_emb, dim=1)
    text_emb = nn.functional.normalize(text_emb, dim=1)
    logits = torch.mm(dance_emb, text_emb.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return nn.CrossEntropyLoss()(logits, labels)

# Dataset Class for Motion Capture Data
class DanceTextDataset(Dataset):
    def __init__(self, dance_data, text_data, tokenizer, max_seq_len=50):
        self.dance_data = dance_data
        self.text_data = text_data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.dance_data)
    
    def __getitem__(self, idx):
        dance_seq = self.dance_data[idx]  # Shape: (t, 3)
        text = self.text_data[idx]
        
        # Pad or truncate dance sequences to max_seq_len
        if len(dance_seq) < self.max_seq_len:
            pad_size = self.max_seq_len - len(dance_seq)
            padding = np.zeros((pad_size, 3))
            dance_seq = np.vstack([dance_seq, padding])
        else:
            dance_seq = dance_seq[:self.max_seq_len]
        
        dance_seq = torch.tensor(dance_seq, dtype=torch.float32)
        text_tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
        return dance_seq, text_tokens["input_ids"].squeeze(0), text_tokens["attention_mask"].squeeze(0)

# Training Loop with DataLoader
def train(model_dance, model_text, dataloader, optimizer, device):
    model_dance.train()
    model_text.train()
    total_loss = 0
    for dance_seq, input_ids, attention_mask in dataloader:
        dance_seq, input_ids, attention_mask = dance_seq.to(device), input_ids.to(device), attention_mask.to(device)
        
        dance_emb = model_dance(dance_seq)  # (batch_size, embedding_dim)
        text_emb = model_text(input_ids, attention_mask)  # (batch_size, embedding_dim)
        
        loss = contrastive_loss(dance_emb, text_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Average Loss: {total_loss / len(dataloader)}")

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dance_model = DanceEncoder().to(device)
text_model = TextEncoder().to(device)

# Optimizer
optimizer = optim.Adam(list(dance_model.parameters()) + list(text_model.parameters()), lr=1e-4)

# Tokenizer for text
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Debugging: Print Model Summary
print(dance_model)
print(text_model)