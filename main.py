import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import MirrorTransformer
from trainer import create_trainer

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = 0, 1, 2
VOCAB_SIZE = 29 # 26 letters + 3 special.

os.makedirs(config["output_dir"], exist_ok=True)

class MirrorDataset(Dataset):
    def __init__(self, num_samples, seq_len):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        chars = torch.randint(3, 29, (self.seq_len ,))
        reversed_chars = torch.flip(chars, dims=[0])

        return torch.cat([torch.tensor([SOS_TOKEN]), chars, torch.tensor([EOS_TOKEN]), reversed_chars])

dataset = MirrorDataset(config["training"]["num_samples"], 10)

dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MirrorTransformer(VOCAB_SIZE,config["model"]["d_model"], config["model"]["nhead"], config["model"]["num_layers"], config["model"]["max_length"]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

trainer = create_trainer(model, optimizer, criterion, device, PAD_TOKEN)

print("Starting training...")

trainer.run(dataloader, max_epochs=config["training"]["epochs"])