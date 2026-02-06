import torch
import torch.nn as nn
import math

# Creating sinusoidal positional encodings, so the model knows the token order.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()

        # Creating a matrix of shape (max_len, d_model).
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even indices, cos to odd indices.
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices.
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices.

        self.register_buffer('pe', pe.unsqueeze(0))  

    def forward(self, x):
        # Add positional encoding to token embeddings.
        return x + self.pe[:, :x.size(1)]
    
# Decoder only Transformer for sequence modelling.
class MirrorTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    #  Causal mask generator for decoder to prevent attending to future tokens.
    def generate_square_subsequent_mask(self, size, device): 
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self, tgt):
        x = self.embedding(tgt) 

        x = self.pos_encoder(x)

        # Automaticallly builds the causal mask.
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)

        # Using x as both memory and target since this is decoder only.
        output = self.transformer_decoder(x, x, tgt_mask=tgt_mask)

        return self.fc_out(output)