import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import string
import math
from transformers import GPT2Tokenizer  

torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SubwordTokenizer:
    def __init__(self):
        # Load GPT2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Add special tokens
        special_tokens = {
            'pad_token': '<pad>',
            'eos_token': '<eos>',
            'bos_token': '<sos>'
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Save special token IDs
        self.pad_token = self.tokenizer.pad_token_id
        self.sos_token = self.tokenizer.bos_token_id
        self.eos_token = self.tokenizer.eos_token_id
        
        # Update vocab size
        self.vocab_size = len(self.tokenizer)
    
    def encode(self, text):
        # Encode text into token IDs and add SOS and EOS tokens
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True
        )
        return [self.sos_token] + tokens + [self.eos_token]
    
    def decode(self, tokens, skip_special=True):
        # Decode token IDs into text
        if skip_special:
            tokens = [t for t in tokens if t not in [self.sos_token, self.eos_token, self.pad_token]]
        return self.tokenizer.decode(tokens)

    # Add properties for special tokens
    @property
    def char_to_idx(self):
        return self.tokenizer.get_vocab()

    @property
    def idx_to_char(self):
        return {v: k for k, v in self.tokenizer.get_vocab().items()}

class CharacterTokenizer:
    def __init__(self):
        # Define basic character set
        self.chars = set(string.printable[:-5])  
        
        # Special tokens
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        
        # Create vocabulary
        self.char_to_idx = {
            self.pad_token: 0,
            self.sos_token: 1,
            self.eos_token: 2,
            **{char: i+3 for i, char in enumerate(sorted(self.chars))}
        }
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, text):
        return [self.char_to_idx[self.sos_token]] + [
            self.char_to_idx[c] for c in text if c in self.chars
        ] + [self.char_to_idx[self.eos_token]]
    
    def decode(self, tokens, skip_special=True):
        if skip_special:
            tokens = [t for t in tokens if t not in [
                self.char_to_idx[self.sos_token],
                self.char_to_idx[self.eos_token],
                self.char_to_idx[self.pad_token]
            ]]
        return ''.join(self.idx_to_char[t] for t in tokens)

class ShakespeareDataset(Dataset):
    def __init__(self, data, tokenizer, seq_length):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        # Pre-tokenize and cache all sequences
        print("Pre-processing dataset...")
        all_tokens = []
        for line in tqdm(data, desc="Tokenizing data"):
            tokens = tokenizer.encode(line)
            # Skip empty sequences
            if len(tokens) > 1:  
                all_tokens.extend(tokens)
        
        tokens = torch.tensor(all_tokens, dtype=torch.long)
        
        # Create sequences with proper padding
        n_sequences = (len(tokens) - 1) // seq_length
        self.sequences = []
        
        for i in range(n_sequences):
            start_idx = i * seq_length
            end_idx = start_idx + seq_length
            
            sequence = tokens[start_idx:end_idx]
            target = tokens[start_idx + 1:end_idx + 1]
            
            if len(sequence) == seq_length:
                self.sequences.append((sequence, target))
        
        print(f"Created {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin and cos functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Linear projections and reshape
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute context vector
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(context)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Self-attention
        x2 = self.norm1(x)
        x = x + self.dropout(self.attn(x2, mask))
        
        # Feed-forward
        x2 = self.norm2(x)
        x = x + self.dropout(self.ff(x2))
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(self.create_positional_encoding(max_seq_length, d_model))
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def create_positional_encoding(self, max_seq_length, d_model):
        pos_encoding = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding
        
    def forward(self, x, mask=None):
        seq_length = x.size(1)
        x = self.embedding(x)
        x = x + self.pos_encoding[:seq_length].unsqueeze(0)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.norm(x)
        return self.fc(x)

def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss))

def train_epoch(model, dataloader, optimizer, scaler, scheduler, accumulation_steps=2):
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    optimizer.zero_grad(set_to_none=True)
    
    for i, (src, tgt) in enumerate(progress_bar):
        src, tgt = src.cuda(non_blocking=True), tgt.cuda(non_blocking=True)
        batch_size, seq_length = src.size()
        
        causal_mask = (torch.triu(torch.ones(batch_size, seq_length, seq_length), diagonal=1) == 1).cuda()
        
        with torch.amp.autocast('cuda'):
            output = model(src, causal_mask)
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                tgt.view(-1),
                ignore_index=0,
                label_smoothing=0.15  
            ) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * accumulation_steps
        perplexity = calculate_perplexity(loss.item() * accumulation_steps)
        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'ppl': f'{perplexity:.2f}',
            'lr': f'{current_lr:.6f}'
        })
    
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    
    for src, tgt in tqdm(dataloader, desc='Evaluating'):
        src, tgt = src.cuda(), tgt.cuda()
        batch_size, seq_length = src.size()
        
        causal_mask = (torch.triu(torch.ones(batch_size, seq_length, seq_length), diagonal=1) == 1).cuda()
        
        with torch.cuda.amp.autocast():
            output = model(src, causal_mask)
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                tgt.view(-1),
                ignore_index=0
            )
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0):
    model.eval()
    
    # Encode prompt
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Break if EOS token
            if next_token.item() == tokenizer.eos_token:  
                break
                
            # Append to input
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist(), skip_special=True)

def main():
    # Set up devices
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        use_multi_gpu = True
    else:
        print("Using single GPU")
        use_multi_gpu = False

    # Model configuration
    config = {
        'd_model': 384,          
        'num_heads': 6,          
        'num_layers': 6,         
        'd_ff': 1536,           
        'dropout': 0.2,
        'max_seq_length': 256
    }

    # Training configuration
    batch_size = 8 if use_multi_gpu else 4
    accumulation_steps = 4
    learning_rate = 1e-3
    num_epochs = 30
    val_split = 0.1

    # Load and preprocess data
    print("Loading data...")
    # Load the dataset from the CSV file
    df = pd.read_csv("Shakespeare_data.csv")

    # Remove rows with missing dialogue
    df = df[df['PlayerLine'].notna()]

    # Combine character names with their lines
    texts = df.apply(lambda row: f"{row['Player']}: {row['PlayerLine']}", axis=1).tolist()

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = SubwordTokenizer()
    config['vocab_size'] = tokenizer.vocab_size

    # Split data
    val_size = int(len(texts) * val_split)
    train_texts = texts[:-val_size]
    val_texts = texts[-val_size:]

    # Create datasets
    print("Creating datasets...")
    train_dataset = ShakespeareDataset(train_texts, tokenizer, config['max_seq_length'])
    val_dataset = ShakespeareDataset(val_texts, tokenizer, config['max_seq_length'])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

    # Initialize model
    print("Initializing model...")
    model = TransformerDecoder(**config)
    
    if use_multi_gpu:
        model = nn.DataParallel(model.cuda())
    else:
        model = model.cuda()
    
    # Initialize optimizer, scheduler, and scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.1
    )

    # More appropriate learning rate schedule
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader) // accumulation_steps,
        pct_start=0.2,
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos'
    )
    scaler = GradScaler()

    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, scheduler, accumulation_steps)
        train_losses.append(train_loss)
        train_ppl = calculate_perplexity(train_loss)
        
        # Validation
        val_loss = evaluate(model, val_loader)
        val_losses.append(val_loss)
        val_ppl = calculate_perplexity(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Perplexity: {train_ppl:.2f}")
        print(f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_ppl:.2f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if use_multi_gpu else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'best_model.pt')
        
        # Generate sample text
        print("\nSample generation:")
        prompt = "HAMLET: "
        model_for_generation = model.module if use_multi_gpu else model
        generated_text = generate_text(model_for_generation, tokenizer, prompt, max_length=100)
        print(generated_text)

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == "__main__":
    # Initialize distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    main()