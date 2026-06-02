# PyTorch Cheat Sheet for RNN, MLP, Transformers & HuggingFace

## Table of Contents
1. [PyTorch Basics](#pytorch-basics)
2. [Data Loading](#data-loading)
3. [MLP (Multi-Layer Perceptron)](#mlp)
4. [RNN (Recurrent Neural Network)](#rnn)
5. [Transformers](#transformers)
6. [HuggingFace Model Modification](#huggingface)
7. [Training Loop](#training-loop)
8. [Useful Patterns](#useful-patterns)

---

## PyTorch Basics

### Imports
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
```

### Device Setup
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
data, target = data.to(device), target.to(device)
```

### Tensor Operations
```python
# Create tensors
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
x = torch.randn(64, 10)           # Random normal
x = torch.zeros(5, 5)              # Zeros
x = torch.ones(5, 5)               # Ones
x = torch.eye(5)                   # Identity

# Tensor properties
x.shape        # Shape
x.dtype        # Data type
x.device       # Device
x.requires_grad  # Track gradients

# Operations
x.view(-1, 10)        # Reshape
x.transpose(0, 1)     # Transpose
torch.cat([x, y], dim=1)  # Concatenate
torch.stack([x, y])      # Stack
x @ y                 # Matrix multiply
```

---

## Data Loading

### Custom Dataset
```python
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# Usage
dataset = CustomDataset(X_train, y_train)
```

### TensorDataset (for numpy/arrays)
```python
from torch.utils.data import TensorDataset

# Convert numpy arrays to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Create dataset
dataset = TensorDataset(X_tensor, y_tensor)
```

### Image Data Loading (MNIST/CIFAR)
```python
from torchvision import datasets, transforms

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Mean, Std for MNIST
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For RGB
    # transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
```

### DataLoader with Custom Dataset
```python
train_loader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True,  # Faster GPU transfer
    drop_last=True    # Drop incomplete batches
)
```

### Iterate through DataLoader
```python
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Training code here
```

### Text Data Loading (for NLP)
```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_texts), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Convert text to tensor
text_pipeline = lambda x: vocab(tokenizer(x))
tensor = torch.tensor(text_pipeline(text), dtype=torch.long)
```

---

## MLP

### Basic MLP
```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(MLP, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

# Usage
model = MLP(input_size=784, hidden_size=128, output_size=10)
```

### MLP with Batch Normalization
```python
class MLP_BN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_BN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
```

### MLP for Regression
```python
class MLP_Regressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP_Regressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Single output
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)  # Remove last dim for scalar output
```

---

## RNN

### Basic RNN
```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # (batch, seq_len, input_size)
            nonlinearity='relu',  # 'tanh' or 'relu'
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass
        out, _ = self.rnn(x, h0)  # out: (batch, seq_len, hidden_size)
        
        # Decode the hidden state
        out = self.fc(out[:, -1, :])  # Use last time step
        return out
```

### LSTM
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use last hidden state
        out = self.fc(out[:, -1, :])
        return out
```

### Bidirectional LSTM
```python
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # Two directions
        )
        # Hidden size doubles for bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
```

### GRU
```python
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
```

### RNN for Sequence to Sequence
```python
class Seq2SeqRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Seq2SeqRNN, self).__init__()
        self.encoder = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.RNN(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, target_len):
        # Encoder
        _, hidden = self.encoder(x)
        
        # Decoder input (start with zeros)
        decoder_input = torch.zeros(x.size(0), target_len, dtype=torch.float32).to(x.device)
        
        # Decoder
        output, _ = self.decoder(decoder_input, hidden)
        output = self.fc(output)
        return output
```

---

## Transformers

### Self-Attention Block
```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, embed_size)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_probs, V)
        return out
```

### Transformer Encoder Layer
```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, src):
        # Self-attention
        src2, _ = self.attention(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
```

### Complete Transformer Model
```python
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, num_classes, seq_length=128):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, seq_length)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size*4
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_size)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (seq_len, batch, embed_size)
        
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch, seq_len, embed_size)
        x = x[:, 0, :]  # Take first token (CLS token pattern)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### Transformer Decoder
```python
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size*4
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, tgt, memory):
        tgt = self.embedding(tgt)
        tgt = self.pos_encoding(tgt)
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)
        
        output = self.transformer(tgt, memory)
        output = output.transpose(0, 1)
        output = self.fc(output)
        return output
```

---

## HuggingFace Model Modification

### Install and Import
```python
# pip install transformers datasets
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    PretrainedConfig
)
from datasets import load_dataset
import torch.nn as nn
```

### Load Pre-trained Model
```python
# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# For classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=10  # Custom number of classes
)

# For sequence to sequence
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
```

### Modify Model Architecture

#### Add Classification Head
```python
from transformers import BertForSequenceClassification, BertConfig

# Load config and modify
config = BertConfig.from_pretrained("bert-base-uncased")
config.num_labels = 5  # Custom classes
config.hidden_dropout_prob = 0.3

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    config=config
)
```

#### Freeze Base Model (Feature Extraction)
```python
# Freeze all BERT layers
for param in model.bert.parameters():
    param.requires_grad = False

# Only train classification head
for param in model.classifier.parameters():
    param.requires_grad = True
```

#### Unfreeze Specific Layers
```python
# Unfreeze last N layers
n = 4  # Unfreeze last 4 layers
for i, layer in enumerate(model.bert.encoder.layer):
    if i >= len(model.bert.encoder.layer) - n:
        for param in layer.parameters():
            param.requires_grad = True
    else:
        for param in layer.parameters():
            param.requires_grad = False
```

#### Custom Model on Top of BERT
```python
from transformers import BertModel

class CustomBERT(nn.Module):
    def __init__(self, model_name, num_classes, hidden_size=768):
        super(CustomBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
        # Custom head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# Usage
model = CustomBERT("bert-base-uncased", num_classes=10)
```

#### Modify Embeddings
```python
from transformers import BertForMaskedLM

model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Add custom tokens
tokenizer.add_tokens(["[SPECIAL1]", "[SPECIAL2]"])
model.resize_token_embeddings(len(tokenizer))  # Update embedding layer
```

#### Modify Attention Mechanism
```python
from transformers import BertAttention

class CustomAttention(BertAttention):
    def forward(self, ...):
        # Custom attention logic
        # Call parent or override completely
        return super().forward(...)

# Replace attention in model
for layer in model.bert.encoder.layer:
    layer.attention = CustomAttention(layer.attention.config)
```

#### Adapter Pattern (Lightweight Modification)
```python
from transformers import BertAdapterConfig, BertModel

# Add adapter to BERT
config = BertConfig.from_pretrained("bert-base-uncased")
adapter_config = BertAdapterConfig(mh_adapter=True, output_adapter=True)
config.adapters = {"custom": adapter_config}

model = BertModel.from_pretrained("bert-base-uncased", config=config)
model.add_adapter("custom")
model.train_adapter("custom")

# Use adapter in forward
outputs = model(input_ids, adapter_names=["custom"])
```

### Load Custom Dataset with Tokenizer
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Create PyTorch DataLoader
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(
    tokenized_dataset["train"],
    batch_size=16,
    shuffle=True
)
```

### Save and Load Custom Model
```python
# Save entire model
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# Load custom model
model = CustomBERT.from_pretrained("./my_model")

# Save only state dict
torch.save(model.state_dict(), "custom_model.pth")

# Load state dict
model = CustomBERT("bert-base-uncased", num_classes=10)
model.load_state_dict(torch.load("custom_model.pth"))
```

---

## Training Loop

### Basic Training Loop
```python
def train(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 100 == 99:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}")
                running_loss = 0.0
```

### Training with Validation
```python
def train_with_validation(model, train_loader, test_loader, criterion, optimizer, epochs, device):
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
    
    print(f"Best Accuracy: {best_accuracy:.2f}%")
```

### Training with Learning Rate Scheduler
```python
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

# Step LR scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Reduce on plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)

# Usage in training loop
for epoch in range(epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)  # For ReduceLROnPlateau
    # scheduler.step()  # For StepLR, CosineAnnealingLR
```

### Mixed Precision Training (GPU)
```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for epoch in range(epochs):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Automatic mixed precision
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Scale loss and backprop
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### Gradient Clipping
```python
# Clip gradients to prevent explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=5)
for epoch in range(epochs):
    val_loss = validate(...)
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping!")
        break
```

---

## Useful Patterns

### Weight Initialization
```python
# Xavier/Glorot initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.xavier_uniform_(m.weight_hh_l0)
        nn.init.zeros_(m.bias_ih_l0)
        nn.init.zeros_(m.bias_hh_l0)

model.apply(init_weights)
```

### Model Summary
```python
from torchsummary import summary

# Print model summary
summary(model, input_size=(1, 28, 28))  # For CNN
summary(model, input_size=(64, 784))    # For MLP (batch, features)
summary(model, input_size=(32, 10, 128)) # For RNN (batch, seq_len, features)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Trainable parameters: {count_parameters(model)}")
```

### Inference Mode
```python
model.eval()
with torch.no_grad():
    # Disable gradient calculation
    outputs = model(inputs)
    predictions = torch.argmax(outputs, dim=1)
```

### Save and Load Models
```python
# Save entire model
torch.save(model, "model.pth")
model = torch.load("model.pth")

# Save state dict (recommended)
torch.save(model.state_dict(), "model_state.pth")
model.load_state_dict(torch.load("model_state.pth"))

# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, "checkpoint.pth")
```

### Custom Loss Functions
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        # label: 1 for similar, 0 for dissimilar
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + 
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
```

### Optimizers Comparison
```python
# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam (most common)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)

# AdamW (fixes weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)

# Adagrad
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# Learning rate finder (requires torch_lr_finder)
# from torch_lr_finder import LRFinder
# lr_finder = LRFinder(model, optimizer, criterion)
# lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
# lr_finder.plot()
# lr_finder.reset()
```

### Data Augmentation
```python
# Image augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Text augmentation (using nlpaug)
# import nlpaug.augmenter.word as naw
# aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
# augmented_text = aug.augment(text)

# Mixup augmentation
class Mixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
```

---

## Quick Start Examples

### MLP for MNIST
```python
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.net(x)

# Training
model = SimpleMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### RNN for Text Classification
```python
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.rnn(x)
        x = hidden[-1]  # Last layer, last time step
        x = self.fc(x)
        return x
```

### Transformer for Text
```python
class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, num_classes):
        super(TextTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(embed_size, num_heads, embed_size*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_size, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)  # (seq_len, batch, embed_size)
        x = self.transformer(x)
        x = x.mean(dim=0)  # Average over sequence
        x = self.fc(x)
        return x
```

---

## Command Reference

| Task | Code |
|------|------|
| Check CUDA | `torch.cuda.is_available()` |
| Move to GPU | `.to(device)` or `.cuda()` |
| Tensor to numpy | `.detach().cpu().numpy()` |
| Gradient check | `torch.autograd.gradcheck` |
| Parameter count | `sum(p.numel() for p in model.parameters())` |
| Model size | `sum(p.numel()*p.element_size() for p in model.parameters()) / 1024 / 1024` |
| Clear cache | `torch.cuda.empty_cache()` |
| Set seed | `torch.manual_seed(42); np.random.seed(42)` |

---

*Created for autumn project - PyTorch Deep Learning Cheat Sheet*
