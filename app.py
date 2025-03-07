import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
from flask import Flask, request, render_template
import os

# Set device (MPS for Apple Silicon, else CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Preprocessing ---
rsmi_file = "data/1976_Sep2016_USPTOgrants_smiles.rsmi"
csv_file = "data/uspto_reactions.csv"

# Process raw .rsmi file into CSV if not already done
if not os.path.exists(csv_file):
    with open(rsmi_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    data = {"reactants": [], "products": []}
    for line in lines:
        if ">>" in line:
            parts = line.split(">>")
            if len(parts) == 2:
                reactants, products = parts[0].strip(), parts[1].strip()
                if Chem.MolFromSmiles(reactants) and Chem.MolFromSmiles(products):
                    data["reactants"].append(reactants)
                    data["products"].append(products)
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Converted {rsmi_file} to {csv_file} with {len(df)} reactions")
else:
    df = pd.read_csv(csv_file)

# Combine reactants and products into sequences
reaction_sequences = [f"{r} >> {p}" for r, p in zip(df['reactants'], df['products'])]

# Character-level tokenization
chars = sorted(set("".join(reaction_sequences)))
char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}  # 0 for padding
idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}
vocab_size = len(chars) + 1

def encode(text):
    return [char_to_idx.get(c, 0) for c in text]

def decode(tokens):
    return ''.join(idx_to_char.get(t, '') for t in tokens if t != 0)

# Encode and pad sequences
max_length = min(max(len(seq) for seq in reaction_sequences), 512)  # Cap for efficiency
sequences = [torch.tensor(encode(seq), dtype=torch.long) for seq in reaction_sequences]
X = pad_sequence(sequences, batch_first=True, padding_value=0)[:, :max_length]
X_input = X[:, :-1].to(device)
y_output = X[:, 1:].to(device)

print(f"Data Preprocessing Complete: X_input shape: {X_input.shape}, y_output shape: {y_output.shape}, vocab_size: {vocab_size}, max_length: {max_length}")

# --- Model Definition ---
class ReactionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_length, d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def generate_causal_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool().to(device)

    def forward(self, x):
        batch_size, seq_len = x.shape
        embedded = self.embedding(x) * (self.d_model ** 0.5)
        embedded = embedded + self.pos_encoder[:, :seq_len, :]
        mask = self.generate_causal_mask(seq_len)
        # Use embedded as both target and memory for decoder-only behavior
        output = self.transformer_decoder(tgt=embedded, memory=embedded, tgt_mask=mask)
        logits = self.fc(output)
        return logits

model = ReactionTransformer(vocab_size, d_model=64, nhead=4, num_layers=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model Initialized: d_model={model.d_model}, nhead=4, num_layers=2")

# --- Training ---
def train_step(model, x, y, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

batch_size = 32
num_epochs = 10
num_batches = len(X_input) // batch_size
print(f"Starting Training: batch_size={batch_size}, num_epochs={num_epochs}, num_batches={num_batches}")

os.makedirs("models", exist_ok=True)

for epoch in range(num_epochs):
    total_loss = 0
    indices = torch.randperm(len(X_input))
    X_shuffled = X_input[indices]
    y_shuffled = y_output[indices]
    
    for i in range(0, len(X_shuffled), batch_size):
        x_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        loss = train_step(model, x_batch, y_batch, optimizer)
        total_loss += loss
        if (i // batch_size) % 100 == 0:  # Print less frequently
            print(f"Epoch {epoch+1}, Batch {i//batch_size + 1}/{num_batches}, Loss: {loss:.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), f"models/transformer_epoch_{epoch+1}.pth")

torch.save(model.state_dict(), "models/transformer.pth")
print("Model saved to models/transformer.pth")

# Load model if exists
try:
    model.load_state_dict(torch.load("models/transformer.pth"))
    print("Loaded model from models/transformer.pth")
except FileNotFoundError:
    print("No pre-trained model found.")

# --- Generation ---
def generate_pathway(start_reactants, target_product, max_steps=5, temperature=0.8, top_k=5):
    model.eval()
    current = start_reactants
    pathway = [current]
    
    with torch.no_grad():
        for _ in range(max_steps):
            seq = torch.tensor(encode(current), dtype=torch.long, device=device).unsqueeze(0)
            seq = pad_sequence([seq], batch_first=True, padding_value=0)[:, :max_length-1]
            logits = model(seq)
            next_logits = logits[0, -1, :] / temperature
            
            if top_k > 0:
                values, indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = torch.full_like(next_logits, -float('inf'))
                next_logits[indices] = values
            
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            current += idx_to_char.get(next_token.item(), '')
            
            if ">>" in current:
                pathway.append(current)
                parts = current.split(">>")
                if len(parts) >= 2:
                    current = parts[1].strip()
                if current == target_product or (
                    Chem.MolFromSmiles(current) and Chem.MolFromSmiles(target_product) and
                    Chem.TanimotoSimilarity(
                        Chem.RDKFingerprint(Chem.MolFromSmiles(current)),
                        Chem.RDKFingerprint(Chem.MolFromSmiles(target_product))
                    ) > 0.9
                ):
                    break
            if len(current) > max_length:
                break
    
    print(f"Generated pathway: {pathway}")
    return pathway

def is_valid_reaction(reaction):
    if ">>" not in reaction:
        return False
    reactants, product = reaction.split(">>")
    r_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
    p_mol = Chem.MolFromSmiles(product.strip())
    return all(m is not None for m in r_mols) and p_mol is not None

def pathway_pipeline(start_reactants, target_product):
    pathway = generate_pathway(start_reactants, target_product)
    valid_pathway = [step for step in pathway if is_valid_reaction(step)]
    return valid_pathway if valid_pathway else ["No valid pathway found"]

# --- Flask App ---
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        start = request.form['start']
        target = request.form['target']
        pathway = pathway_pipeline(start, target)
        return render_template('result.html', pathway=pathway)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)