import pandas as pd
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from rdkit import Chem
from rdkit.Chem import RDKFingerprint, DataStructs
from flask import Flask, request, render_template
import pickle
import os

# MLX uses GPU by default on Apple Silicon; no explicit device check available
print("MLX initialized (GPU should be used on Apple Silicon)")

# --- Data Preprocessing ---
rsmi_file = "data/1976_Sep2016_USPTOgrants_smiles.rsmi"
csv_file = "data/uspto_reactions.csv"

# Load data
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} reactions from {csv_file}")
else:
    # Handle data creation (simplified for this fix)
    print(f"Error: {csv_file} not found.")
    exit(1)

# Sample a smaller subset of the data to avoid memory issues
# Adjust this number based on your system's capabilities
MAX_SAMPLES = 10000
if len(df) > MAX_SAMPLES:
    print(f"Sampling {MAX_SAMPLES} reactions from dataset of {len(df)} to avoid memory issues")
    df = df.sample(MAX_SAMPLES, random_state=42)

# Add validation to ensure we have data
if len(df) == 0:
    raise ValueError("Dataset is empty. Please provide valid reaction data.")

reaction_sequences = [f"{r} >> {p}" for r, p in zip(df['reactants'], df['products'])]

# Character-level tokenization
chars = sorted(set("".join(reaction_sequences)))
char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}
idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}
vocab_size = len(chars) + 1

def encode(text):
    return [char_to_idx.get(c, 0) for c in text]

def decode(tokens):
    return ''.join(idx_to_char.get(t, '') for t in tokens if t != 0)

# Calculate maximum sequence length
max_length = min(max(len(seq) for seq in reaction_sequences), 512)
print(f"Maximum sequence length: {max_length}")

# Process in smaller batches to avoid memory issues
BATCH_SIZE = 100
num_batches = (len(reaction_sequences) + BATCH_SIZE - 1) // BATCH_SIZE

# Create empty arrays for X_input and y_output
all_X_input = []
all_y_output = []

for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(reaction_sequences))
    print(f"Processing batch {batch_idx+1}/{num_batches} (samples {start_idx} to {end_idx})")
    
    batch_sequences = reaction_sequences[start_idx:end_idx]
    
    # Encode sequences
    encoded_sequences = [encode(seq)[:max_length] for seq in batch_sequences]
    
    # Pad sequences manually to avoid GPU memory issues
    padded_data = []
    for seq in encoded_sequences:
        # Create padded array with zeros
        padded_seq = [0] * max_length
        # Copy sequence data
        for i, token in enumerate(seq):
            if i < max_length:
                padded_seq[i] = token
        padded_data.append(padded_seq)
    
    # Convert to MLX arrays
    batch_X = mx.array(padded_data)
    batch_X_input = batch_X[:, :-1]
    batch_y_output = batch_X[:, 1:]
    
    all_X_input.append(batch_X_input)
    all_y_output.append(batch_y_output)
    
    # Free memory
    del batch_sequences, encoded_sequences, padded_data, batch_X

# Stack all batches (or leave as separate batches if still causing memory issues)
try:
    X_input = mx.concatenate(all_X_input, axis=0)
    y_output = mx.concatenate(all_y_output, axis=0)
    print(f"Successfully combined all batches")
except Exception as e:
    print(f"Warning: Could not combine batches due to: {e}")
    print("Will use batched training instead")
    X_input = all_X_input
    y_output = all_y_output

print(f"Data Preprocessing Complete: X_input samples: {len(df)}, vocab_size: {vocab_size}, max_length: {max_length}")

# Save tokenizer and sequence info for later use
tokenizer_data = {
    "char_to_idx": char_to_idx,
    "idx_to_char": idx_to_char,
    "vocab_size": vocab_size,
    "max_length": max_length
}

os.makedirs("models", exist_ok=True)
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer_data, f)
print("Saved tokenizer data to models/tokenizer.pkl")

# Define model (to be added)
class ReactionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = [
            nn.TransformerDecoderLayer(
                d_model=d_model,
                num_heads=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ]
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.nhead = nhead
        self.max_length = max_length-1  # Adjust for input/output shift

    def generate_causal_mask(self, sz):
        mask = mx.ones((sz, sz))
        mask = mx.tril(mask)
        return mask

    def __call__(self, x):
        batch_size, seq_len = x.shape
        
        # Embedding
        embedded = self.embedding(x) * mx.sqrt(mx.array(self.d_model))
        
        # Positional encoding
        positions = mx.arange(seq_len)
        div_term = mx.exp(mx.arange(0, self.d_model, 2) * (-mx.log(mx.array(10000.0)) / self.d_model))
        
        pe = mx.zeros((seq_len, self.d_model))
        pe = pe.at[:, 0::2].set(mx.sin(positions.reshape(-1, 1) * div_term))
        pe = pe.at[:, 1::2].set(mx.cos(positions.reshape(-1, 1) * div_term))
        
        embedded = embedded + pe.reshape(1, seq_len, self.d_model)
        
        # Apply transformer layers with causal mask
        mask = self.generate_causal_mask(seq_len)
        output = embedded
        for layer in self.layers:
            output = layer(output, mask=mask)
            
        logits = self.fc(output)
        return logits

# Initialize model
model = ReactionTransformer(vocab_size, d_model=64, nhead=4, num_layers=2)
optimizer = optim.Adam(learning_rate=0.001)
print(f"Model Initialized: d_model={model.d_model}, nhead=4, num_layers=2")

# --- Training Function ---
def compute_loss(params, model, x, y):
    model.update(params)
    logits = model(x)
    loss = nn.losses.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction='mean')
    return loss

def train_step(model, x, y):
    loss_and_grad_fn = mx.value_and_grad(compute_loss)
    loss, grads = loss_and_grad_fn(model.parameters(), model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss

# --- Training Loop ---
batch_size = 32
num_epochs = 10
num_samples = X_input.shape[0]
num_batches = max(1, num_samples // batch_size)

# Create models directory
os.makedirs("models", exist_ok=True)

print(f"Starting Training: batch_size={batch_size}, num_epochs={num_epochs}, num_batches={num_batches}")

for epoch in range(num_epochs):
    total_loss = 0
    # Shuffle data
    indices = np.random.permutation(num_samples)
    X_shuffled = X_input[indices]
    y_shuffled = y_output[indices]
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        x_batch = X_shuffled[i:end_idx]
        y_batch = y_shuffled[i:end_idx]
        
        if x_batch.shape[0] == 0:
            continue
            
        loss = train_step(model, x_batch, y_batch)
        total_loss += loss.item()
        
        batch_num = i // batch_size + 1
        if batch_num % 10 == 0 or batch_num == num_batches:
            print(f"Epoch {epoch+1}, Batch {batch_num}/{num_batches}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save model checkpoint
    with open(f"models/transformer_epoch_{epoch+1}.pkl", "wb") as f:
        pickle.dump(model.parameters(), f)

# Save final model
with open("models/transformer.pkl", "wb") as f:
    pickle.dump(model.parameters(), f)
print("Model saved to models/transformer.pkl")

# --- Generation ---
def generate_pathway(start_reactants, target_product, max_steps=5, temperature=0.8, top_k=5):
    current = start_reactants
    pathway = [current]
    
    for _ in range(max_steps):
        seq = encode(current)
        # Truncate if too long
        if len(seq) >= max_length-1:
            seq = seq[:max_length-2]
            
        # Pad sequence
        padded = mx.array(seq)
        padded = mx.pad(padded, pad_value=0, shape=(max_length-1,))
        
        # Get model prediction
        model_input = padded[None, :]  # Add batch dimension
        logits = model(model_input)[0, -1, :] / temperature
        
        # Apply top-k sampling
        if top_k > 0:
            # Get top-k values and indices
            values, indices = mx.topk(logits, k=min(top_k, logits.shape[0]))
            mask = mx.full_like(logits, float('-inf'))
            mask = mask.at[indices].set(logits[indices])
            logits = mask
            
        # Sample from distribution
        probs = mx.softmax(logits)
        next_token = mx.argmax(probs).item()  # Get highest probability token
        
        # Add predicted token to current sequence
        current += idx_to_char.get(next_token, '')
        
        # Check if we have a complete reaction
        if ">>" in current:
            pathway.append(current)
            parts = current.split(">>")
            if len(parts) >= 2:
                # Current becomes the product for the next step
                current = parts[1].strip()
                
            # Check if we've reached the target
            if current == target_product:
                break
                
            # Try using RDKit similarity as a stopping condition
            try:
                current_mol = Chem.MolFromSmiles(current)
                target_mol = Chem.MolFromSmiles(target_product)
                
                if current_mol and target_mol:
                    current_fp = RDKFingerprint(current_mol)
                    target_fp = RDKFingerprint(target_mol)
                    similarity = DataStructs.TanimotoSimilarity(current_fp, target_fp)
                    
                    if similarity > 0.9:  # High similarity threshold
                        break
            except Exception as e:
                print(f"Error computing similarity: {e}")
                
        # Prevent sequences from growing too long
        if len(current) > max_length:
            break
            
    return pathway

def is_valid_reaction(reaction):
    """Check if a reaction string represents a valid chemical reaction."""
    if ">>" not in reaction:
        return False
        
    try:
        reactants, products = reaction.split(">>", 1)
        
        # Check reactants
        for r in reactants.split("."):
            if not r.strip():
                continue
            r_mol = Chem.MolFromSmiles(r.strip())
            if r_mol is None:
                return False
                
        # Check products
        for p in products.split("."):
            if not p.strip():
                continue
            p_mol = Chem.MolFromSmiles(p.strip())
            if p_mol is None:
                return False
                
        return True
    except Exception:
        return False

def validate_smiles(smiles):
    """Validate SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def pathway_pipeline(start_reactants, target_product, max_attempts=3):
    """Generate a synthetic pathway with validation and multiple attempts."""
    # Validate input
    if not validate_smiles(start_reactants):
        return ["Invalid starting reactants SMILES"]
    if not validate_smiles(target_product):
        return ["Invalid target product SMILES"]
        
    best_pathway = []
    
    for attempt in range(max_attempts):
        pathway = generate_pathway(
            start_reactants, 
            target_product,
            max_steps=5,
            temperature=0.8 - (attempt * 0.2),  # Decrease temperature with each attempt
            top_k=5
        )
        
        valid_steps = [step for step in pathway if is_valid_reaction(step)]
        
        if len(valid_steps) > len(best_pathway):
            best_pathway = valid_steps
            
    return best_pathway if best_pathway else ["No valid pathway found"]

# --- Flask Web Application ---
app = Flask(__name__)

# Create templates directory and template files if they don't exist
os.makedirs("templates", exist_ok=True)

if not os.path.exists("templates/index.html"):
    with open("templates/index.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Chemical Reaction Pathway Generator</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="text"] { width: 100%; padding: 8px; font-family: monospace; }
        button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Chemical Reaction Pathway Generator</h1>
        <form method="POST">
            <div class="form-group">
                <label for="start">Starting Reactants (SMILES):</label>
                <input type="text" id="start" name="start" placeholder="CC(=O)OC1=CC=CC=C1C(=O)O" required>
            </div>
            <div class="form-group">
                <label for="target">Target Product (SMILES):</label>
                <input type="text" id="target" name="target" placeholder="O=C(O)C1=CC=CC=C1O" required>
            </div>
            <button type="submit">Generate Pathway</button>
        </form>
    </div>
</body>
</html>
        """)
    print("Created templates/index.html")

if not os.path.exists("templates/result.html"):
    with open("templates/result.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Pathway Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .pathway-step { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .reaction { font-family: monospace; margin-bottom: 10px; font-size: 16px; }
        .back-button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; text-decoration: none; display: inline-block; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Generated Reaction Pathway</h1>
        
        {% if pathway and pathway[0] != "No valid pathway found" %}
            {% for step in pathway %}
                <div class="pathway-step">
                    <div class="reaction">{{ step }}</div>
                </div>
            {% endfor %}
        {% else %}
            <div class="pathway-step">
                <p>{{ pathway[0] }}</p>
            </div>
        {% endif %}
        
        <a href="/" class="back-button">Generate Another Pathway</a>
    </div>
</body>
</html>
        """)
    print("Created templates/result.html")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        start = request.form['start']
        target = request.form['target']
        pathway = pathway_pipeline(start, target)
        return render_template('result.html', pathway=pathway)
    return render_template('index.html')

if __name__ == '__main__':
    try:
        # Try to load a pre-trained model
        with open("models/transformer.pkl", "rb") as f:
            params = pickle.load(f)
            model.update(params)
        print("Loaded pre-trained model from models/transformer.pkl")
    except FileNotFoundError:
        print("No pre-trained model found. Using the newly trained model.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using the newly trained model.")
    
    # Start the Flask app
    print("Starting Flask web application. Navigate to http://127.0.0.1:5000 in your browser.")
    app.run(debug=True)