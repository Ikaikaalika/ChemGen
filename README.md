# ChemGen

Below is a well-structured `README.md` for the **Reaction Pathway Generator with MLX Transformer, or ChemGen** project. It provides an overview, setup instructions, usage details, and additional context to make the project accessible to others (e.g., for a portfolio or collaboration). Since this is an MLX-based project, I’ll emphasize the Apple Silicon requirement and include clear steps for running it.

---

# Reaction Pathway Generator with MLX Transformer

## Overview
This project uses generative AI to predict plausible chemical reaction pathways from starting reactants to a target product, represented as SMILES strings. It leverages **MLX**, Apple’s machine learning framework optimized for Apple Silicon, with a transformer model to generate reaction sequences autoregressively. The system is designed for applications in synthetic chemistry, such as drug synthesis or materials design.

### Key Features
- **Input**: Starting reactants and target product (SMILES strings, e.g., "CCO" → "CC(=O)O").
- **Output**: A sequence of reaction steps (e.g., "CCO >> CC(=O)O").
- **Model**: Transformer-based architecture from `mlx.nn`.
- **Validation**: Chemical validity checked with RDKit.
- **Deployment**: Web app interface via Flask.

### Goals
- Generate chemically valid reaction steps (>70% validity rate).
- Match the final product to the target (or achieve high molecular similarity).

---

## Prerequisites
This project requires an **Apple Silicon Mac** (M1, M2, M3, etc.) due to MLX’s hardware-specific optimization.

### Dependencies
- Python 3.8+
- MLX (`pip install mlx`)
- RDKit (`conda install -c conda-forge rdkit` or `pip install rdkit`)
- Flask (`pip install flask`)
- NumPy (`pip install numpy`)
- TensorFlow (for tokenization only, `pip install tensorflow`)

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Dataset
- Uses the USPTO reaction dataset (e.g., from Kaggle or figshare).
- Expected format: CSV with `reactants` and `products` columns (SMILES strings).

---

## Project Structure
```
reaction_pathway_generator_mlx_transformer/
├── data/                  # USPTO dataset (e.g., uspto_reactions.csv)
├── models/                # Trained MLX model (saved manually if needed)
├── templates/             # HTML files for Flask app
│   ├── index.html        # Input form
│   └── result.html       # Pathway output
├── app.py                # Flask app and generation pipeline
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/reaction_pathway_generator_mlx_transformer.git
   cd reaction_pathway_generator_mlx_transformer
   ```

2. **Prepare the Dataset**:
   - Place `uspto_reactions.csv` in the `data/` folder.
   - Alternatively, modify `app.py` to point to your dataset location.

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Environment**:
   - Ensure you’re on an Apple Silicon Mac.
   - Run `python -c "import mlx.core; print(mlx.core.metal.is_available())"` to confirm MLX can use the GPU.

---

## Training the Model
1. **Preprocess Data**:
   - The script tokenizes SMILES strings and converts them to MLX arrays.

2. **Train the Transformer**:
   - Edit `app.py` to uncomment the training loop if not pre-trained:
     ```python
     for epoch in range(20):
         for i in range(0, len(X_input), batch_size):
             x_batch = X_input[i:i+batch_size]
             y_batch = y_output[i:i+batch_size]
             loss = train_step(model, x_batch, y_batch)
         print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
     ```
   - Run: `python app.py` (stop after training, then comment out again).

3. **Save Model** (Optional):
   - MLX doesn’t have a built-in save method yet, so you may need to pickle the model parameters:
     ```python
     import pickle
     with open("models/transformer.pkl", "wb") as f:
         pickle.dump(model.parameters(), f)
     ```

---

## Usage
1. **Run the Web App**:
   ```bash
   python app.py
   ```
   - Open your browser to `http://127.0.0.1:5000`.

2. **Generate a Pathway**:
   - Enter starting reactants (e.g., "CCO" for ethanol).
   - Enter the target product (e.g., "CC(=O)O" for acetic acid).
   - Click "Generate Pathway" to see the predicted reaction steps.

3. **Example Output**:
   ```
   Generated Pathway:
   - CCO
   - CCO >> CC(=O)O
   ```

---

## Evaluation
- **Validity**: Checked with RDKit (`is_valid_reaction` function).
- **Target Match**: Final product compared to target using Tanimoto similarity.
- Metrics are printed during generation (e.g., validity rate, target reached).

---

## Visualization
- Reaction steps are visualized as 2D molecular structures using RDKit:
  ```python
  from rdkit.Chem import Draw
  mols = [Chem.MolFromSmiles(s.split(">>")[1].strip()) for s in pathway if ">>" in s]
  Draw.MolsToGridImage(mols, molsPerRow=2).save("pathway.png")
  ```
- Output saved as `pathway.png`.

---

## Limitations
- **MLX**: Requires Apple Silicon; not portable to other platforms.
- **Single-Step Bias**: Trained on single-step reactions; multi-step pathways are generated iteratively.
- **Conditions**: Reaction conditions (e.g., catalysts) not included yet.

---

## Future Improvements
- **Multi-Step Training**: Use datasets with full pathways (e.g., Reaxys).
- **Conditions Prediction**: Extend the model to generate reagents/catalysts.
- **Retrosynthesis**: Reverse the process for target-to-reactant prediction.
- **Model Saving**: Implement a robust save/load mechanism for MLX models.

---

## Acknowledgments
- Built with [MLX](https://github.com/ml-explore/mlx) by Apple.
- Chemical validation powered by [RDKit](https://www.rdkit.org/).
- Inspired by retrosynthesis and reaction prediction research.

---

## License
MIT License - feel free to use, modify, and distribute.

---

This README provides a clear roadmap for anyone to understand, set up, and extend your project. It balances technical detail with accessibility, making it suitable for GitHub or a portfolio. Want to adjust anything—like adding more usage examples or tweaking the tone? Let me know!