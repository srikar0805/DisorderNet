# DisorderNet

DisorderNet is a Colab-ready machine learning project for predicting intrinsically disordered protein regions from amino acid sequence. It builds residue-level labels from DisProt disorder annotations, trains a neural sequence model, and evaluates residue predictions with metrics suited to imbalanced biological labels.

## Project Goal

Intrinsically disordered regions (IDRs) are protein segments that do not fold into one stable 3D structure under physiological conditions. Predicting IDRs from sequence is useful because disorder is linked to regulation, signaling, protein interactions, and disease biology.

This project asks:

> Given a protein sequence, can we predict for each residue whether it belongs to an intrinsically disordered region?

## What The Notebook Does

- Downloads curated disorder annotations from DisProt when the API is reachable.
- Falls back to CAID reference data before using the tiny embedded demo dataset.
- Converts annotated disorder spans into residue-level binary labels.
- Splits data at the protein level to avoid residue leakage across train, validation, and test sets.
- Encodes residues using amino acid identity plus physicochemical features by default.
- Optionally uses ESM-2 protein language model embeddings for a stronger experiment.
- Trains `HybridDisorderNet`, a CNN + Transformer + BiLSTM residue classifier in PyTorch.
- Reports APS, ROC-AUC, F1, MCC, balanced accuracy, specificity, and Fmax.
- Exports plots, per-protein metrics, a confusion matrix, and a custom sequence prediction trace.

## Repository Contents

- `IDR_Project_28.ipynb`: Main end-to-end notebook.
- `requirements.txt`: Python packages used by the notebook.
- `LICENSE`: Repository license.

Generated files are written to:

- `checkpoints/`: Best model checkpoint.
- `outputs/plots/`: Training curves and evaluation figures.
- `outputs/tables/`: Metrics tables and per-protein results.

These generated folders are ignored by Git so large outputs do not accidentally enter the repository.

## How To Run

The easiest path is Google Colab:

1. Open `IDR_Project_28.ipynb` in Colab.
2. Runtime -> Change runtime type -> GPU is recommended, especially if enabling ESM-2.
3. Run all cells from top to bottom.
4. Confirm the printed values:
   - `Data mode: disprot_api` or `Data mode: caid_reference`
   - `Live data mode: True`
   - `Usable proteins after cleaning: ...`
   - `ESM-2 embeddings enabled: False` or `True`
5. Use the final test metric table for the submission report.

For a local run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook IDR_Project_28.ipynb
```

## Important Reporting Note

If DisProt is unavailable, the notebook now tries CAID reference disorder labels before falling back to the 10-protein embedded demo. If it prints `Data mode: caid_reference`, that is still a real residue-labeled benchmark-style dataset. If it prints `Data mode: embedded_demo`, the run only proves the pipeline works and should not be used as a final benchmark result.

For the final project submission, rerun with live DisProt data and report:

- Data mode and number of usable proteins after cleaning.
- Whether ESM-2 embeddings were enabled.
- Test APS and ROC-AUC.
- Test F1, MCC, balanced accuracy, and Fmax.
- One or two visual examples of residue-level predictions.

## Model Architecture

The main model is `HybridDisorderNet`. It is designed to combine complementary signals that matter for intrinsic disorder:

- Multi-scale residual CNN branches capture local residue motifs and composition patterns.
- Transformer self-attention captures longer-range dependencies between residues.
- A bidirectional LSTM smooths sequence context from both directions.
- A learned fusion gate combines local, attention-based, and recurrent features for each residue.
- A residue-level classifier outputs one disorder probability per amino acid.

This is stronger than a plain BiLSTM because IDRs are influenced by local amino acid composition, surrounding sequence context, and longer-range protein patterns.

## Training Safeguards

The notebook uses a minimum number of epochs before early stopping and checkpoints on a combined validation score:

```text
0.40 * APS + 0.30 * Fmax + 0.20 * MCC + 0.10 * ROC-AUC
```

This avoids stopping immediately when APS reaches `1.0000` on a small or easy validation split. If validation metrics are near-perfect from epoch 1, report that as a limitation and rely on the held-out test metrics for the final claim.

## Baseline vs. ESM-2 Upgrade

The default feature setup is a fast, reproducible run using sequence-derived residue features. To run the stronger ESM-2 feature version, set this in the configuration cell:

```python
RUN_ESM2_SECTION = True
```

Then rerun the notebook from the ESM-2 initialization cell onward. This downloads a pretrained ESM-2 model, so it requires internet access and more compute.

## Best Next Improvements

1. Add sequence-identity-aware splits with MMseqs2 or CD-HIT so homologous proteins do not appear across train/test splits.
2. Compare three variants in a small results table: old Conv+BiLSTM baseline, HybridDisorderNet with hand-built features, and HybridDisorderNet with ESM-2.
3. Cache downloaded DisProt data and extracted ESM-2 embeddings so reruns are faster.
4. Add a short error analysis section showing proteins where the model performs well and poorly.
5. Export a clean prediction CSV with `protein_id`, `position`, `residue`, `true_label`, and `predicted_probability`.

## Suggested Submission Framing

DisorderNet is best presented as a complete, honest ML pipeline rather than only a high-score model. The strongest parts are the residue-level label construction, protein-level split, broad evaluation metrics, and custom sequence inference. The main limitation is that rigorous benchmarking should use sequence-identity-aware splits.
