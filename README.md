# Spectrum Sensing using VTCNN2

This project uses DeepSig RF dataset to train a VT-CNN2 model for classifying RF signals based on their modulation types to reproduce results from O'Shea et al.

I leverage a VTCNN2 (Vision Transformer-inspired CNN) architecture to identify signal types from I/Q sample data for spectrum sensing tasks. The data is chunked and processed incrementally to handle large-scale input efficiently.

---

## Dataset Overview

- **Total Size**: ~20.94 GB
- **Format**: `.npy` files containing I/Q sample data
- **Classes**: 24 modulation types
- **Preprocessing Steps**:
  - Data was explored and visualized (`notebooks/dataset_exploration.ipynb`)
  - Split into training and test sets (`notebooks/test_train_split.ipynb`)
  - Chunked into manageable portions for sequential training (`notebooks/train_chunks_split.ipynb`)

---

## Model: VTCNN2

Implemented using `TensorFlow Keras`, the model includes:

- Conv2D(256, (1,3), relu, L2 regularization)
- Conv2D(80, (2,3), relu, L2 regularization)
- Batch Normalization after each conv layer
- Flatten → Dense(256) → Dropout(0.5)
- Final Dense layer with Softmax (24 classes)

---

## Training Setup

- **Loss**: `categorical_crossentropy`
- **Optimizer**: Adam (`lr=0.001` with ReduceLROnPlateau scheduling)
- **Callbacks**:
  - TensorBoard for visualizations
  - ModelCheckpoint for saving weights
  - EarlyStopping to prevent overfitting
  - Learning rate scheduler (ReduceLROnPlateau)
- **Chunk-wise Training**:
  - Model is trained sequentially on 19 data chunks
  - Checkpoint is loaded and continued for each new chunk

---

## How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/spectrum-sensing.git
   cd spectrum-sensing
