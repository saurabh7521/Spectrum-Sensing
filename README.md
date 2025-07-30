# Spectrum Sensing using VTCNN2

This project uses DeepSig RF dataset to train a VT-CNN2 model for classifying RF signals based on their modulation types to reproduce some results from O'Shea et al [10.1109/JSTSP.2018.2797022](https://ieeexplore.ieee.org/abstract/document/8267032). The model is intentionally kept shallow with the aim of being able to deploy on low cost hardware like Arduino, Raspberry Pi.

I leverage a VTCNN2 (Vision Transformer-inspired CNN) architecture to identify signal types from I/Q sample data for spectrum sensing tasks. The data is chunked and processed incrementally to handle large-scale input efficiently.

---

## Dataset Overview
- **Source** `https://www.kaggle.com/datasets/aleksandrdubrovin/deepsigio-radioml-201801a-new`
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

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Train model on chunk**
   ```bash
   python train.py --chunk_id=0
3. **Outputs**
- Checkpoints saved to `/models/`
- Logs and metrics saved to `/logs/`
- TensorBoard:
  ```bash
  tensorboard --logdir=logs/

## Limitations

1. **Shallow Network Depth**
   The VTCNN2 model is much simpler than deeper CNN or ResNet-based models used in O'Shea et al., limiting its ability to capture complex modulation features.

2. **High Dropout Rate (0.5)**
   A dropout rate of 50% can overly regularize a small network. Causes underfitting, especially when the model has limited representational capacity to begin with.

3. **Aggressive Normalization**
   Per-sample normalization will distort amplitude-sensitive modulations, reducing class separability. Some modulation types are inherently amplitude-dependent (e.g., ASK, QAM).

4. **Label Smoothing**
   Label smoothing was not applied. For large multi-class classification, label smoothing can regularize and reduce overconfidence, improving generalization.

5. **No SNR Awareness**
   The model lacks per-sample SNR information, which is crucial for generalizing across different noise conditions.

---

**Signal visualization**

The following GIF was created using dataset_exploration.ipynb as an example. It shows the constellation plot of some datapoints for the modulation type QPSK.

![QPSK_constellation](https://github.com/user-attachments/assets/6c3ce6d8-1255-4e07-9ff4-d4504c3e140a)

