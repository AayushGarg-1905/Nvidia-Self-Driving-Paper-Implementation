#  End-to-End Self-Driving Car using Deep Learning

This project implements NVIDIA’s end-to-end learning approach for autonomous driving using PyTorch and the Udacity Self-Driving Car Simulator. The goal is to train a convolutional neural network (CNN) that learns to steer a car directly from raw pixel data, eliminating the need for complex modular pipelines.

---

## Overview

The model is trained to predict steering angles from images captured by the center camera of a virtual self-driving car. The training is based on the paper [**"End to End Learning for Self-Driving Cars"** by NVIDIA](https://arxiv.org/abs/1604.07316).

---

## Dataset Generation

- **Simulator**: Udacity Self-Driving Car Simulator (v1)
- **Driving Time**: ~30 minutes on Track 1
- **Collected Data**: Images from center, left and right camera with corresponding steering angles

---

## Preprocessing & Augmentation

### Preprocessing
- Cropping irrelevant sky and car hood pixels
- Resizing to **200x66** resolution
- Normalization using `x / 127.5 - 1.0` scale
- Converted color space from RGB to YUV (as used in the paper)

### Augmentations
- **Random horizontal translation**: simulates lateral shifts and adjusts the steering angle proportionally.
- **Random brightness adjustment**: applied to simulate varying lighting conditions.
- **Random horizontal flipping**: image is flipped, and the steering angle is negated to balance left/right turn bias.
- **Undersampling of near-zero steering angles**: reduces overrepresentation of straight driving to improve learning for turns.

---

## Model Architecture

The CNN architecture exactly replicates the NVIDIA paper:

- Input layer: 200×66×3 image (YUV)
- 5 Convolutional layers with ELU activation
- 3 Fully connected layers
- Final output: single value representing the steering angle

> **Note:**  
> Unlike the original paper which did not specify the activation function, this implementation uses **ELU (Exponential Linear Units)** for better convergence and performance in deep networks.

## Training Details

- **Framework**: PyTorch
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Batch Size**: 64
- **Epochs**: 25
- **Train/Test Split**: 80/20

---

## Evaluation & Results

- The model **successfully completed a full lap** on an unseen test track without going off-road.
- It occasionally drifted near road boundaries but was able to **self-correct** and recover.
- On the training track, the model performed well except in one dark region, suggesting the need for more data under varied lighting conditions.

---

##  Key Insights

- End-to-end learning is highly data-dependent. NVIDIA used over **72 hours of driving data** under diverse conditions, while this project uses only **30 minutes**, making the successful generalization on unseen data a significant achievement.
- Data augmentation and balanced steering angle distribution are crucial to model performance.
- Even with limited data, the model demonstrates the viability of the original paper’s architecture in real-time control.

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/nvidia-self-driving-car.git
cd nvidia-self-driving-car/Implementation
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data

- Create a `data` folder in the root directory.
- Launch the simulator in **training mode**.
- Collect driving data for at least 30 minutes. The simulator will save images and a `driving_log.csv` file in the `data` folder.

### 5. Handle Dataset Imbalance

- Use the scripts in the `handling-dataset-imbalance` folder to analyze and balance your dataset:
    - Check the steering angle distribution:
      ```bash
      python handling-dataset-imbalance/csv_file_check.py
      ```
    - Undersample near-zero steering angles and create a balanced CSV:
      ```bash
      python handling-dataset-imbalance/make_csv_balanced.py
      ```
    - This will create a `balanced_data` folder with a `driving_log_balanced.csv` file.

### 6. Train the Model

```bash
python train.py
```
- The model will be trained using the balanced dataset.
- Checkpoints will be saved in the `checkpointsV2` folder.

### 7. Run in Autonomous Mode

- Launch the simulator in **autonomous mode**.
- Run the following command to start the inference server:
  ```bash
  python drive.py
  ```
- The car should now drive autonomously using the trained model.

---

## Notes

- Make sure your folder structure matches the expected paths in the scripts.
- You can adjust training parameters (epochs, batch size, learning rate) in `train.py`.
---

## References

- [End to End Learning for Self-Driving Cars (arXiv)](https://arxiv.org/pdf/1604.07316)
- [Udacity Simulator](https://github.com/udacity/self-driving-car-sim)


