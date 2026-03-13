# MSBA: Backdoor Attacks and Defenses in Medical Vision-Language Models (MedCLIP)

This repository contains the source code for the project related to **MedCLIP**, focusing on backdoor attacks, federated learning (FedAvg), and model explainability (Grad-CAM visualization).

## 🚀 Project Overview

The project explores the security of Medical Vision-Language Models (specifically MedCLIP) and includes components for:
- **Federated Learning / Training**: Implementing `fedavg.py`.
- **Backdoor Attacks**: Simulating and evaluating backdoor attacks (`backdoor_data.py`).
- **Model Evaluation**: Metrics evaluation script (`evaluate_metrics.py`) computing Macro-AUC, F1-Score, Balanced Accuracy, etc., on datasets like MIMIC-CXR.
- **Explainability**: Using Grad-CAM to visualize model attention on clean and poisoned (backdoored) medical images (`visualize_gradcam.py`).

## 📥 Data and Pretrained Models

To run the experiments, you need to prepare the datasets and pretrained models. Please download them and place them in their respective directories at the project root.

### 1. Pretrained Checkpoints
Since Hugging Face / Model weights are large, they are not included in this repository. You need to obtain and place them as follows:
- **Bio_ClinicalBERT**: Place the models in `./Bio_ClinicalBERT/`
- **Swin_Transformer**: Place the models in `./Swin_Transformer/`
- **MedCLIP Pretrained Models**:
  - ResNet variant: `./pretrained/medclip-resnet/pytorch_model.bin`
  - ViT variant: `./pretrained/medclip-vit/pytorch_model.bin`

*(You can usually download these standard base models directly from HuggingFace Hub.)*

### 2. Datasets
- **MIMIC-CXR** / Other Medical Datasets: Place the necessary dataset files, images, and metadata CSVs (e.g., `mimic-cxr-train-meta.csv`) into the `./data/` and `./local_data/` directories.

### 3. ⭐️ Our Best Model Checkpoint
To reproduce the best results or evaluate our trained model, please download our fine-tuned/best model checkpoint from the link below and place it under `./outputs/models/best_model/`:

- **Download Link**: [Quark Drive (夸克网盘) Link](https://pan.quark.cn/s/1b49bbb37908)

*(Filename expected: `global_model_fed_avg_0ToAll_6_998_1wbd_i.pth`)*

## 🛠️ Usage

### Evaluation
To evaluate the model's performance on the 6-class disease diagnosis task:
```bash
python evaluate_metrics.py
```

### Grad-CAM Visualization
To generate Grad-CAM heatmaps for clean and poisoned images and visualize where the model focuses its attention:
```bash
python visualize_gradcam.py
```

## 📄 License
This codebase is intended for academic and research purposes.
