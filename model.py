import torch
import torch.nn as nn
import numpy as np
from resnet_architecture import ResNetECG

def load_model(model_path=r"models\best_model_epoch_69.pth", input_channels=12, device="cpu"):
    model = ResNetECG(input_channels=input_channels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    return model

def preprocess_ecg(ecg_array, expected_length=1000):
    ecg = np.array(ecg_array)

    if ecg.ndim != 2 or ecg.shape[0] != 12:
        raise ValueError(f"Expected shape (12, {expected_length}), got {ecg.shape}")

    if ecg.shape[1] != expected_length:
        raise ValueError(f"Expected signal length {expected_length}, got {ecg.shape[1]}")

    ecg = (ecg - np.mean(ecg)) / np.std(ecg)

    ecg_tensor = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 12, 5000)
    return ecg_tensor

def predict(model, ecg_tensor):
     index_to_label = {
                    0: 'NORM',
                    1: 'CD',
                    2: 'HYP',
                    3: 'MI',
                    4: 'STTC',
                    5: 'LVH',
                    6: 'LAFB',
                    7: 'ISC_',
                    8: 'IRBBB',
                    9: 'IVCD'      }
     with torch.no_grad():
        output = model(ecg_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
        label = index_to_label[predicted_index]
        return {
            "predicted_class_index": predicted_index,
            "predicted_label": label
        }
