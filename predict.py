# predict.py

import os
import torch
from PIL import Image
from torchvision import transforms
from utils import compute_directional_features
from model import ScriptCNN
from config import resize_x, resize_y, input_channels, num_classes, class_names
from utils import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------
# Load Model (for prediction)
# -----------------------------------
model = ScriptCNN(num_classes=num_classes).to(device)
checkpoint_path = "checkpoints/final_weights.pt"

if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path}")
else:
    logger.error(f"Model weights not found at {checkpoint_path}")

# -----------------------------------
# Prediction Transforms
# -----------------------------------
transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((resize_x, resize_y)),
    transforms.Lambda(lambda img: compute_directional_features(img))
])

# -----------------------------------
# Main Prediction Function
# -----------------------------------
# -----------------------------------
# Main Prediction Function (Modified)
# -----------------------------------
def classify_images(list_of_image_paths):
    results = []
    
    for img_path in list_of_image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform_test(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

                pred_idx = probs.argmax()
                pred_label = class_names[pred_idx]
                confidence = probs[pred_idx]

            result = {
                "image_path": img_path,
                "predicted_label": pred_label,
                "confidence": confidence,
                "probabilities": {class_name: float(prob) for class_name, prob in zip(class_names, probs)}
            }

            results.append(result)
            logger.info(f"Image: {os.path.basename(img_path)} | Predicted: {pred_label} | Confidence: {confidence:.4f}")
        
        except Exception as e:
            logger.warning(f"Skipped {img_path}: {str(e)}")
            results.append({
                "image_path": img_path,
                "predicted_label": "Error",
                "confidence": 0.0,
                "probabilities": {}
            })

    return results
