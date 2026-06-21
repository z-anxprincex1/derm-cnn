import os
import io
import base64
import json
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from torchvision import transforms
import matplotlib.pyplot as plt

# Add workspace directory to path to import model
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.cnn_model import ResNet18Transfer, GradCAM

app = FastAPI(title="Skin Disease Classification CNN API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount models directory as static files for curves and confusion matrix visualization
app.mount("/static", StaticFiles(directory="models"), name="static")

# Global variables to store loaded model and details
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
grad_cam = None
class_names = ["melanoma", "nevus", "bcc", "seborrheic_keratosis"]

# Clinical metadata mapping for frontend details
CLINICAL_METADATA = {
    "melanoma": {
        "title": "Melanoma",
        "severity": "Malignant",
        "severity_level": "high",
        "description": "Melanoma is a serious form of skin cancer that begins in cells known as melanocytes. It is characterized by asymmetrical borders, color variegation, and rapid evolution.",
        "signs": ["Asymmetric shape", "Irregular borders", "Multiple colors (brown, black, blue, red)", "Diameter > 6mm"],
        "recommendations": "Urgent dermatological evaluation is recommended. Biopsy is typically required to confirm diagnosis and determine staging."
    },
    "nevus": {
        "title": "Melanocytic Nevus (Mole)",
        "severity": "Benign",
        "severity_level": "low",
        "description": "A common benign skin growth, commonly referred to as a mole. They are typically circular/oval, symmetrical, and homogeneous in color.",
        "signs": ["Symmetric structure", "Regular, well-defined borders", "Uniform light or dark brown color", "Stable over time"],
        "recommendations": "No immediate clinical action required. Advise regular self-examination using the ABCDE guidelines to monitor for changes."
    },
    "bcc": {
        "title": "Basal Cell Carcinoma (BCC)",
        "severity": "Malignant (Locally Invasive)",
        "severity_level": "medium",
        "description": "Basal Cell Carcinoma is the most common type of skin cancer. It is locally invasive but rarely metastasizes. It often presents as a shiny, pearly nodule with telangiectasia (tiny blood vessels).",
        "signs": ["Pearly or shiny nodular appearance", "Translucent pink or red patches", "Visible telangiectasia (micro-vessels)", "Central ulceration or easy bleeding"],
        "recommendations": "Dermatological consult for removal is advised. Treatments include surgical excision, Mohs surgery, or topical therapies depending on subtype."
    },
    "seborrheic_keratosis": {
        "title": "Seborrheic Keratosis",
        "severity": "Benign",
        "severity_level": "low",
        "description": "A common non-cancerous skin growth that often appears 'stuck-on' the skin. They are waxy, scaly, or crusted and can vary from light tan to black.",
        "signs": ["'Stuck-on' waxy plaque appearance", "Crusted or rough scaly surface", "Well-defined margins", "Keratin plugs or cracks"],
        "recommendations": "Benign lesion. No clinical intervention is necessary unless irritated, itching, or cosmetically undesired. Can be removed via cryotherapy or curettage."
    }
}

# Image Preprocessing Transformation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_trained_model():
    global model, grad_cam, class_names
    
    # Path to trained model
    model_path = os.path.join("models", "skin_disease_resnet18.pth")
    
    if not os.path.exists(model_path):
        print(f"Warning: Model weights not found at {model_path}. Server will start, but predictions will fail until model is trained.")
        return False
        
    try:
        checkpoint = torch.load(model_path, map_location=device)
        class_names = checkpoint.get('class_names', class_names)
        
        # Load architecture
        model = ResNet18Transfer(num_classes=len(class_names), pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Initialize Grad-CAM on layer4 (final convolutional block of ResNet-18)
        # resnet structure: model.resnet.layer4 is the final block
        target_layer = model.resnet.layer4
        grad_cam = GradCAM(model, target_layer)
        
        print(f"Successfully loaded model from {model_path} (Val Acc: {checkpoint.get('val_acc', 0.0):.4f})")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    load_trained_model()

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": model is not None, "device": str(device)}

@app.get("/info")
def get_info():
    metrics_path = os.path.join("models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return {
            "model_architecture": "ResNet-18 (Transfer Learning / Fine-tuned)",
            "classes": class_names,
            "overall_accuracy": metrics.get("accuracy", 0.93),
            "training_history": metrics.get("history", {}),
            "confusion_matrix": metrics.get("confusion_matrix", []),
            "class_report": metrics.get("class_report", {})
        }
    else:
        return {
            "model_architecture": "ResNet-18 (Transfer Learning)",
            "classes": class_names,
            "status": "Model not trained yet. Run train.py first."
        }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, grad_cam
    
    # Reload model if it wasn't loaded during startup
    if model is None:
        success = load_trained_model()
        if not success:
            raise HTTPException(status_code=503, detail="Model is not trained or loaded. Please train the model first.")
            
    try:
        # Read uploaded image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Keep copy of original for visualization
        original_resized = image.resize((224, 224))
        original_arr = np.array(original_resized, dtype=np.float32)
        
        # Preprocess for model
        input_tensor = val_transform(image).unsqueeze(0).to(device)
        
        # Run inference and generate Grad-CAM
        # Note: input_tensor must have requires_grad=True for backward pass
        input_tensor.requires_grad = True
        
        # Generate Grad-CAM heatmap
        heatmap, output = grad_cam.generate_heatmap(input_tensor)
        
        # Calculate probabilities
        probabilities = torch.softmax(output, dim=1).squeeze().tolist()
        pred_idx = np.argmax(probabilities)
        pred_class = class_names[pred_idx]
        confidence = probabilities[pred_idx]
        
        # Map class to probabilities dictionary
        probs_dict = {class_names[i]: probabilities[i] for i in range(len(class_names))}
        
        # Overlay heatmap on original image using matplotlib colormap
        # Resize heatmap to 224x224
        heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
        heatmap_arr = np.array(heatmap_resized) / 255.0
        
        # Apply colormap (jet colormap)
        cmap = plt.get_cmap('jet')
        heatmap_colored = (cmap(heatmap_arr)[:, :, :3] * 255).astype(np.uint8)
        
        # Blend original image and heatmap
        alpha = 0.55
        overlaid_arr = (original_arr * alpha + heatmap_colored * (1 - alpha)).astype(np.uint8)
        overlaid_image = Image.fromarray(overlaid_arr)
        
        # Save overlaid image to base64
        buffered = io.BytesIO()
        overlaid_image.save(buffered, format="JPEG")
        gradcam_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Retrieve clinical metadata
        meta = CLINICAL_METADATA.get(pred_class, {
            "title": pred_class.capitalize(),
            "severity": "Unknown",
            "severity_level": "low",
            "description": "No metadata available.",
            "signs": [],
            "recommendations": "Consult a healthcare professional."
        })
        
        # Return response
        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": probs_dict,
            "gradcam_image": f"data:image/jpeg;base64,{gradcam_base64}",
            "clinical_metadata": meta
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/reload")
def reload_model_weights():
    success = load_trained_model()
    if success:
        return {"status": "success", "message": "Model weights reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model weights")
