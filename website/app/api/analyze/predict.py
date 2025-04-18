import os
import sys
import json
import warnings
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze.rag import get_care_recommendations


# ───────────── silence TF/Keras noise ─────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ───────────── class index → full label ─────────────
# ⚠️ The order **must** match the class order used when the model was trained
wanted_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
    'Strawberry___healthy', 'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

def _simplify_label(full_label: str) -> str:
    """
    Convert 'Tomato___Early_blight'  →  'Early blight'
    Convert 'Apple___healthy'        →  'Healthy'
    """
    disease = full_label.split('___')[1]         # take RHS of triple‑underscore
    if 'healthy' in disease.lower():
        return 'Healthy'
    # clean up underscores / double spaces
    disease = disease.replace('_', ' ').strip()
    disease = ' '.join(disease.split())
    # title‑case for nicer display
    return disease[0].upper() + disease[1:]

# ───────────── inference helper ─────────────
def get_img_array(img_path, size):
    """Helper function to load and preprocess image"""
    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = array / 255.0
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap"""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_heatmap_image(img_path, heatmap, alpha=0.4):
    """Generate the final heatmap overlay image and return it as base64"""
    import cv2
    import base64

    # Load and resize original image
    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))

    # Resize and colorize heatmap
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Create overlay
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Convert to base64 with compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Reduce quality to 85%
    _, buffer = cv2.imencode('.jpg', overlay, encode_param)  # Use JPEG instead of PNG
    
    # Add the data URL prefix to the base64 string
    base64_string = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_string}"

# Add these plant species classes after the wanted_classes list
plant_species = [
    'Apple', 'Cherry', 'Corn', 'Grape', 'Peach', 
    'Pepper', 'Potato', 'Strawberry', 'Tomato'
]

def make_gradcam_heatmap_pytorch(model, image_tensor, target_layer_name, pred_index):
    """Generate Grad-CAM heatmap for PyTorch models"""
    # Get the target layer
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    
    # Register hooks for the target layer
    activation = {}
    gradients = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    def get_gradient(name):
        def hook(grad):
            gradients[name] = grad.detach()
        return hook
    
    # Register forward and backward hooks
    handle_forward = target_layer.register_forward_hook(get_activation(target_layer_name))
    
    # Get model output and register hook for the target layer
    output = model(image_tensor)
    if pred_index is None:
        pred_index = output.argmax(dim=1)
    
    target_layer_activation = activation[target_layer_name]
    target_layer_activation.register_hook(get_gradient(target_layer_name))
    
    # Backward pass
    model.zero_grad()
    output[0, pred_index].backward()
    
    # Get gradients and activations
    gradients = gradients[target_layer_name]
    activations = activation[target_layer_name]
    
    # Calculate weights
    weights = torch.mean(gradients, dim=(2, 3))[0]
    
    # Generate heatmap
    heatmap = torch.sum(weights[:, None, None] * activations[0], dim=0)
    heatmap = torch.relu(heatmap)  # ReLU to only keep positive contributions
    
    # Normalize heatmap
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-10)
    
    # Clean up
    handle_forward.remove()
    
    return heatmap

def get_plant_species(image_path: str) -> tuple[str, float, str]:
    """Predict plant species using the DenseNet model and generate heatmap"""
    # Initialize model
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, len(plant_species))
    
    # Load model weights
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    model_path = os.path.join(project_root, "Plant Classification", "best_model_so_far.pth")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Transform image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    
    species = plant_species[pred.item()]
    confidence = round(float(conf.item()) * 100, 2)
    
    # Grad-CAM implementation
    features = None
    
    def hook_fn(module, input, output):
        nonlocal features
        features = output.detach()
    
    # Register hook on the last conv layer
    handle = model.features.denseblock4.denselayer16.conv2.register_forward_hook(hook_fn)
    
    # Forward pass to get features
    with torch.no_grad():
        _ = model(image_tensor)
    
    # Remove the hook
    handle.remove()
    
    # Generate basic activation map from features
    activation_map = torch.mean(features[0], dim=0)  # Average across channels
    
    # Normalize activation map
    activation_map = activation_map.numpy()
    activation_map = np.maximum(activation_map, 0)  # ReLU
    activation_map = activation_map - np.min(activation_map)
    activation_map = activation_map / (np.max(activation_map) + 1e-8)
    
    # Resize activation map to match input image size
    activation_map = cv2.resize(activation_map, (224, 224))
    
    # Generate the final heatmap image
    heatmap_image = generate_heatmap_image(image_path, activation_map)
    
    return species, confidence, heatmap_image

def predict_image(image_path: str) -> None:
    try:
        # Get plant species with heatmap
        species, species_conf, species_heatmap = get_plant_species(image_path)

        # locate model for health classification
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
        model_path = os.path.join(project_root, "Plant Disease", "plant_disease_classifier.keras")

        model = tf.keras.models.load_model(model_path, compile=False)

        # preprocess
        img_arr = get_img_array(image_path, size=(224, 224))

        # predict
        preds = model.predict(img_arr, verbose=0)[0]
        top = int(np.argmax(preds))
        conf = round(float(np.max(preds) * 100), 2)

        # Generate health condition heatmap
        last_conv_layer_name = "conv5_block16_2_conv"
        heatmap = make_gradcam_heatmap(img_arr, model, last_conv_layer_name, top)
        health_heatmap = generate_heatmap_image(image_path, heatmap)

        # map to simplified condition
        full_label = wanted_classes[top] if top < len(wanted_classes) else f"Class_{top}"
        condition = _simplify_label(full_label)

        # craft recommendation
        if condition == "Healthy":
            recommendation = "Based on the analysis, maintain your current care routine."
        else:
            # Get RAG-based recommendation
            rag_recommendation = get_care_recommendations(species, condition)
            
            # Combine recommendations
            if condition == "Healthy":
                recommendation = (
                    "Based on the analysis, maintain your current care routine. "
                    f"{rag_recommendation}"
                )
            else:
                recommendation = (
                    f"Based on the analysis, your plant shows signs of **{condition}**. "
                    f"{rag_recommendation}"
                )


        # response
        result = {
            "species": species,
            "speciesConfidence": species_conf,
            "speciesHeatmap": species_heatmap,
            "condition": condition,
            "confidence": conf,
            "heatmapImage": health_heatmap,
            "recommendation": recommendation,
            "infoReliability": 95
        }

        print(json.dumps(result))

    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Image path not provided"}), file=sys.stderr)
        sys.exit(1)

    predict_image(sys.argv[1])
