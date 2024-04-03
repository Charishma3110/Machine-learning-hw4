import numpy as np
from torchvision import models, transforms
import torch
from PIL import Image
from captum.attr import DeepLift, IntegratedGradients
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model
model = models.googlenet(pretrained=True).to(device)
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image paths and corresponding labels
image_info = [
    (r"C:\Users\Cherry\OneDrive\Desktop\ml hw4\goldfish.jpg", 1),  # Goldfish
    (r"C:\Users\Cherry\OneDrive\Desktop\ml hw4\hummingbird.jpg", 94),  # Hummingbird
    (r"C:\Users\Cherry\OneDrive\Desktop\ml hw4\blackswan.jpg", 100),  # Black swan
    (r"C:\Users\Cherry\OneDrive\Desktop\ml hw4\gret.jpg", 207),  # Golden retriever
    (r"C:\Users\Cherry\OneDrive\Desktop\ml hw4\daisy.jpg", 985),  # Daisy
]

# Initialize attribution methods
deep_lift = DeepLift(model)
integrated_gradients = IntegratedGradients(model)

# Function to visualize attributions
def visualize_attributions(input_image, attributions, method_name, class_name):
    # Sum across color channels and normalize
    attribution_sum = np.mean(attributions.squeeze().cpu().detach().numpy(), axis=0)
    norm_attr = (attribution_sum - np.min(attribution_sum)) / (np.max(attribution_sum) - np.min(attribution_sum))
    
    plt.imshow(norm_attr, cmap='hot')
    plt.colorbar()
    plt.title(f"{class_name}\n{method_name}")
    plt.axis('off')

# Process and visualize for each image
for image_path, true_label in image_info:
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get model output
    output = model(input_tensor)
    prediction_score, pred_label_idx = torch.max(output, 1)
    predicted_label = pred_label_idx.item()

    # Compute attributions
    attribution_dl_pred = deep_lift.attribute(input_tensor, target=predicted_label)
    attribution_dl_true = deep_lift.attribute(input_tensor, target=true_label)
    attribution_ig_pred = integrated_gradients.attribute(input_tensor, target=predicted_label)
    attribution_ig_true = integrated_gradients.attribute(input_tensor, target=true_label)

    # Visualize
    plt.figure(figsize=(18, 8))
    
    # Original image
    plt.subplot(2, 5, 1)
    plt.imshow(original_image)
    plt.title(f'Original\n{image_path.split("\\")[-1]}')
    plt.axis('off')
    
    # DeepLift - Predicted
    plt.subplot(2, 5, 2)
    visualize_attributions(input_tensor, attribution_dl_pred, "DeepLift (Pred)", "")

    # DeepLift - True
    plt.subplot(2, 5, 3)
    visualize_attributions(input_tensor, attribution_dl_true, "DeepLift (True)", "")
    
    # Integrated Gradients - Predicted
    plt.subplot(2, 5, 6)
    visualize_attributions(input_tensor, attribution_ig_pred, "IntGrad (Pred)", "")
    
    # Integrated Gradients - True
    plt.subplot(2, 5, 7)
    visualize_attributions(input_tensor, attribution_ig_true, "IntGrad (True)", "")

    plt.suptitle(f"{image_path.split('\\')[-1].split('.')[0]} - Predicted: {predicted_label}, True: {true_label}", fontsize=16)
    plt.show()
