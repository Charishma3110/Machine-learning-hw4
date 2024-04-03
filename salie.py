from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision import models
from captum.attr import Saliency
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the model with the correct weights parameter
model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(device)
model.eval()

# Defining the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image paths and corresponding ImageNet class indices
images_info = [
    (r"C:\Users\Cherry\OneDrive\Desktop\ml hw4\goldfish.jpg", 1),  # Goldfish
    (r"C:\Users\Cherry\OneDrive\Desktop\ml hw4\hummingbird.jpg", 94),  # Hummingbird
    (r"C:\Users\Cherry\OneDrive\Desktop\ml hw4\blackswan.jpg", 100),  # Black swan
    (r"C:\Users\Cherry\OneDrive\Desktop\ml hw4\gret.jpg", 207),  # Golden retriever
    (r"C:\Users\Cherry\OneDrive\Desktop\ml hw4\daisy.jpg", 985),  # Daisy
]

# Function to load an image, transform it, and create predictions and saliency attributions
def predict_and_visualize(image_path, label_index):
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform(image).unsqueeze(0).to(device)
    
    # Making a prediction
    output = model(transformed_image)
    prediction_score, pred_label_idx = torch.max(output, 1)
    predicted_label = pred_label_idx.item()

    # Initializing Saliency
    saliency = Saliency(model)

    # Saliency map for the predicted label
    saliency_map_pred = saliency.attribute(transformed_image, target=predicted_label)
    saliency_map_pred = saliency_map_pred.squeeze().cpu().detach().numpy()

    # Saliency map for the true label
    saliency_map_true = saliency.attribute(transformed_image, target=label_index)
    saliency_map_true = saliency_map_true.squeeze().cpu().detach().numpy()

    return image, predicted_label, saliency_map_pred, saliency_map_true

# Processing and visualizing for each image
for image_path, true_label in images_info:
    original_image, predicted_label, saliency_map_pred, saliency_map_true = predict_and_visualize(image_path, true_label)
    
    # Plot original image and saliency maps
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(np.maximum(0, np.sum(saliency_map_pred, axis=0)), cmap='hot', interpolation='nearest')
    axs[1].set_title('Saliency Map (Predicted)')
    axs[1].axis('off')
    
    axs[2].imshow(np.maximum(0, np.sum(saliency_map_true, axis=0)), cmap='hot', interpolation='nearest')
    axs[2].set_title(f'Saliency Map (True: {true_label})')
    axs[2].axis('off')
    
    plt.show()
