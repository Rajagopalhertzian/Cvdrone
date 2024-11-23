import torch
import torchvision.transforms as transforms
from PIL import Image
from scripts.train import LandmarkModel  # Import the model

# Define parameters
model_path = "models/landmark_model.pth"
image_path = "path/to/test_image.jpg"
num_classes = 10

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LandmarkModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Perform inference
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    print(f"Predicted Class: {predicted.item()}")
