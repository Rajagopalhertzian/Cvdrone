import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os

# Define the dataset class
class LandmarkDataset(Dataset):
    def _init_(self, image_dir, labels, transform=None):
        """
        Args:
            image_dir (str): Directory with images.
            labels (dict): Dictionary mapping image names to labels.
            transform (callable, optional): Optional transforms to apply to the images.
        """
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.image_names = list(labels.keys())
    
    def _len_(self):
        return len(self.image_names)
    
    def _getitem_(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        label = self.labels[image_name]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define the CNN model
class LandmarkModel(nn.Module):
    def _init_(self, num_classes):
        super(LandmarkModel, self)._init_()
        # Use a pre-trained model (ResNet18) for feature extraction
        self.resnet = models.resnet18(pretrained=True)
        # Modify the final fully connected layer to match the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Hyperparameters
num_classes = 10  # Adjust based on the number of landmarks
learning_rate = 0.001
batch_size = 16
num_epochs = 10

# Transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Example dataset setup (replace with your actual data paths and labels)
image_dir = "path/to/images"
labels = {"image1.jpg": 0, "image2.jpg": 1, "image3.jpg": 2}  # Example
dataset = LandmarkDataset(image_dir, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LandmarkModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "landmark_model.pth")

print("Model training complete.")
