# Cvdrone
Create a computer vision model to detect target reach destination and able to navigate through gmaps


landmark-detection/
│
├── data/                         # Dataset folder
│   ├── train/                    # Directory containing training images
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...                   # Additional images
│   ├── labels.json               # JSON file mapping images to their labels
│
├── models/                       # Directory to save trained model weights
│   └── landmark_model.pth        # Trained model file
│
├── scripts/                      # Python scripts for training and inference
│   ├── train.py                  # Script to train the landmark detection model
│   ├── inference.py              # Script for performing inference on new images or videos
│
├── requirements.txt              # Python dependencies required for the project
│
├── README.md                     # Project documentation (this file)
│
└── main.py                       # Main script combining vision-based detection and GPS data
