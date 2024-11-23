# Cvdrone
Create a computer vision model to detect target reach destination and able to navigate through gmaps


landmark-detection/
│
├── data/                         # Dataset folder
│   ├── train/                    # Training images
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── labels.json               # JSON file with image-to-label mapping
│
├── models/                       # Folder for saving trained models
│   └── landmark_model.pth        # Trained model weights
│
├── scripts/                      # Code files
│   ├── train.py                  # Script to train the model
│   ├── inference.py              # Script for inference on images/videos
│
├── requirements.txt              # Dependencies
│
├── README.md                     # Project documentation
│
└── main.py                       # Main script to combine vision and GPS
