import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# So model can be loaded
import sys
sys.path.append('../train')


# Load your trained model
model = torch.load("../train/results/colab/rn50_uf34_1_us_final.pth",
                   map_location=torch.device('cpu'))
model.eval()

# Define your transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Open the video
cap = cv2.VideoCapture("../data/sample_vid.mp4")

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('classified.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    # Apply transforms
    input_tensor = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = probabilities.argmax(1).item()
        confidence = probabilities[0][predicted_class].item()

    # Prepare text to display
    labels = ['non_smiling', 'smiling']
    text = f"{labels[predicted_class]}: {confidence:.2f}"

    # Display the label and confidence on the frame
    height = frame.shape[0]  # Get the height of the frame
    cv2.putText(frame, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)

    # Write the frame with the text
    out.write(frame)

    cv2.imshow('Frame', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
