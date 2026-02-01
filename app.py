import cv2
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
)

PATH = 'model.pt'
state_dict = torch.load(PATH)
model.load_state_dict(state_dict)
model.eval()

def preprocess_for_mnist_pytorch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    inverted = cv2.bitwise_not(resized) 
    scaled = inverted.astype('float32') / 255.0
    normalized = (scaled - 0.1307) / 0.3081
    reshaped = normalized.reshape(1, 28, 28, 1)
    final = torch.from_numpy(reshaped).float()

    return final

def predict_number(image):
    if image is None:
        return "Bạn chưa tải ảnh lên"

    input = preprocess_for_mnist_pytorch(image)
    logit = model(input)
    prediction = logit.argmax(dim=1)
    return f"This is the number:  {prediction.item()}"

demo = gr.Interface(
    fn=predict_number,
    inputs=gr.Image(label="Upload your digit image (white background)"),
    outputs=gr.Textbox(label="Result"),
    api_name="Predict handwriting numbers from 0-9",
    title="Predict the handwritten digits from 0-9",
    description="Project by team CONQ027, AIO Conquer 2026 program"
)

demo.launch()