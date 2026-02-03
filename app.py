import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

INPUT_SIZE = 28 * 28
HIDDEN_SIZE_1 = 512
HIDDEN_SIZE_2 = 256
OUTPUT_SIZE = 10

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE)
)

PATH = 'model.pt'
state_dict = torch.load(PATH)
model.load_state_dict(state_dict)
model.eval()

def center_digit(img):
    # img là ảnh nhị phân 28x28 (chữ trắng nền đen)

    # Công thức M_{ij} = \sum x^i * y^j * giá trị pixel (độ sáng)
    # Ví dụ m00 = \sum x^0 * y^0 * giá trị pixel = tổng độ sáng bức hình
    # Tương tự m10 = \sum x * giá trị pixel = trung bình trọng số độ sáng theo x
    # Tương tự m01 = \sum y * giá trị pixel = trung bình trọng số độ sáng theo y
    M = cv2.moments(img)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00']) # Tọa độ x của trọng tâm (x ngang)
        cy = int(M['m01'] / M['m00']) # Tọa độ y của trọng tâm (y dọc)
        
        # Tính toán độ lệch so với tâm hình học (14, 14)
        shift_x = 14 - cx
        shift_y = 14 - cy
        
        # Ma trận dịch chuyển
        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # Lấy toạ độ từng pixel của img nhân với ma trận dịch chuyển,
        # Nếu ra ngoài 28*28 thì bỏ, các ô chưa có thông tin mặc định = 0
        centered_img = cv2.warpAffine(img, M_shift, (28, 28))

        return centered_img
    return img

def preprocess_for_mnist_pytorch(image):
    # 1. Chuyển từ RGB (Gradio) sang Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. Resize ảnh về kích thước 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # 3. Đảo màu nếu là chữ đen nền trắng
    corners = [resized[0,0], resized[0,-1], resized[-1,0], resized[-1,-1]]
    if np.mean(corners) > 127:
        resized = cv2.bitwise_not(resized)
    
    # 4. Otsu Thresholding để làm sạch ảnh tuyệt đối (0 hoặc 255)
    _, clean = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Căn giữa theo trọng tâm
    centered = center_digit(clean)

    # 6. Đưa về khoảng [0,1]
    scaled = centered.astype('float32') / 255.0

    # 7. Reshape để phù hợp với đầu vào của mô hình
    reshaped = scaled.reshape(1, 1, 28, 28)

    # 8. Chuyển ảnh sang pytorch tensor
    final = torch.from_numpy(reshaped).float()

    return final

def predict_number(image):
    if image is None:
        return "Bạn chưa tải ảnh lên"

    input = preprocess_for_mnist_pytorch(image)
    logit = model(input)
    prediction = logit.argmax(dim=1)
    return f"Đây là số:  {prediction.item()}"

demo = gr.Interface(
    fn=predict_number,
    inputs=gr.Image(label="Tải ảnh chữ số của bạn"),
    outputs=gr.Textbox(label="Kết quả"),
    api_name="Predict handwritten digits",
    title="Dự đoán chữ số viết tay từ 0-9",
    description="Dự án của nhóm CONQ027, chương trình AIO Conquer 2026"
)

demo.launch()