import asyncio
import httpx
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, Response
import time
import onnxruntime as ort
import torch
import os
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
import base64

# RTSP stream config
ESP_IP = "192.168.1.102"  # Địa chỉ IP của ESP32
ESP32_URL = f"http://{ESP_IP}/trigger_light"  # Endpoint để kích hoạt đèn
RTSP_URL = f"rtsp://{ESP_IP}:5005/mjpeg/1"  # Luồng RTSP từ ESP32-CAM

app = FastAPI()  # Khởi tạo ứng dụng FastAPI


onnx_session = ort.InferenceSession("best.onnx") # Tải mô hình YOLOv10 bản ONNX lên
inception_resnet = InceptionResnetV1(pretrained='vggface2').eval() # Khởi tạo mô hình FaceNet pretrained để trích xuất đặc trưng từ khuôn mặt

vcap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG) # Cấu hình luồng RTSP
frame_buffer = None 
familiar_faces = None
unfamiliar_face = None 

last_signal_time = 0 # Thời gian gửi tín hiệu cuối cùng
SIGNAL_COOLDOWN = 10  # Giới hạn khoảng thời gian giữa các tín hiệu bật đèn gửi tới ESP32


# Preprocess image for ONNX model
async def preprocess_image(image, input_size=(320, 320)):
    def sync_preprocess(image):
        # Resize to model's input size
        image = cv2.resize(image, input_size)
        # Normalize pixel values
        image_data = image.astype(np.float32) / 255.0
        # Transpose channels from (H, W, C) to (C, H, W)
        image_data = np.transpose(image_data, (2, 0, 1))
        # Add batch dimension
        return np.expand_dims(image_data, axis=0)
    
    return await asyncio.to_thread(sync_preprocess, image)


# Function to run face detection using ONNX, async for FastAPI
async def detect_faces_onnx(image, image_height, image_width, input_size=(320, 320), confidence_threshold=0.6):
    input_data = await preprocess_image(image, input_size)
    input_name = onnx_session.get_inputs()[0].name

    # Run inference asynchronously
    def sync_inference():
        outputs = onnx_session.run(None, {input_name: input_data})
        detections = outputs[0][0]
        boxes, scores, labels = [], [], []
        for detection in detections:
            confidence = detection[4]
            if confidence >= confidence_threshold:
                x_min, y_min, x_max, y_max = detection[:4]
                x_min = int(x_min * image_width / input_size[0])
                y_min = int(y_min * image_height / input_size[1])
                x_max = int(x_max * image_width / input_size[0])
                y_max = int(y_max * image_height / input_size[1])
                class_label = int(detection[5])
                boxes.append([x_min, y_min, x_max, y_max])
                scores.append(confidence)
                labels.append(class_label)
        return boxes, scores, labels
    
    return await asyncio.to_thread(sync_inference)


# Function to compare embeddings
async def compare_embeddings(embedding1, embedding2):
    return await asyncio.to_thread(cosine, embedding1, embedding2)


# Check if a face is familiar or unfamiliar
async def check_for_unfamiliar_face(embedding, familiar_faces, threshold=0.5):
    face2score = {}
    for face_name, face_embeddings in familiar_faces.items():
        similarity_scores = []
        for face_embedding in face_embeddings:
            similarity = 1 - await compare_embeddings(embedding, face_embedding)
            similarity_scores.append(similarity)
        max_similarity = np.max(similarity_scores)
        face2score[face_name] = max_similarity

    print(f"Similarity scores: {face2score}")
    best_match = max(face2score, key=face2score.get)
    if face2score[best_match] > threshold:
        return f"{best_match}"

    return "Doi tuong la"


# Load known faces and generate embeddings
async def load_known_faces(folder_path):
    familiar_faces = {}

    async def process_face(image_path):
        img = cv2.imread(image_path)
        image_height, image_width = img.shape[:2]
        boxes, scores, labels = await detect_faces_onnx(img, image_height, image_width)
        embeddings = []
        for box, score in zip(boxes, scores):
            if score > 0.6:
                x1, y1, x2, y2 = map(int, box[:4])
                cropped_face = img[y1:y2, x1:x2]
                cropped_face = cv2.resize(cropped_face, (112, 112))
                cropped_face_tensor = torch.tensor(cropped_face).permute(2, 0, 1).float() / 255.0
                cropped_face_tensor = cropped_face_tensor.unsqueeze(0)
                with torch.no_grad():
                    embedding = inception_resnet(cropped_face_tensor).squeeze(0).numpy()
                embedding /= np.linalg.norm(embedding)
                embeddings.append(embedding)
        return embeddings

    for familiar_name in os.listdir(folder_path):
        print(f"Processing {familiar_name}...")
        tasks = []
        for filename in os.listdir(f"{folder_path}/{familiar_name}"):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(folder_path, familiar_name, filename)
                tasks.append(process_face(image_path))
        embeddings = await asyncio.gather(*tasks)
        familiar_faces[familiar_name] = [e for sublist in embeddings for e in sublist]

    return familiar_faces


async def send_signal_to_esp32():
    """Send HTTP request to ESP32 to activate the light."""
    global last_signal_time
    current_time = time.time()
    if current_time - last_signal_time >= SIGNAL_COOLDOWN:
        async with httpx.AsyncClient() as client:
            try:
                await client.get(ESP32_URL)
                print("Signal sent to ESP32 to activate light.")
                last_signal_time = current_time
            except httpx.RequestError as e:
                print(f"Failed to send signal to ESP32: {e}")


async def process_frames():
    global frame_buffer
    global familiar_faces
    global unfamiliar_face

    retry_count = 0  # Retry counter for reconnecting to RTSP
    threshold=0.65

    while True:
        # Attempt to read a frame from the RTSP stream
        ret, frame = vcap.read()
        suspicious_detected = False

        if not ret or frame is None:
            print("Failed to retrieve frame from stream. Attempting to reconnect...")
            vcap.release()  # Close the current connection

            # Try reconnecting up to a maximum number of attempts
            while retry_count < 5:
                await asyncio.sleep(0.3)  # Wait a bit before retrying
                vcap.open(RTSP_URL)
                ret, frame = vcap.read()
                if ret:
                    print("Reconnected to RTSP stream.")
                    retry_count = 0  # Reset the retry counter
                    break
                retry_count += 1

            if retry_count >= 5:
                print("Failed to reconnect after multiple attempts. Restarting.")
                retry_count = 0
            continue

        try:
            image_height, image_width = frame.shape[:2]
            # Run ONNX model for detection
            boxes, scores, labels = await detect_faces_onnx(frame, image_height, image_width)

            for box, score in zip(boxes, scores):
                if score > 0.7:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Crop face
                    cropped_face = frame[y1:y2, x1:x2]
                    tmp = cropped_face
                    cropped_face = cv2.resize(cropped_face, (112, 112))

                    # cv2.imwrite(f"cropped_test.jpg", cropped_face)
                    cropped_face_tensor = torch.tensor(cropped_face).permute(2, 0, 1).float() / 255.0
                    cropped_face_tensor = cropped_face_tensor.unsqueeze(0)

                    # Generate embedding
                    with torch.no_grad():
                        embedding = inception_resnet(cropped_face_tensor).squeeze(0).numpy()

                    embedding /= np.linalg.norm(embedding)

                    # Check familiarity
                    label = await check_for_unfamiliar_face(embedding, familiar_faces, threshold)
                    # Label is in Vietnamese
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Check for suspicious faces
                    if label == "Doi tuong la":
                        suspicious_detected = True
                        # Save the unfamiliar face in the buffer
                        unfamiliar_face = tmp
                        break

        except Exception as e:
            print(e)
            pass

        # If any suspicious action is detected, send a signal to the ESP32
        if suspicious_detected:
            await send_signal_to_esp32()

        # Encode the annotated frame for display
        _, processed_img = cv2.imencode('.jpg', frame)
        frame_buffer = processed_img.tobytes()

        await asyncio.sleep(0.5)  # Adjust frame processing rate as needed


@app.on_event("startup")
async def on_startup() -> None:
    """Start the frame processing task."""
    asyncio.create_task(process_frames())


# WebSocket endpoint to stream images to the frontend
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global unfamiliar_face
    await websocket.accept()
    try:
        while True:
            if frame_buffer:
                # Encode the frame buffer as Base64
                encoded_frame = base64.b64encode(frame_buffer).decode('utf-8')
                await websocket.send_json({"type": "main", "data": encoded_frame})

            if unfamiliar_face is not None and unfamiliar_face.any():
                _, encoded_face = cv2.imencode('.jpg', unfamiliar_face)
                encoded_face_base64 = base64.b64encode(encoded_face).decode('utf-8')
                await websocket.send_json({"type": "unfamiliar", "data": encoded_face_base64})
                unfamiliar_face = None
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Serve the HTML page
@app.get("/", response_class=Response)
async def index():
    html_content = """
<!DOCTYPE html>
<html lang="vn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Auter</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            display: flex;
            flex-wrap: wrap;
            margin: 0;
            padding: 0;
            background-color: #121212;
            color: #ffffff;
        }
        #main {
            width: 80%;
            padding: 0px;
            display: flex;
            justify-content: center;
            flex-direction: column;
            align-items: center;
        }
        #sidebar {
            width: 20%;
            padding-top: 20px;
            background-color: #1e1e1e;
            border-left: 2px solid #333;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            height: 100vh;
            overflow-y: auto;
        }
        h1 {
            text-align: center;
            color: #00ccff;
            margin-bottom: 20px;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 1.8em;
        }
        h2 {
            text-align: center;
            color: #00ccff;
            margin-top: 0;
            font-size: 1.0em;
            text-transform: uppercase;
        }
        #faces {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
            width: 90%;
        }
        .face {
            width: 70%;
            border: 2px solid #00cc44;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
            background-color: #2a2a2a;
            box-shadow: 0px 4px 10px rgba(0, 255, 100, 0.2);
            animation: fadeIn 0.5s ease-in-out;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        img {
            width: 70%;
            height: auto;
            display: block;
            border-radius: 10px;
            border: 2px solid #555;
        }
        .placeholder {
            width: 70%;
            background-color: #2a2a2a;
            margin-bottom: 20px;
            border: 2px dashed #444;
            text-align: center;
            color: #777;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 14px;
            height: 150px;
            border-radius: 10px;
        }
        p {
            font-size: 14px;
            text-align: center;
            margin: 10px 0 0;
            color: #aaaaaa;
        }
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: scale(0.95);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        #imageStream {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 255, 0.2);
            border: 2px solid #555;
        }
        @media (max-width: 768px) {
            #main, #sidebar {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div id="main">
        <h1>Auter - Hệ thống giám sát và cảnh báo đột nhập</h1>
        <img id="imageStream" alt="Live Stream" />
    </div>
    <div id="sidebar">
        <h2>Đối tượng lạ đã phát hiện</h2>
        <div id="faces"></div>
    </div>
    <script>
        const mainImg = document.getElementById('imageStream');
        const facesContainer = document.getElementById('faces');
        const ws = new WebSocket("ws://" + window.location.host + "/ws");

        // Image stack of Unfamiliar faces, max 4
        const imageStack = [];
        const MAX_IMAGES = 4;

        // Timer to control updates
        let lastUpdateTime = 0;
        const UPDATE_INTERVAL = 10 * 1000; // 10 seconds in milliseconds

        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);

            if (message.type === "main") {
                // Handle the main stream
                const binaryData = atob(message.data);
                const arrayBuffer = new Uint8Array(binaryData.length);
                for (let i = 0; i < binaryData.length; i++) {
                    arrayBuffer[i] = binaryData.charCodeAt(i);
                }
                const blob = new Blob([arrayBuffer], { type: 'image/jpeg' });
                const objectURL = URL.createObjectURL(blob);
                mainImg.src = objectURL;
                mainImg.onload = function() {
                    URL.revokeObjectURL(objectURL);
                };
            } else if (message.type === "unfamiliar" && message.data !== null) {
                // Check if enough time has passed since the last update
                const currentTime = Date.now();
                if (currentTime - lastUpdateTime >= UPDATE_INTERVAL) {
                    lastUpdateTime = currentTime; // Update the last update time

                    // Handle unfamiliar faces
                    const binaryData = atob(message.data);
                    const arrayBuffer = new Uint8Array(binaryData.length);
                    for (let i = 0; i < binaryData.length; i++) {
                        arrayBuffer[i] = binaryData.charCodeAt(i);
                    }
                    const blob = new Blob([arrayBuffer], { type: 'image/jpeg' });
                    const objectURL = URL.createObjectURL(blob);

                    // Add new face to stack
                    imageStack.push(objectURL);
                    if (imageStack.length > MAX_IMAGES) {
                        imageStack.shift(); // Remove the oldest image if stack exceeds limit
                    }

                    // Update faces in the sidebar
                    facesContainer.innerHTML = ''; // Clear current faces
                    imageStack.forEach((url) => {
                        const faceContainer = document.createElement('div');
                        faceContainer.className = 'face';

                        const faceImg = document.createElement('img');
                        faceImg.src = url;

                        const timestamp = document.createElement('p');
                        timestamp.textContent = `Phát hiện vào ${new Date().toLocaleTimeString()}`;

                        faceContainer.appendChild(faceImg);
                        faceContainer.appendChild(timestamp);
                        facesContainer.appendChild(faceContainer);
                    });
                }
            }
            // Add placeholders if fewer than MAX_IMAGES
            while (facesContainer.childElementCount < MAX_IMAGES) {
                const placeholder = document.createElement('div');
                placeholder.className = 'placeholder';
                placeholder.textContent = 'Chưa phát hiện ra đối lượng lạ nào';
                facesContainer.appendChild(placeholder);
            }
        };

        ws.onclose = function() {
            console.log('WebSocket closed.');
        };
    </script>
</body>
</html>
    """
    return Response(content=html_content, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    import asyncio

    folder_path = 'familiar_faces'  # Thư mục chứa ảnh khuôn mặt
    print("Đang tải ảnh khuôn mặt trong CSDL...")

    # Tạo 1 event loop mới và chạy hàm async để tải ảnh khuôn mặt
    loop = asyncio.get_event_loop()
    familiar_faces = loop.run_until_complete(load_known_faces(folder_path))  # Chạy hàm async trong event loop
    print("Tải ảnh khuôn mặt hoàn tất.")

    # Chạy ứng dụng FastAPI
    uvicorn.run(app, host="192.168.1.104", port=8000)
