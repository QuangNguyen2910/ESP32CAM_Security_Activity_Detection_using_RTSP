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
ESP_IP = "192.168.1.139"
RTSP_URL = f"rtsp://admin:L2345074@{ESP_IP}:37777/cam/realmonitor?channel=1&subtype=0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")  # Debug GPU usage

app = FastAPI()

# Model setup with GPU support
onnx_providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
onnx_session = ort.InferenceSession("best.onnx", providers=onnx_providers)
inception_resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

vcap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
frame_buffer = None
familiar_faces = None
unfamiliar_face = None
last_signal_time = 0
SIGNAL_COOLDOWN = 10

# Preprocess image for ONNX (unchanged)
async def preprocess_image(image, input_size=(320, 320)):
    def sync_preprocess(image):
        image = cv2.resize(image, input_size)
        image_data = image.astype(np.float32) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        return np.expand_dims(image_data, axis=0)
    return await asyncio.to_thread(sync_preprocess, image)

# Detect faces with ONNX (unchanged)
async def detect_faces_onnx(image, image_height, image_width, input_size=(320, 320), confidence_threshold=0.6):
    input_data = await preprocess_image(image, input_size)
    input_name = onnx_session.get_inputs()[0].name
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

# Compare embeddings (unchanged)
async def compare_embeddings(embedding1, embedding2):
    return await asyncio.to_thread(cosine, embedding1, embedding2)

# Check unfamiliar face (unchanged)
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

# Load known faces with GPU support
async def load_known_faces(folder_path):
    global familiar_faces
    familiar_faces_local = {}
    async def process_face(image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return []
        image_height, image_width = img.shape[:2]
        boxes, scores, labels = await detect_faces_onnx(img, image_height, image_width)
        embeddings = []
        for box, score in zip(boxes, scores):
            if score > 0.6:
                x1, y1, x2, y2 = map(int, box[:4])
                cropped_face = img[y1:y2, x1:x2]
                cropped_face = cv2.resize(cropped_face, (112, 112))
                cropped_face_tensor = torch.tensor(cropped_face).permute(2, 0, 1).float() / 255.0
                cropped_face_tensor = cropped_face_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = inception_resnet(cropped_face_tensor).squeeze(0).cpu().numpy()
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
        familiar_faces_local[familiar_name] = [e for sublist in embeddings for e in sublist if e is not None]
    familiar_faces = familiar_faces_local

# Send signal to ESP32 (unchanged)
async def send_signal_to_esp32():
    global last_signal_time
    current_time = time.time()
    if current_time - last_signal_time >= SIGNAL_COOLDOWN:
        async with httpx.AsyncClient() as client:
            try:
                print("Signal sent to ESP32 to activate light.")
                last_signal_time = current_time
            except httpx.RequestError as e:
                print(f"Failed to send signal to ESP32: {e}")

# Process frames with robust stream handling
async def process_frames():
    global frame_buffer, familiar_faces, unfamiliar_face
    retry_count = 0
    threshold = 0.65

    while True:
        ret, frame = vcap.read()
        suspicious_detected = False

        if not ret or frame is None:
            print("Failed to retrieve frame. Attempting to reconnect...")
            vcap.release()
            while retry_count < 5:
                await asyncio.sleep(1)  # Increased delay for stability
                vcap.open(RTSP_URL)
                ret, frame = vcap.read()
                if ret and frame is not None:
                    print("Reconnected to RTSP stream.")
                    retry_count = 0
                    break
                retry_count += 1
            if retry_count >= 5:
                print("Max retries reached. Using placeholder frame.")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder black frame
                cv2.putText(frame, "Stream Offline", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            continue

        try:
            image_height, image_width = frame.shape[:2]
            boxes, scores, labels = await detect_faces_onnx(frame, image_height, image_width)
            for box, score in zip(boxes, scores):
                if score > 0.7:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cropped_face = frame[y1:y2, x1:x2]
                    tmp = cropped_face
                    cropped_face = cv2.resize(cropped_face, (112, 112))
                    cropped_face_tensor = torch.tensor(cropped_face).permute(2, 0, 1).float() / 255.0
                    cropped_face_tensor = cropped_face_tensor.unsqueeze(0).to(device)  # Fix GPU mismatch
                    with torch.no_grad():
                        embedding = inception_resnet(cropped_face_tensor).squeeze(0).cpu().numpy()
                    embedding /= np.linalg.norm(embedding)
                    label = await check_for_unfamiliar_face(embedding, familiar_faces, threshold)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if label == "Doi tuong la":
                        suspicious_detected = True
                        unfamiliar_face = tmp
                        break
        except Exception as e:
            print(f"Processing error: {e}")

        if suspicious_detected:
            await send_signal_to_esp32()

        _, processed_img = cv2.imencode('.jpg', frame)
        frame_buffer = processed_img.tobytes()
        await asyncio.sleep(0.1)  # Faster update rate

# Startup event (modern lifespan recommended later)
@app.on_event("startup")
async def on_startup():
    asyncio.create_task(process_frames())

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global frame_buffer, unfamiliar_face
    await websocket.accept()
    try:
        while True:
            if frame_buffer is not None:
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

# Serve HTML (unchanged)
@app.get("/", response_class=Response)
async def index():
    html_content = """[Your HTML content unchanged]"""
    return Response(content=html_content, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    folder_path = 'familiar_faces'
    print("Đang tải ảnh khuôn mặt trong CSDL...")
    asyncio.run(load_known_faces(folder_path))
    print("Tải ảnh khuôn mặt hoàn tất.")
    uvicorn.run(app, host="192.168.1.16", port=8000)