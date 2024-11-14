import asyncio
import httpx
from typing import Dict
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, Response
from ultralytics import YOLO
import io
import time

app = FastAPI()

# YOLOv8 model
model_detect = YOLO("best.pt")
model_classify = YOLO("cls_yolov8_best.pt")
# model_detect.to('cuda')

# RTSP stream configuration
ESP_IP = "192.168.1.9"  # Replace with your ESP32 IP
ESP32_URL = f"http://{ESP_IP}/trigger_light"
RTSP_URL = f"rtsp://{ESP_IP}:5005/mjpeg/1"
vcap = cv2.VideoCapture(RTSP_URL)

# Buffer to hold the latest processed image
frame_buffer = None

# Rate limiting for ESP32 light activation signal
last_signal_time = 0
SIGNAL_COOLDOWN = 5  # Increase the cooldown to 5 seconds


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

# async def process_frames():
#     global frame_buffer
#     retry_count = 0  # Retry counter for reconnecting to RTSP

#     # Define actions and corresponding colors
#     actions = ["Normal", "Peaking", "Sneaking", "Stealing"]
#     colors = {
#         "Normal": (0, 255, 0),     # Green for "Normal"
#         "Peaking": (255, 255, 0),  # Yellow for "Peaking"
#         "Stealing": (0, 0, 255),   # Red for "Stealing"
#         "Sneaking": (255, 0, 0)    # Blue for "Sneaking"
#     }

#     while True:
#         # Attempt to read a frame from the RTSP stream
#         ret, frame = vcap.read()

#         if not ret or frame is None:
#             print("Failed to retrieve frame from stream. Attempting to reconnect...")
#             vcap.release()  # Close the current connection

#             # Try reconnecting up to a maximum number of attempts
#             while retry_count < 5:
#                 await asyncio.sleep(0.5)  # Wait a bit before retrying
#                 vcap.open(RTSP_URL)
#                 ret, frame = vcap.read()
#                 if ret:
#                     print("Reconnected to RTSP stream.")
#                     retry_count = 0  # Reset the retry counter
#                     break
#                 retry_count += 1

#             if retry_count >= 5:
#                 print("Failed to reconnect after multiple attempts. Restarting.")
#                 retry_count = 0
#             continue

#         # YOLO detection for "person" objects
#         results = model_detect(frame, verbose=False)
#         result = results[0]

#         # Loop through detected objects and filter for "person"
#         suspicious_action_detected = False
#         for box in result.boxes:
#             conf = box.conf
#             label = result.names[int(box.cls)]
#             if label == "person" and conf > 0.7:
#                 # Extract the bounding box coordinates
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])

#                 # Crop the image based on the bounding box
#                 cropped_person = frame[y1:y2, x1:x2]

#                 # Classify the action within the cropped area
#                 classify_result = model_classify(cropped_person, verbose=False)

#                 probs = classify_result[0].probs
#                 top_action_index = probs.top1  # Get the index of the action with the highest confidence
#                 action_label = actions[top_action_index]

#                 # Draw bounding box with corresponding color and label
#                 color = colors[action_label]
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, action_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#                 # Check for suspicious actions
#                 if action_label in ["Peaking", "Stealing", "Sneaking"]:
#                     suspicious_action_detected = True
#                     break

#         # If any suspicious action is detected, send a signal to the ESP32
#         if suspicious_action_detected:
#             await send_signal_to_esp32()

#         # Encode the annotated frame for display
#         _, processed_img = cv2.imencode('.jpg', frame)
#         frame_buffer = processed_img.tobytes()

#         await asyncio.sleep(0.5)  # Adjust frame processing rate as needed

async def process_frames():
    global frame_buffer
    retry_count = 0  # Retry counter for reconnecting to RTSP

    while True:
        # Attempt to read a frame from the RTSP stream
        ret, frame = vcap.read()

        if not ret or frame is None:
            print("Failed to retrieve frame from stream. Attempting to reconnect...")
            vcap.release()  # Close the current connection

            # Try reconnecting up to a maximum number of attempts
            while retry_count < 5:
                await asyncio.sleep(1)  # Wait a bit before retrying
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

        # YOLO detection for "person" objects
        results = model_detect(frame, verbose=False)
        result = results[0]

        # Loop through detected objects and filter for "person"
        for box in result.boxes:
            conf = box.conf
            label = result.names[int(box.cls)]
            if conf > 0.7:  # Adjust confidence threshold as needed
                # Extract the bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box around the detected person with a fixed color (e.g., blue)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Encode the annotated frame for display
        _, processed_img = cv2.imencode('.jpg', frame)
        frame_buffer = processed_img.tobytes()

        await asyncio.sleep(0.1)  # Adjust frame processing rate as needed

@app.on_event("startup")
async def on_startup() -> None:
    """Start the frame processing task."""
    asyncio.create_task(process_frames())


# WebSocket endpoint to stream images to the frontend
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send the latest frame from the buffer if available
            if frame_buffer:
                await websocket.send_bytes(frame_buffer)
            await asyncio.sleep(0.01)  # Control the WebSocket frame rate
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Serve the HTML page
@app.get("/", response_class=Response)
async def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live Video Stream</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
            }
            img {
                width: 50%;
                height: auto;
                margin: 20px auto;
                display: block;
                border: 1px solid #ccc;
            }
        </style>
    </head>
    <body>
        <h1>Live Object Detection Stream</h1>
        <img id="imageStream" alt="Live Stream" />
        <script>
            var imgElement = document.getElementById('imageStream');
            var ws = new WebSocket("ws://" + window.location.host + "/ws");

            ws.binaryType = 'blob';  // Receive binary data (images) as blobs

            ws.onmessage = function(event) {
                var blob = event.data;
                var objectURL = URL.createObjectURL(blob);
                imgElement.src = objectURL;
                imgElement.onload = function() {
                    URL.revokeObjectURL(objectURL);  // Free memory once the image is loaded
                };
            };

            ws.onclose = function(event) {
                console.log('WebSocket closed.');
            };
        </script>
    </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.1.6", port=8000)
