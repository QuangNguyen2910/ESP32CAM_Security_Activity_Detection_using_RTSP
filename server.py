import asyncio
from typing import Dict
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, Response
from ultralytics import YOLO
import io

app = FastAPI()

# YOLOv8 model
model = YOLO("best.pt")
# model.to('cuda')

# RTSP stream configuration
ESP_IP = "192.168.1.37"  # Replace with your ESP32 IP
RTSP_URL = f"rtsp://{ESP_IP}:5005/mjpeg/1"
vcap = cv2.VideoCapture(RTSP_URL)

# Buffer to hold the processed image
processed_image_data: bytes = None

async def process_frames():
    global processed_image_data

    while True:
        # Read frame from RTSP stream
        ret, frame = vcap.read()
        if not ret or frame is None:
            print("Failed to retrieve frame from stream.")
            await asyncio.sleep(1)  # Wait before retrying
            continue

        # Run YOLO model on the frame (optional)
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Encode the frame to JPEG and store it in memory
        _, processed_img = cv2.imencode('.jpg', annotated_frame)
        processed_image_data = processed_img.tobytes()
        
        await asyncio.sleep(0.05)  # Adjust frame processing rate as needed

# Start processing frames in the background
@app.on_event("startup")
async def on_startup() -> None:
    asyncio.create_task(process_frames())

# WebSocket endpoint to stream images to the frontend
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send the processed image if available
            if processed_image_data:
                await websocket.send_bytes(processed_image_data)
            await asyncio.sleep(0.05)  # Control frame rate
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
    uvicorn.run(app, host="192.168.1.8", port=8000)