# ESP32CAM_Security_Activity_Detection_using_RTSP (Security Alarm System using ESP32CAM and FastAPI)

## Description
This repository contains the implementation of a security alarm system built using FastAPI as the server framework. The system leverages an ESP32CAM module to stream live images via RTSP to the server. The server processes these images using advanced machine learning models to detect and analyze faces. If an unfamiliar or suspicious face is detected, the system triggers an alert by calling an HTTP API endpoint on the ESP32CAM, causing an LED to flash as a warning signal.

## Features
- **Live Face Detection**: Real-time face detection using YOLOv10.
- **Face Recognition**: Familiar face matching using FaceNet embeddings.
- **Alert Mechanism**: LED flashing via ESP32CAM's HTTP API endpoint for unfamiliar faces.
- **User Interface**: A FastAPI-based web interface for monitoring detected faces and managing the familiar faces database.

## System Architecture
1. **ESP32CAM**: Streams live images using RTSP to the FastAPI server.
2. **Server (FastAPI)**:
   - Processes live image streams.
   - Uses YOLOv10 for face detection.
   - Extracts face embeddings using FaceNet.
   - Matches embeddings against the database of familiar faces.
   - Triggers alerts for unfamiliar faces.
3. **ML Models**:
   - YOLOv10 trained on a Kaggle dataset for face detection.
   - FaceNet for generating face embeddings.

## Setup and Installation

### Requirements
- ESP32CAM module
- Python 3.8+
- FastAPI framework

### Hardware Setup
1. Flash the `esp32_rtsp.ino` code onto your ESP32CAM module.
2. Ensure the ESP32CAM is connected to a Wi-Fi network.

### Software Setup
1. Start the FastAPI server:
   ```bash
   python server.py
   ```

## Usage
1. Deploy the ESP32CAM with the `esp32_rtsp.ino` code.
2. Run the `server.py` script to start the FastAPI server.
3. Access the FastAPI web interface at `http://localhost:8000` to monitor detected faces and manage the familiar faces database.
4. When an unfamiliar face is detected, the ESP32CAM's LED will flash as an alert.

## Files
- `esp32_rtsp.ino`: Arduino sketch for configuring the ESP32CAM to stream images via RTSP and respond to HTTP API requests.
- `server.py`: FastAPI server implementation.

## Models
- **YOLOv10**: Used for face detection, trained on a Kaggle dataset.
- **FaceNet**: Used for generating face embeddings and similarity calculations.

## Future Improvements
- Add support for multiple ESP32CAM devices.
- Improve the accuracy of face detection and recognition models. Using better embedding model like InsightFace,...
- Enhance the user interface with additional features such as notification systems. Design a better looking and user-friendly interface.

## License
This project is licensed under the MIT License.
