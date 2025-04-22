#include <WiFi.h>
#include <WebServer.h>
#include <WiFiClient.h>
#include "OV2640.h"
#include "OV2640Streamer.h"
#include "CRtspSession.h"


// const char* ssid = "IEC Server";
// const char* password = "IEC@2024";
// const char* ssid = "Interdimensional_Wifi";
// const char* password = "0974603489";
const char* ssid = "Interdimensional_Wifi2";
const char* password = "0974603489";

WiFiServer rtspServer(5005); // Tạo rstp server tại port 80
WebServer webServer(80);  // Tạo web server tại port 80

OV2640 cam;
const int lightPin = 4; 

CStreamer *streamer; // Đối tượng phát video qua RTSP
CRtspSession *session; // Quản lý phiên RTSP
WiFiClient client; // Xử lý kết nối giữa client với RTSP server


void setup() {
  Serial.begin(115200);
  pinMode(lightPin, OUTPUT);
  digitalWrite(lightPin, LOW);  // Ban đầu đèn sẽ tắt
  Serial.setDebugOutput(true);
  Serial.println();
  // Kết nối WIFI
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(ssid);
    WiFi.begin(ssid, password);
    delay(10000); // Đợi 10 giây để kết nối
  }
  Serial.println("WiFi connected");
  Serial.println("ESP32-CAM IP Address: ");
  Serial.println(WiFi.localIP());


  esp_err_t err = cam.init(esp32cam_aithinker_config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }


  WiFi.setTxPower(WIFI_POWER_19_5dBm);  // Cài đặt truyền tải dữ liệu WIFI tối đa
  rtspServer.begin(); // Khởi chạy rstp server
  // Thiết lập 1 api endpoint "/trigger_light" để bật/tắt đèn
  webServer.on("/trigger_light", []() {
    digitalWrite(lightPin, HIGH);  // Bật đèn
    delay(500);                    // Giữ đèn bật trong 500ms
    digitalWrite(lightPin, LOW);   // Tắt đèn
    webServer.send(200, "text/plain", "Light activated");
  });
  webServer.begin();  // Khởi chạy web server
}


void loop() {
  uint32_t msecPerFrame = 400; // Điều chỉnh tốc độ khung hình (2.5 FPS)
  static uint32_t lastimage = millis();
  webServer.handleClient();  // Xử lý các yêu cầu từ WebServer
  // Nếu có kết nối RTSP client đang hoạt động
  if (session) {
    session->handleRequests(0); // Gửi khung hình (frame) mới
    uint32_t now = millis();
    if (now > lastimage + msecPerFrame || now < lastimage) { 
      session->broadcastCurrentFrame(now); // Phát khung hình hiện tại
      lastimage = now;
      now = millis();
      if (now > lastimage + msecPerFrame)
        printf("warning exceeding max frame rate of %d ms\n", now - lastimage); // Cảnh báo nếu vượt quá tốc độ khung hình tối đa
    }
    if (session->m_stopped) { // Nếu kết nối RTSP bị ngừng
      delete session;
      delete streamer;
      session = NULL;
      streamer = NULL;
    }
  } else {
    client = rtspServer.accept();
    if (client) {
      streamer = new OV2640Streamer(&client, cam); // Tạo streamer cho kết nối
      session = new CRtspSession(&client, streamer); // Tạo phiên RTSP
    }
  }
}

