#include <WiFi.h>
#include <WebServer.h>
#include <WiFiClient.h>
#include "OV2640.h"
#include "OV2640Streamer.h"
#include "CRtspSession.h"

OV2640 cam;
const int lightPin = 4;  // Pin connected to the light
// const char* ssid = "IEC Server";
// const char* password = "IEC@2024";
const char* ssid = "Interdimensional_Wifi";
const char* password = "0974603489";

WiFiServer rtspServer(5005);
WebServer webServer(80);  // Create a web server on port 80

void setup()
{
  Serial.begin(115200);
  pinMode(lightPin, OUTPUT);
  digitalWrite(lightPin, LOW);  // Keep the light off initially
  Serial.setDebugOutput(true);
  Serial.println();

  // WiFi.config(ip, gatway, subnet);

  // attempt to connect to Wifi network:
  while (WiFi.status() != WL_CONNECTED)
  {
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(ssid);
    // Connect to WPA/WPA2 network. Change this line if using open or WEP network:
    WiFi.begin(ssid, password);

    // wait 10 seconds for connection:
    delay(10000);
  }
  Serial.println("WiFi connected");

  Serial.println("ESP32-CAM IP Address: ");
  Serial.println(WiFi.localIP());

  esp_err_t err = cam.init(esp32cam_aithinker_config);
  if (err != ESP_OK)
  {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  WiFi.setTxPower(WIFI_POWER_19_5dBm);  // Set maximum WiFi transmission power
  rtspServer.begin();

  // Set up a handler for the "/trigger_light" URL
  webServer.on("/trigger_light", []() {
    digitalWrite(lightPin, HIGH);  // Turn on the light
    delay(500);                    // Keep it on for 500ms
    digitalWrite(lightPin, LOW);    // Turn off the light
    webServer.send(200, "text/plain", "Light activated");
  });

  webServer.begin();  // Start the web server
}

CStreamer *streamer;
CRtspSession *session;
WiFiClient client; // FIXME, support multiple clients

void loop()
{
  uint32_t msecPerFrame = 400; // Adjust frame rate to 2.5 FPS
  static uint32_t lastimage = millis();
  webServer.handleClient();  // Process web server requests

  // If we have an active client connection, just service that until gone
  // (FIXME - support multiple simultaneous clients)
  if (session)
  {
    session->handleRequests(0); // we don't use a timeout here,
    // instead we send only if we have new enough frames

    uint32_t now = millis();
    if (now > lastimage + msecPerFrame || now < lastimage)
    { // handle clock rollover
      session->broadcastCurrentFrame(now);
      lastimage = now;

      // check if we are overrunning our max frame rate
      now = millis();
      if (now > lastimage + msecPerFrame)
        printf("warning exceeding max frame rate of %d ms\n", now - lastimage);
    }

    if (session->m_stopped)
    {
      delete session;
      delete streamer;
      session = NULL;
      streamer = NULL;
    }
  }
  else
  {
    client = rtspServer.accept();

    if (client)
    {
      //streamer = new SimStreamer(&client, true);             // our streamer for UDP/TCP based RTP transport
      streamer = new OV2640Streamer(&client, cam); // our streamer for UDP/TCP based RTP transport

      session = new CRtspSession(&client, streamer); // our threads RTSP session and state
    }
  }
}
