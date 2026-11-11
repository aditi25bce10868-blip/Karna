
# KARNA – Setup & User Instructions

## 1. Download the Project

1. Open the KARNA GitHub repository.
2. Click the **Code** button.
3. Select **Download ZIP**.
4. Extract the downloaded ZIP file to your computer.
5. Open the extracted KARNA project folder.

## 2. Install the Requirements

Open Terminal or Command Prompt and navigate to the extracted project directory.

```bash
cd path/to/KARNA
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Make sure Python and pip are installed before proceeding.

## 3. Launch the Application

Start the KARNA application using:

```bash
streamlit run app.py
```

Once launched, the application will open in your default web browser, displaying the **KARNA Operator Interface**.

## 4. Operator Identification

1. On the main operator screen, enter your operator identification, such as your name.
2. Click **Initiate System**.
3. The application will take you into the KARNA workspace dashboard.

The workspace provides access to the live audio recording, hazard classification and alert monitoring features.

## 5. Record Live Audio and Classify Hazards

1. Navigate to the audio detection section.
2. Click **Record Live Audio and Classify Hazard**.
3. Allow microphone access when prompted.
4. Record the surrounding audio that needs to be analysed.
5. Once recording is complete, the audio is passed through KARNA's backend audio-processing and machine-learning inference pipeline.
6. The system analyses the recording and displays the predicted sound category along with its confidence score.

When an emergency sound is detected, the system identifies the corresponding hazard and generates its associated alert event.

## 6. ESP32 and Wokwi Haptic Alert Simulation

The generated hazard event is bridged to the ESP32-based alert simulation.

1. The system sends the corresponding alert information to the ESP32 simulation.
2. The ESP32 processes the received event and activates the LED in the Wokwi simulation.
3. The LED blinks according to the number of vibration pulses assigned to the detected sound.
4. The corresponding output can be observed through the Wokwi Serial Monitor.

This simulation demonstrates how KARNA translates detected audio hazards into distinct haptic alert patterns.

**Note:** The LED is used only as a visual representation of the vibration pulses for prototype demonstration. The intended final product is a dedicated wearable wristband with a physical vibration motor.

---

## System Workflow

**Live Audio Input → Audio Processing → ML-Based Sound Classification → Hazard Detection → Pulse Mapping → ESP32 Alert Simulation → Wokwi LED and Serial Monitor**

KARNA aims to convert important environmental and emergency sounds into clear, distinguishable alerts, helping deaf and hard-of-hearing individuals become more aware of critical events in their surroundings.
