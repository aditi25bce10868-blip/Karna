
# KARNA – Emergency Sound Classification & Pulse Bridge

KARNA is an emergency and environmental sound classification prototype that identifies important sounds from audio recordings and converts the prediction into a structured pulse event for a haptic simulator.

The current prototype supports prerecorded WAV input. It performs model inference and generates a JSON event; it does not directly control a physical haptic device.

## Supported Sound Classes

1. ambulance_siren
2. fire_truck_siren
3. long_horn_ahead
4. long_horn_alongside
5. long_horn_behind
6. short_horn_ahead
7. short_horn_alongside
8. short_horn_behind
9. traffic_noise
10. varied_horn_ahead
11. varied_horn_alongside
12. varied_horn_behind

## Features

- Emergency and environmental sound classification.
- Mono and stereo CNN model inference.
- WAV audio preprocessing and feature extraction.
- Prediction confidence and top-three predictions.
- Class-to-pattern mapping through `pulse_mapping.json`.
- Structured JSON pulse events for simulator integration.
- Pulse timing fields intentionally left unconfigured until hardware validation.

## Project Structure

```text
karna/
│
├── pulse_bridge.py
├── pulse_mapping.json
├── README.md
├── requirements.txt
│
├── datasets/
│   ├── emergency_audio.py
│   ├── emergency_audio_stereo.py
│   └── other dataset loaders
│
├── models/
│   ├── small_cnn.py
│   └── small_cnn_stereo.py
│
├── transforms/
│   └── audio.py
│
├── artifacts/
│   ├── emergency_mono_control/
│   │   ├── best_emergency_1ch_model.pt
│   │   └── class_map.json
│   │
│   └── emergency_stereo/
│       ├── best_emergency_2ch_model.pt
│       └── class_map.json
│
├── emergency_datset/
│   ├── Dataset/                  # Original audio dataset (not committed)
│   ├── dataset_preparation/
│   ├── audit reports/            # Generated reports (not committed)
│   └── preparation scripts
│
└── pulse_events/
    ├── mono_event.json
    └── stereo_event.json
```

This structure highlights the main project components. Other existing source files and scripts remain part of the repository.

## Models and Evaluation

Two model configurations were evaluated using the same dataset split.

| Metric | Mono model | Stereo model |
|---|---:|---:|
| Clean validation macro F1 | 67.20% | 66.49% |
| Clean test macro F1 | 41.85% | 42.49% |
| Noisy test emergency-only macro F1 | 36.09% | 39.66% |

The mono model is the primary baseline because stereo did not demonstrate a decisive overall advantage. The stereo checkpoint is retained as an experimental alternative.

These results indicate that generalisation, particularly on unseen test recordings, still needs improvement. They should not be interpreted as proof of reliable real-world emergency detection.

## Requirements

Python dependencies are listed in `requirements.txt`.

Install them using:

```bash
pip install -r requirements.txt
```

The inference bridge uses PyTorch, torchaudio and SoundFile.

## Running Inference

Run the mono model:

```bash
python pulse_bridge.py --audio "path/to/audio.wav" --model mono
```

Run the stereo model:

```bash
python pulse_bridge.py --audio "path/to/audio.wav" --model stereo
```

To save separate output events:

```bash
python pulse_bridge.py --audio "path/to/audio.wav" --model mono --output "pulse_events/mono_event.json"

python pulse_bridge.py --audio "path/to/audio.wav" --model stereo --output "pulse_events/stereo_event.json"
```

The default output location is:

```text
pulse_events/latest_event.json
```

## Pulse Event Mapping

`pulse_mapping.json` maps each predicted sound class to a stable pattern ID (`P01`–`P12`).

The generated event contains:

- Predicted sound class.
- Prediction confidence.
- Top-three model predictions.
- Pattern ID.
- Pulse count.
- Pulse duration.
- Pause duration.

Pulse count and timing values are currently `null`. They must be filled in only after the haptic simulator or hardware developer verifies the actual pulse patterns.

Pattern IDs are identifiers, not pulse instructions.

## Current Prototype Limitations

- Input is prerecorded WAV audio; continuous microphone streaming is not implemented in this bridge.
- No direct physical haptic hardware actuation is implemented.
- Pulse timings and counts are pending simulator/hardware validation.
- Classification performance, particularly for horn direction and unseen recordings, needs further improvement.
- Real-world testing in varied traffic and emergency environments remains necessary.

## Project Goal

KARNA aims to bridge emergency sound recognition and haptic feedback through an audio classification model and a structured event interface. The current implementation establishes the inference-to-event pipeline for further simulator and hardware integration.
