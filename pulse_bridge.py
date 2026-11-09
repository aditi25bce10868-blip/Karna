
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from models.small_cnn_stereo import SmallCNN
from transforms.audio import get_mel_transform, wav_to_logmel


ROOT = Path(__file__).resolve().parent

CHECKPOINTS = {
    "mono": ROOT / "artifacts/emergency_mono_control/best_emergency_1ch_model.pt",
    "stereo": ROOT / "artifacts/emergency_stereo/best_emergency_2ch_model.pt",
}


def load_audio(audio_path, channels, sample_rate, duration):
    """Load WAV and reproduce the training audio preprocessing."""

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, original_sr = sf.read(
        str(audio_path),
        always_2d=True,
        dtype="float32",
    )

    # soundfile returns [samples, channels]; PyTorch expects [channels, samples].
    wav = torch.from_numpy(audio).T.contiguous()

    # Match the selected training experiment.
    if channels == 1:
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
    else:
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

    # Resample.
    if original_sr != sample_rate:
        wav = torchaudio.functional.resample(
            wav,
            orig_freq=int(original_sr),
            new_freq=sample_rate,
        )

    # Pad or trim to the training duration.
    num_samples = int(duration * sample_rate)

    if wav.shape[1] < num_samples:
        wav = F.pad(wav, (0, num_samples - wav.shape[1]))
    else:
        wav = wav[:, :num_samples]

    return wav


def create_features(wav, checkpoint):
    """Generate per-channel log-Mel features exactly as in training."""

    sample_rate = int(checkpoint["sample_rate"])
    channels = int(checkpoint["input_channels"])

    mel_transform = get_mel_transform(
        sample_rate=sample_rate,
        n_fft=int(checkpoint["n_fft"]),
        hop_length=int(checkpoint["hop_length"]),
        n_mels=int(checkpoint["n_mels"]),
    )

    channel_features = []

    for ch in range(channels):
        feature = wav_to_logmel(
            wav[ch:ch + 1],
            sr=sample_rate,
            mel_transform=mel_transform,
            target_sr=sample_rate,
        )
        channel_features.append(feature.squeeze(0))

    # [channels, n_mels, time_frames]
    features = torch.stack(channel_features, dim=0)

    # Add batch dimension: [1, channels, n_mels, time_frames]
    return features.unsqueeze(0)


def load_model(checkpoint_path):
    """Load checkpoint metadata and model weights."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Check the checkpoint filename or pass --checkpoint."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("model_name") != "smallcnn":
        raise ValueError(
            f"Unsupported model: {checkpoint.get('model_name')}"
        )

    class_names = checkpoint["class_names"]
    input_channels = int(checkpoint["input_channels"])

    if input_channels not in (1, 2):
        raise ValueError(f"Unsupported input channel count: {input_channels}")

    model = SmallCNN(
        n_classes=len(class_names),
        in_channels=input_channels,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, class_names


def load_mapping(mapping_path):
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Pulse mapping not found: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def predict(model, features, class_names):
    with torch.inference_mode():
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_count = min(3, len(class_names))
    top_values, top_indices = torch.topk(probabilities, k=top_count)

    predicted_index = int(top_indices[0])
    predicted_class = class_names[predicted_index]

    top_predictions = [
        {
            "class_name": class_names[int(index)],
            "confidence": round(float(value), 6),
        }
        for value, index in zip(top_values, top_indices)
    ]

    return predicted_class, float(probabilities[predicted_index]), top_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Convert a prerecorded WAV classification into a pulse event."
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the prerecorded WAV file.",
    )

    parser.add_argument(
        "--model",
        choices=["mono", "stereo"],
        default="mono",
        help="Choose mono or stereo model. Default: mono.",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path.",
    )

    parser.add_argument(
        "--mapping",
        default=str(ROOT / "pulse_mapping.json"),
        help="Path to pulse_mapping.json.",
    )

    parser.add_argument(
        "--output",
        default=str(ROOT / "pulse_events/latest_event.json"),
        help="Where to save the generated event JSON.",
    )

    args = parser.parse_args()

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else CHECKPOINTS[args.model]
    )

    mapping_path = Path(args.mapping)
    output_path = Path(args.output)

    # Load selected model.
    model, checkpoint, class_names = load_model(checkpoint_path)

    expected_channels = 1 if args.model == "mono" else 2

    if int(checkpoint["input_channels"]) != expected_channels:
        raise ValueError(
            f"You selected {args.model}, but this checkpoint expects "
            f"{checkpoint['input_channels']} channel(s)."
        )

    # Verify class map if it exists beside the checkpoint.
    class_map_path = checkpoint_path.parent / "class_map.json"

    if class_map_path.is_file():
        with class_map_path.open("r", encoding="utf-8") as file:
            class_map = json.load(file)

        saved_names = [
            class_map[str(i)] for i in range(len(class_names))
        ]

        if saved_names != class_names:
            raise ValueError(
                "Checkpoint class_names and class_map.json do not match."
            )

    # Load mapping and audio.
    mapping = load_mapping(mapping_path)

    wav = load_audio(
        args.audio,
        channels=expected_channels,
        sample_rate=int(checkpoint["sample_rate"]),
        duration=float(checkpoint["duration"]),
    )

    features = create_features(wav, checkpoint)

    # Predict.
    predicted_class, confidence, top_predictions = predict(
        model,
        features,
        class_names,
    )

    if predicted_class not in mapping:
        raise KeyError(
            f"No pulse mapping exists for predicted class: {predicted_class}"
        )

    pattern = mapping[predicted_class]

    # Create simulator event. Pulse values remain unset until hardware validation.
    event = {
        "event_type": "PULSE_EVENT",
        "event_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_audio": str(Path(args.audio)),
        "model": {
            "selection": args.model,
            "checkpoint": str(checkpoint_path),
            "input_channels": expected_channels,
            "best_validation_macro_f1": float(
                checkpoint["best_validation_macro_f1"]
            ),
        },
        "prediction": {
            "class_name": predicted_class,
            "confidence": round(confidence, 6),
            "top_3": top_predictions,
        },
        "pulse": {
            "pattern_id": pattern["pattern_id"],
            "pulse_count": pattern.get("pulse_count"),
            "pulse_duration_ms": pattern.get("pulse_duration_ms"),
            "pause_duration_ms": pattern.get("pause_duration_ms"),
            "configuration_status": (
                "pending_hardware_validation"
                if any(
                    pattern.get(key) is None
                    for key in (
                        "pulse_count",
                        "pulse_duration_ms",
                        "pause_duration_ms",
                    )
                )
                else "configured"
            ),
        },
    }

    # Save event.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(event, file, indent=4)

    print("\n========== PULSE BRIDGE RESULT ==========")
    print(f"Model             : {args.model}")
    print(f"Audio             : {args.audio}")
    print(f"Predicted class   : {predicted_class}")
    print(f"Confidence        : {confidence:.4f}")
    print(f"Pattern ID        : {pattern['pattern_id']}")
    print(f"Pulse count       : {pattern.get('pulse_count')}")
    print(f"Pulse duration    : {pattern.get('pulse_duration_ms')}")
    print(f"Pause duration    : {pattern.get('pause_duration_ms')}")
    print(f"Event saved to    : {output_path}")
    print("=========================================\n")

    print(json.dumps(event, indent=4))


if __name__ == "__main__":
    main()
