
"""Dataset loader for controlled mono/stereo emergency audio experiments."""

from pathlib import Path
from typing import List, Tuple

import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F

from transforms.audio import get_mel_transform, wav_to_logmel


def resolve_audio_path(value: str, project_root: Path) -> Path:
    path = Path(str(value))

    candidates = [
        path,
        project_root / path,
        project_root / "emergency_datset" / path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Audio file not found: {value}")


class EmergencyAudioDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        records: List[Tuple[str, int]],
        project_root: str,
        target_sr=16000,
        duration=3.0,
        n_mels=64,
        n_fft=1024,
        hop_length=256,
        channels=1,
    ):

        if channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")

        self.records = records
        self.project_root = Path(project_root)
        self.target_sr = target_sr
        self.num_samples = int(duration * target_sr)
        self.channels = channels

        self.mel_transform = get_mel_transform(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def __len__(self):
        return len(self.records)

    def _load_wave(self, raw_path):

        path = resolve_audio_path(raw_path, self.project_root)

        try:
            audio, sr = sf.read(
                str(path),
                always_2d=True,
                dtype="float32",
            )

            wav = torch.from_numpy(audio).T.contiguous()

        except Exception:
            wav, sr = torchaudio.load(
                str(path),
                backend="soundfile",
            )

        # MONO EXPERIMENT
        if self.channels == 1:

            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

        # STEREO EXPERIMENT
        else:

            if wav.shape[0] == 1:
                # Duplicate mono recordings into two identical channels.
                wav = wav.repeat(2, 1)

            elif wav.shape[0] > 2:
                wav = wav[:2]

        if int(sr) != self.target_sr:

            wav = torchaudio.functional.resample(
                wav,
                orig_freq=int(sr),
                new_freq=self.target_sr,
            )

        if wav.shape[1] < self.num_samples:

            wav = F.pad(
                wav,
                (0, self.num_samples - wav.shape[1]),
            )

        else:
            wav = wav[:, :self.num_samples]

        return wav

    def __getitem__(self, index):

        path, label = self.records[index]

        wav = self._load_wave(path)

        # Calculate log-Mel spectrogram independently for each channel.
        channel_features = []

        for ch in range(self.channels):

            feature = wav_to_logmel(
                wav[ch:ch + 1],
                sr=self.target_sr,
                mel_transform=self.mel_transform,
            )

            channel_features.append(feature.squeeze(0))

        log_mel = torch.stack(channel_features, dim=0)

        return log_mel, int(label)


def build_records(frame, path_column, class_to_index):

    if path_column not in frame.columns:
        raise ValueError(
            f"Column '{path_column}' not found. "
            f"Available columns: {list(frame.columns)}"
        )

    records = []

    for _, row in frame.iterrows():

        name = str(row["class_name"])

        if name in class_to_index:

            records.append(
                (
                    str(row[path_column]),
                    class_to_index[name],
                )
            )

    return records
