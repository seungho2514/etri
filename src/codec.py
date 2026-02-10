import torch
import torch.nn as nn
import torchaudio.functional as AF
import numpy as np

from encodec import EncodecModel
class Encodec:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = EncodecModel.encodec_model_24khz().to(device)
        self.model.set_target_bandwidth(6.0)
        self.model.eval()

    def encode_decode(self, audio, bitrate):
        """
        EnCodec 처리 파이프라인:
        1. 16kHz -> 24kHz 리샘플링 (EnCodec 요구사항)
        2. 압축 및 복원
        3. 24kHz -> 16kHz 리샘플링
        4. 길이 맞추기
        """
        self.model.bandwidth(bitrate)

        wav_24k = AF.resample(audio, 16000, 24000)

        with torch.no_grad():
            encoded = self.model.encode(wav_24k)
            decoded = self.model.decode(encoded)

        wav_16k = AF.resample(decoded, 24000, 16000)

        target_len = audio.shape[-1]
        current_len = wav_16k.shape[-1]

        if current_len > target_len:
            wav_16k = wav_16k[..., :target_len]
        elif current_len < target_len:
            wav_16k = torch.nn.functional.pad(wav_16k, (0, target_len - current_len))
            
        return wav_16k