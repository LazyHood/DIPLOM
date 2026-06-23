"""
Модуль для встраивания и обнаружения водяных знаков в аудио
Использует высокочастотный диапазон 22-25 кГц (неслышимый для человека)
"""

import io
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.fft import fft, ifft
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import json


def _load_audio(path: str, **kwargs):
    """Загрузка аудио через BytesIO — обходит баг libsndfile с кириллицей на Windows"""
    p = Path(path)
    raw = p.read_bytes()
    return librosa.load(io.BytesIO(raw), **kwargs)


@dataclass
class WatermarkConfig:
    """Конфигурация параметров водяного знака"""
    freq_range: Tuple[int, int] = (20000, 21000)  # Диапазон частот (Гц)
    duration_ms: int = 100  # Длительность одного символа (мс)
    amplitude: float = 0.01  # Амплитуда водяного знака (1% от сигнала)
    pattern_length: int = 32  # Длина паттерна в битах


class AudioWatermark:
    """Класс для работы с водяными знаками в аудио"""

    def __init__(self, config: WatermarkConfig = None):
        self.config = config or WatermarkConfig()

    def generate_user_pattern(self, user_id: str) -> np.ndarray:
        """Генерирует уникальный бинарный паттерн для пользователя"""
        hash_obj = hashlib.sha256(user_id.encode())
        hash_bytes = hash_obj.digest()
        pattern = []
        for byte in hash_bytes[:self.config.pattern_length // 8]:
            for i in range(8):
                pattern.append((byte >> i) & 1)
        return np.array(pattern[:self.config.pattern_length])

    def embed_watermark(
        self,
        audio_path: str,
        output_path: str,
        user_id: str,
        method: str = 'frequency'
    ) -> dict:
        """Встраивает водяной знак в аудиофайл"""
        audio, sr = _load_audio(audio_path, sr=None, mono=False)

        if len(audio.shape) == 1:
            audio = audio.reshape(1, -1)
            is_mono = True
        else:
            is_mono = False

        pattern = self.generate_user_pattern(user_id)

        if method == 'frequency':
            watermarked = self._embed_frequency_domain(audio, pattern, sr)
        else:
            watermarked = self._embed_spread_spectrum(audio, pattern, sr)

        if is_mono:
            watermarked = watermarked[0]

        sf.write(output_path, watermarked.T, sr)

        return {
            'user_id': user_id,
            'pattern': pattern.tolist(),
            'sample_rate': sr,
            'duration': len(audio[0]) / sr,
            'method': method
        }

    def _embed_frequency_domain(
        self,
        audio: np.ndarray,
        pattern: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """Встраивание в частотной области (FSK-модуляция)"""
        channels, samples = audio.shape
        watermarked = audio.copy()

        bit_duration = int(sr * self.config.duration_ms / 1000)
        freq_step = (self.config.freq_range[1] - self.config.freq_range[0]) / 2
        freq_0 = self.config.freq_range[0]
        freq_1 = self.config.freq_range[0] + freq_step

        for i, bit in enumerate(pattern):
            start_sample = (i * bit_duration) % samples
            end_sample = min(start_sample + bit_duration, samples)

            freq = freq_1 if bit == 1 else freq_0
            t = np.arange(end_sample - start_sample) / sr
            watermark_signal = self.config.amplitude * np.sin(2 * np.pi * freq * t)

            window = signal.windows.hann(len(watermark_signal))
            watermark_signal *= window

            for ch in range(channels):
                watermarked[ch, start_sample:end_sample] += watermark_signal

        max_val = np.abs(watermarked).max()
        if max_val > 1.0:
            watermarked /= max_val

        return watermarked

    def _embed_spread_spectrum(
        self,
        audio: np.ndarray,
        pattern: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """Встраивание методом расширенного спектра (DSSS)"""
        channels, samples = audio.shape
        watermarked = audio.copy()

        np.random.seed(int(hashlib.md5(str(pattern).encode()).hexdigest(), 16) % (2**32))
        chip_rate = 100
        chips_per_bit = sr // chip_rate

        for ch in range(channels):
            for i, bit in enumerate(pattern):
                pn_sequence = np.random.randn(chips_per_bit)
                if bit == 0:
                    pn_sequence *= -1

                start = (i * chips_per_bit) % samples
                end = min(start + chips_per_bit, samples)
                length = end - start

                nyquist = sr / 2
                low = self.config.freq_range[0] / nyquist
                high = min(self.config.freq_range[1] / nyquist, 0.99)

                sos = signal.butter(4, [low, high], btype='band', output='sos')
                filtered_pn = signal.sosfilt(sos, pn_sequence[:length])

                watermarked[ch, start:end] += self.config.amplitude * filtered_pn

        return watermarked

    def detect_watermark(
        self,
        audio_path: str,
        user_id: Optional[str] = None,
        threshold: float = 0.6
    ) -> dict:
        """
        Обнаруживает водяной знак в аудиофайле.

        Логика:
        1. Если user_id задан — проверяем конкретный паттерн через корреляцию.
        2. Если user_id не задан — слепое обнаружение: сравниваем энергию
           в диапазоне водяного знака с энергией соседних полос.
           Дополнительно извлекаем наиболее вероятный паттерн.
        """
        audio, sr = _load_audio(audio_path, sr=None, mono=True)

        # Проверяем, достаточно ли высокая частота дискретизации
        nyquist = sr / 2
        if nyquist < self.config.freq_range[0]:
            return {
                'detected': False,
                'error': f'Sample rate {sr} Hz слишком низкий для анализа диапазона '
                         f'{self.config.freq_range[0]}-{self.config.freq_range[1]} Hz',
                'correlation': 0.0,
                'confidence': 0.0,
                'snr_db': 0.0,
            }

        if user_id:
            # --- Режим проверки конкретного паттерна ---
            pattern = self.generate_user_pattern(user_id)
            correlation = self._detect_pattern(audio, pattern, sr)
            detected = correlation > threshold

            return {
                'detected': detected,
                'user_id': user_id,
                'correlation': float(correlation),
                'threshold': threshold,
                'confidence': float(max(correlation, 0.0)),
                'snr_db': self._compute_snr(audio, sr),
            }
        else:
            # --- Слепое обнаружение по энергии полосы ---
            snr_db = self._compute_snr(audio, sr)
            # Порог SNR: если энергия в WM-полосе значимо выше фона — есть знак
            detected = snr_db > 3.0  # >3 дБ над соседними полосами

            extracted_pattern = self._extract_pattern(audio, sr)

            # Псевдо-корреляция: насколько решительно FSK декодер выбирал биты
            confidence = float(np.clip((snr_db - 3.0) / 10.0, 0.0, 1.0)) if detected else 0.0

            return {
                'detected': detected,
                'pattern': extracted_pattern.tolist(),
                'correlation': float(np.clip(snr_db / 13.0, 0.0, 1.0)),
                'confidence': confidence,
                'snr_db': snr_db,
                'method': 'blind_energy',
            }

    def _compute_snr(self, audio: np.ndarray, sr: int) -> float:
        """
        Вычисляет SNR (дБ) диапазона водяного знака относительно соседних полос.
        Возвращает значение > 0, если в WM-полосе аномально высокая энергия.
        """
        nyquist = sr / 2
        wm_low = self.config.freq_range[0]
        wm_high = self.config.freq_range[1]
        bw = wm_high - wm_low  # ширина полосы водяного знака

        # Соседние полосы той же ширины для сравнения
        ref_low1 = max(wm_low - bw, 100)
        ref_high1 = wm_low
        ref_low2 = wm_high
        ref_high2 = min(wm_high + bw, nyquist * 0.99)

        def band_energy(low, high):
            low_n = low / nyquist
            high_n = min(high / nyquist, 0.99)
            if low_n >= high_n:
                return 1e-10
            sos = signal.butter(6, [low_n, high_n], btype='band', output='sos')
            filtered = signal.sosfilt(sos, audio)
            return float(np.mean(filtered ** 2))

        e_wm = band_energy(wm_low, wm_high)
        e_ref1 = band_energy(ref_low1, ref_high1)
        e_ref2 = band_energy(ref_low2, ref_high2)
        e_ref = (e_ref1 + e_ref2) / 2 + 1e-12

        snr_db = 10 * np.log10(e_wm / e_ref)
        return float(snr_db)

    def _detect_pattern(
        self,
        audio: np.ndarray,
        pattern: np.ndarray,
        sr: int
    ) -> float:
        """
        Корреляционный детектор FSK: сравниваем энергию на freq_0 и freq_1
        в каждом временном слоте и сверяем с известным паттерном.
        """
        bit_duration = int(sr * self.config.duration_ms / 1000)
        freq_step = (self.config.freq_range[1] - self.config.freq_range[0]) / 2
        freq_0 = self.config.freq_range[0]
        freq_1 = self.config.freq_range[0] + freq_step

        detected_bits = []

        for i in range(len(pattern)):
            start = (i * bit_duration) % len(audio)
            end = min(start + bit_duration, len(audio))
            segment = audio[start:end]

            if len(segment) < 4:
                detected_bits.append(0)
                continue

            fft_result = np.fft.rfft(segment)
            freqs = np.fft.rfftfreq(len(segment), 1 / sr)

            # Энергия в узкой полосе ±50 Гц вокруг каждой частоты
            def freq_energy(f):
                mask = (freqs >= f - 50) & (freqs <= f + 50)
                return np.sum(np.abs(fft_result[mask]) ** 2) if mask.any() else 0.0

            energy_0 = freq_energy(freq_0)
            energy_1 = freq_energy(freq_1)

            detected_bits.append(1 if energy_1 > energy_0 else 0)

        detected_bits = np.array(detected_bits)

        # Корреляция Пирсона между ожидаемым и детектированным паттерном
        if len(detected_bits) < 2:
            return 0.0

        corr = np.corrcoef(pattern.astype(float), detected_bits.astype(float))[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def _extract_pattern(
        self,
        audio: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """Извлекает паттерн из аудио (слепое FSK-извлечение)"""
        bit_duration = int(sr * self.config.duration_ms / 1000)
        freq_step = (self.config.freq_range[1] - self.config.freq_range[0]) / 2
        freq_0 = self.config.freq_range[0]
        freq_1 = self.config.freq_range[0] + freq_step

        extracted_bits = []

        for i in range(self.config.pattern_length):
            start = (i * bit_duration) % len(audio)
            end = min(start + bit_duration, len(audio))
            segment = audio[start:end]

            if len(segment) < 4:
                extracted_bits.append(0)
                continue

            fft_result = np.fft.rfft(segment)
            freqs = np.fft.rfftfreq(len(segment), 1 / sr)

            def freq_energy(f):
                mask = (freqs >= f - 50) & (freqs <= f + 50)
                return np.sum(np.abs(fft_result[mask]) ** 2) if mask.any() else 0.0

            energy_0 = freq_energy(freq_0)
            energy_1 = freq_energy(freq_1)

            extracted_bits.append(1 if energy_1 > energy_0 else 0)

        return np.array(extracted_bits)