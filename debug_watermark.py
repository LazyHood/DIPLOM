"""
Диагностика: встраиваем и сразу проверяем, выводим все числа
Запускай: python debug_watermark.py "файл.wav"
"""
import sys
import io
import numpy as np
from pathlib import Path


def load_audio_safe(path_str):
    """Загружает аудио через BytesIO — обходит баг libsndfile с кириллицей на Windows"""
    import librosa
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path_str}")
    raw = path.read_bytes()
    buf = io.BytesIO(raw)
    audio, sr = librosa.load(buf, sr=None, mono=True)
    return audio, sr


def diagnose(input_path_str):
    from watermark_system import AudioWatermark, WatermarkConfig
    from scipy import signal as sig
    import librosa

    config = WatermarkConfig(
        freq_range=(20000, 21000),
        duration_ms=100,
        amplitude=0.01,
        pattern_length=32
    )
    wm = AudioWatermark(config)

    print(f"\n=== ДИАГНОСТИКА ===")
    print(f"Входной файл: {input_path_str}")

    try:
        audio_orig, sr = load_audio_safe(input_path_str)
    except Exception as e:
        print(f"ОШИБКА загрузки: {e}")
        return

    print(f"Sample rate:  {sr} Hz")
    print(f"Nyquist:      {sr // 2} Hz")
    print(f"Длительность: {len(audio_orig) / sr:.2f} сек")
    print(f"WM диапазон:  {config.freq_range[0]}–{config.freq_range[1]} Hz")

    if sr // 2 < config.freq_range[0]:
        print(f"\n!!! ПРОБЛЕМА: Sample rate {sr} Hz слишком низкий!")
        print(f"    Nyquist = {sr // 2} Hz < {config.freq_range[0]} Hz")
        print(f"    Водяной знак физически не может быть встроен/обнаружен.")
        print(f"\n    РЕШЕНИЕ: снизь freq_range до {sr // 2 - 3000}–{sr // 2 - 200} Hz в настройках.")
        return

    # Встраиваем — патчим librosa.load внутри watermark_system через BytesIO
    _orig_load = librosa.load

    def patched_load(path, **kwargs):
        p = Path(str(path))
        if p.exists():
            raw = p.read_bytes()
            return _orig_load(io.BytesIO(raw), **kwargs)
        return _orig_load(path, **kwargs)

    librosa.load = patched_load

    try:
        output_path = str(Path("_debug_watermarked.wav").resolve())
        result = wm.embed_watermark(input_path_str, output_path, "test_user", method="frequency")
        print(f"\n--- Встраивание ---")
        print(f"User ID:  {result['user_id']}")
        print(f"Паттерн:  {''.join(map(str, result['pattern'][:32]))}")

        audio_wm, sr2 = load_audio_safe(output_path)
        print(f"\n--- Сохранённый файл ---")
        print(f"Sample rate: {sr2} Hz  |  Сэмплов: {len(audio_wm)}")

        # Энергии
        def band_energy(audio, sr, low, high):
            nyq = sr / 2
            l, h = low / nyq, min(high / nyq, 0.99)
            if l >= h:
                return 0.0
            sos = sig.butter(6, [l, h], btype='band', output='sos')
            return float(np.mean(sig.sosfilt(sos, audio) ** 2))

        bw = config.freq_range[1] - config.freq_range[0]
        e_wm   = band_energy(audio_wm, sr2, config.freq_range[0], config.freq_range[1])
        e_ref1 = band_energy(audio_wm, sr2, config.freq_range[0] - bw, config.freq_range[0])
        e_ref2 = band_energy(audio_wm, sr2, config.freq_range[1], min(config.freq_range[1] + bw, sr2 // 2 - 100))
        e_ref  = (e_ref1 + e_ref2) / 2 + 1e-12
        snr    = 10 * np.log10(e_wm / e_ref)

        print(f"\n--- Энергии ---")
        print(f"WM полоса (после):    {e_wm:.2e}")
        print(f"Соседняя ниже:        {e_ref1:.2e}")
        print(f"Соседняя выше:        {e_ref2:.2e}")
        print(f"SNR:                  {snr:.2f} дБ  (порог в коде: 3 дБ)")

        # Детекция
        detect_result = wm.detect_watermark(output_path, "test_user", threshold=0.3)
        print(f"\n--- Детекция (порог корреляции=0.3) ---")
        print(f"Обнаружен:   {detect_result['detected']}")
        print(f"Корреляция:  {detect_result.get('correlation', 0):.4f}")
        print(f"SNR:         {detect_result.get('snr_db', 0):.2f} дБ")

        # Побитовый анализ
        pattern    = np.array(result['pattern'])
        bit_dur    = int(sr2 * config.duration_ms / 1000)
        freq_0     = config.freq_range[0]
        freq_1     = freq_0 + (config.freq_range[1] - config.freq_range[0]) / 2

        print(f"\n--- Побитовый FSK (первые 8 бит) ---")
        print(f"{'#':>3} {'ожид':>5} {'детект':>7} {'E_f0':>12} {'E_f1':>12} {'':>3}")
        matches = 0
        for i in range(min(8, len(pattern))):
            s = (i * bit_dur) % len(audio_wm)
            e = min(s + bit_dur, len(audio_wm))
            seg = audio_wm[s:e]
            fr  = np.fft.rfft(seg)
            fqs = np.fft.rfftfreq(len(seg), 1 / sr2)

            def fe(f):
                mask = (fqs >= f - 50) & (fqs <= f + 50)
                return float(np.sum(np.abs(fr[mask]) ** 2)) if mask.any() else 0.0

            e0, e1 = fe(freq_0), fe(freq_1)
            db = 1 if e1 > e0 else 0
            ok = "✓" if db == pattern[i] else "✗"
            if db == pattern[i]: matches += 1
            print(f"{i:>3} {int(pattern[i]):>5} {db:>7} {e0:>12.2e} {e1:>12.2e} {ok:>3}")

        print(f"Совпадений: {matches}/8")

    finally:
        librosa.load = _orig_load

    # Итог
    print(f"\n=== ИТОГ ===")
    if sr // 2 < config.freq_range[1]:
        print(f"⚠  Nyquist ({sr//2} Hz) < верхней границы WM ({config.freq_range[1]} Hz)")
        print(f"   → В настройках GUI снизь верхнюю частоту до {sr // 2 - 200} Hz")
    else:
        print(f"✓  Sample rate достаточный ({sr} Hz)")

    if snr > 3:
        print(f"✓  WM физически присутствует (SNR={snr:.1f} дБ)")
    elif snr > 0:
        print(f"⚠  WM едва различим (SNR={snr:.1f} дБ) → увеличь амплитуду до 0.03–0.05")
    else:
        print(f"✗  WM не виден энергетически (SNR={snr:.1f} дБ) → увеличь амплитуду до 0.05")

    if matches >= 6:
        print(f"✓  FSK декодер работает ({matches}/8 бит верно)")
    else:
        print(f"✗  FSK декодер даёт ошибки ({matches}/8 бит верно)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Использование: python debug_watermark.py "файл.wav"')
    else:
        diagnose(sys.argv[1])