"""
GUI приложение для системы аудио водяных знаков
Позволяет встраивать и обнаруживать водяные знаки с визуализацией
"""

import sys
import os
try:
    import PyQt5
    qt_platforms = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms")
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", qt_platforms)
except Exception:
    pass
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
import json
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QTextEdit,
    QProgressBar, QGroupBox, QGridLayout, QComboBox, QSpinBox,
    QDoubleSpinBox, QMessageBox, QSplitter, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon

import io

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from watermark_system import AudioWatermark, WatermarkConfig


def _load_audio(path: str, **kwargs):
    """BytesIO обёртка — фикс кириллических путей на Windows (баг libsndfile)"""
    raw = Path(path).read_bytes()
    return librosa.load(io.BytesIO(raw), **kwargs)


class MplCanvas(FigureCanvas):
    """Холст для отображения matplotlib графиков"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

    def clear_plot(self, title=""):
        """Полностью очищает figure и пересоздаёт axes"""
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        if title:
            self.axes.set_title(title, color='#888888')
        self.axes.text(
            0.5, 0.5, 'Нет данных',
            transform=self.axes.transAxes,
            ha='center', va='center',
            color='#aaaaaa', fontsize=13
        )
        self.draw()


class WatermarkThread(QThread):
    """Поток для фоновой обработки аудио (не блокирует GUI)"""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, operation, input_file, output_file=None, user_id=None,
                 config=None, method='frequency'):
        super().__init__()
        self.operation = operation
        self.input_file = input_file
        self.output_file = output_file
        self.user_id = user_id
        self.config = config or WatermarkConfig()
        self.method = method
        self.watermark = AudioWatermark(self.config)

    def run(self):
        try:
            if self.operation == 'embed':
                self.embed_watermark()
            elif self.operation == 'detect':
                self.detect_watermark()
        except Exception as e:
            self.error.emit(str(e))

    def embed_watermark(self):
        self.status.emit("Загрузка аудиофайла...")
        self.progress.emit(10)
        self.status.emit("Генерация уникального паттерна...")
        self.progress.emit(30)
        self.status.emit(f"Встраивание водяного знака методом '{self.method}'...")
        self.progress.emit(50)

        result = self.watermark.embed_watermark(
            self.input_file, self.output_file, self.user_id, self.method
        )

        self.progress.emit(90)
        self.status.emit("Сохранение файла...")
        self.progress.emit(100)
        self.status.emit("Готово!")

        result['output_file'] = self.output_file
        self.finished.emit(result)

    def detect_watermark(self):
        self.status.emit("Загрузка аудиофайла...")
        self.progress.emit(20)
        self.status.emit("Анализ частотного спектра...")
        self.progress.emit(50)
        self.status.emit("Поиск водяного знака...")
        self.progress.emit(80)

        result = self.watermark.detect_watermark(
            self.input_file,
            self.user_id,
            threshold=0.6
        )

        self.progress.emit(100)
        self.status.emit("Анализ завершен!")

        result['input_file'] = self.input_file
        self.finished.emit(result)


class AudioWatermarkGUI(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.embed_input_file = None   # файл для встраивания
        self.embed_output_file = None  # результат встраивания
        self.detect_input_file = None  # файл для проверки
        self.watermark_config = WatermarkConfig()  # 18000-21000 Hz по умолчанию
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Audio Watermark System - Система водяных знаков')
        self.setGeometry(100, 100, 1400, 900)

        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
            QTabWidget::pane { border: 1px solid #cccccc; border-radius: 4px; }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 10px 20px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background-color: #4CAF50; color: white; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        header = QLabel('🎵 Audio Watermark System')
        header.setFont(QFont('Arial', 24, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #2196F3; padding: 20px;")
        main_layout.addWidget(header)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.create_embed_tab()
        self.create_detect_tab()
        self.create_settings_tab()
        self.create_help_tab()

        self.statusBar().showMessage('Готов к работе')

    # ─────────────────────────────────────────────────────────────────────────
    # Вкладка «Встроить»
    # ─────────────────────────────────────────────────────────────────────────
    def create_embed_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 1. Выбор файла
        top_panel = QGroupBox("1. Выбор файла и настройки")
        top_layout = QGridLayout()

        self.embed_file_label = QLabel("Файл не выбран")
        self.embed_file_label.setStyleSheet(
            "padding: 5px; background-color: white; border: 1px solid #ccc;"
        )
        select_file_btn = QPushButton("📂 Выбрать аудиофайл")
        select_file_btn.clicked.connect(self.select_file_for_embedding)

        top_layout.addWidget(QLabel("Входной файл:"), 0, 0)
        top_layout.addWidget(self.embed_file_label, 0, 1)
        top_layout.addWidget(select_file_btn, 0, 2)

        self.user_id_input = QComboBox()
        self.user_id_input.setEditable(True)
        self.user_id_input.addItems(['user_001', 'user_002', 'user_003', 'test_user'])
        top_layout.addWidget(QLabel("User ID (метка владельца):"), 1, 0)
        top_layout.addWidget(self.user_id_input, 1, 1, 1, 2)

        self.embed_method_combo = QComboBox()
        self.embed_method_combo.addItems(['frequency', 'spread_spectrum'])
        top_layout.addWidget(QLabel("Метод:"), 2, 0)
        top_layout.addWidget(self.embed_method_combo, 2, 1, 1, 2)

        top_panel.setLayout(top_layout)
        layout.addWidget(top_panel)

        # Кнопка
        embed_btn = QPushButton("🔒 Встроить водяной знак")
        embed_btn.setStyleSheet("background-color: #2196F3; font-size: 16px; padding: 12px;")
        embed_btn.clicked.connect(self.embed_watermark)
        layout.addWidget(embed_btn)

        # 2. Прогресс
        progress_group = QGroupBox("2. Прогресс обработки")
        progress_layout = QVBoxLayout()
        self.embed_progress = QProgressBar()
        self.embed_progress.setStyleSheet("""
            QProgressBar { border: 2px solid grey; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background-color: #4CAF50; }
        """)
        progress_layout.addWidget(self.embed_progress)
        self.embed_status = QLabel("Ожидание...")
        self.embed_status.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.embed_status)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # 3. Визуализация
        viz_group = QGroupBox("3. Визуализация (До и После)")
        viz_layout = QHBoxLayout()
        self.embed_canvas_before = MplCanvas(self, width=6, height=4)
        self.embed_canvas_before.clear_plot("Оригинальный спектр (До)")
        viz_layout.addWidget(self.embed_canvas_before)
        self.embed_canvas_after = MplCanvas(self, width=6, height=4)
        self.embed_canvas_after.clear_plot("Спектр с водяным знаком (После)")
        viz_layout.addWidget(self.embed_canvas_after)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # 4. Информация
        info_group = QGroupBox("4. Информация о результате")
        self.embed_info_text = QTextEdit()
        self.embed_info_text.setReadOnly(True)
        self.embed_info_text.setMaximumHeight(150)
        info_layout = QVBoxLayout()
        info_layout.addWidget(self.embed_info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Сохранить
        save_btn = QPushButton("💾 Сохранить файл с водяным знаком")
        save_btn.setStyleSheet("background-color: #FF9800;")
        save_btn.clicked.connect(self.save_watermarked_file)
        layout.addWidget(save_btn)

        self.tabs.addTab(tab, "🔒 Встроить водяной знак")

    # ─────────────────────────────────────────────────────────────────────────
    # Вкладка «Обнаружить»
    # ─────────────────────────────────────────────────────────────────────────
    def create_detect_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 1. Выбор файла
        top_panel = QGroupBox("1. Выбор файла для проверки")
        top_layout = QGridLayout()

        self.detect_file_label = QLabel("Файл не выбран")
        self.detect_file_label.setStyleSheet(
            "padding: 5px; background-color: white; border: 1px solid #ccc;"
        )
        select_file_btn = QPushButton("📂 Выбрать файл для проверки")
        select_file_btn.clicked.connect(self.select_file_for_detection)

        top_layout.addWidget(QLabel("Проверяемый файл:"), 0, 0)
        top_layout.addWidget(self.detect_file_label, 0, 1)
        top_layout.addWidget(select_file_btn, 0, 2)

        # User ID — необязательное поле
        self.detect_user_id = QComboBox()
        self.detect_user_id.setEditable(True)
        self.detect_user_id.addItems(['', 'user_001', 'user_002', 'user_003', 'test_user'])
        top_layout.addWidget(QLabel("User ID (необязательно):"), 1, 0)
        top_layout.addWidget(self.detect_user_id, 1, 1, 1, 2)

        hint = QLabel(
            "💡 Оставьте User ID пустым — система определит наличие знака автоматически.\n"
            "   Укажите User ID — проверит принадлежность конкретному владельцу."
        )
        hint.setStyleSheet("color: #555; font-size: 11px; padding: 4px;")
        top_layout.addWidget(hint, 2, 0, 1, 3)

        top_panel.setLayout(top_layout)
        layout.addWidget(top_panel)

        # Кнопка
        detect_btn = QPushButton("🔍 Проверить наличие водяного знака")
        detect_btn.setStyleSheet(
            "background-color: #9C27B0; font-size: 16px; padding: 12px;"
        )
        detect_btn.clicked.connect(self.detect_watermark)
        layout.addWidget(detect_btn)

        # 2. Прогресс
        progress_group = QGroupBox("2. Прогресс анализа")
        progress_layout = QVBoxLayout()
        self.detect_progress = QProgressBar()
        self.detect_progress.setStyleSheet("""
            QProgressBar { border: 2px solid grey; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background-color: #9C27B0; }
        """)
        progress_layout.addWidget(self.detect_progress)
        self.detect_status = QLabel("Ожидание...")
        self.detect_status.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.detect_status)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # 3. Результат
        result_group = QGroupBox("3. Результат проверки")
        result_layout = QVBoxLayout()
        self.detect_result_label = QLabel("Водяной знак: НЕ ПРОВЕРЕНО")
        self.detect_result_label.setFont(QFont('Arial', 18, QFont.Bold))
        self.detect_result_label.setAlignment(Qt.AlignCenter)
        self.detect_result_label.setStyleSheet(
            "padding: 20px; background-color: #e0e0e0; border-radius: 8px;"
        )
        result_layout.addWidget(self.detect_result_label)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        # 4. Спектрограмма
        viz_group = QGroupBox("4. Спектральный анализ")
        viz_layout = QVBoxLayout()
        self.detect_canvas = MplCanvas(self, width=10, height=5)
        self.detect_canvas.clear_plot("Ожидание анализа...")
        viz_layout.addWidget(self.detect_canvas)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # 5. Детальная информация
        info_group = QGroupBox("5. Детальная информация")
        self.detect_info_text = QTextEdit()
        self.detect_info_text.setReadOnly(True)
        self.detect_info_text.setMaximumHeight(130)
        info_layout = QVBoxLayout()
        info_layout.addWidget(self.detect_info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        self.tabs.addTab(tab, "🔍 Обнаружить водяной знак")

    # ─────────────────────────────────────────────────────────────────────────
    # Вкладка «Настройки»
    # ─────────────────────────────────────────────────────────────────────────
    def create_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        settings_group = QGroupBox("Настройки водяного знака")
        settings_layout = QGridLayout()

        settings_layout.addWidget(QLabel("Минимальная частота (Гц):"), 0, 0)
        self.freq_min_spin = QSpinBox()
        self.freq_min_spin.setRange(1000, 24000)
        self.freq_min_spin.setValue(20000)
        self.freq_min_spin.setSingleStep(500)
        settings_layout.addWidget(self.freq_min_spin, 0, 1)

        settings_layout.addWidget(QLabel("Максимальная частота (Гц):"), 1, 0)
        self.freq_max_spin = QSpinBox()
        self.freq_max_spin.setRange(2000, 24000)
        self.freq_max_spin.setValue(21000)        
        self.freq_max_spin.setSingleStep(500)
        settings_layout.addWidget(self.freq_max_spin, 1, 1)

        settings_layout.addWidget(QLabel("Длительность символа (мс):"), 2, 0)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(50, 500)
        self.duration_spin.setValue(100)
        self.duration_spin.setSingleStep(10)
        settings_layout.addWidget(self.duration_spin, 2, 1)

        settings_layout.addWidget(QLabel("Амплитуда (0.001-0.1):"), 3, 0)
        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0.001, 0.1)
        self.amplitude_spin.setValue(0.01)
        self.amplitude_spin.setSingleStep(0.001)
        self.amplitude_spin.setDecimals(4)
        settings_layout.addWidget(self.amplitude_spin, 3, 1)

        settings_layout.addWidget(QLabel("Длина паттерна (бит):"), 4, 0)
        self.pattern_length_spin = QSpinBox()
        self.pattern_length_spin.setRange(16, 128)
        self.pattern_length_spin.setValue(32)
        self.pattern_length_spin.setSingleStep(8)
        settings_layout.addWidget(self.pattern_length_spin, 4, 1)

        apply_btn = QPushButton("✓ Применить настройки")
        apply_btn.clicked.connect(self.apply_settings)
        settings_layout.addWidget(apply_btn, 5, 0, 1, 2)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        info_group = QGroupBox("Текущая конфигурация")
        self.settings_info = QTextEdit()
        self.settings_info.setReadOnly(True)
        self.update_settings_info()
        info_layout = QVBoxLayout()
        info_layout.addWidget(self.settings_info)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        self.tabs.addTab(tab, "⚙️ Настройки")

    # ─────────────────────────────────────────────────────────────────────────
    # Вкладка «Справка»
    # ─────────────────────────────────────────────────────────────────────────
    def create_help_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h1 style="color: #2196F3;">📖 Руководство пользователя</h1>

        <h2>🔒 Встраивание водяного знака</h2>
        <ol>
            <li><b>Выберите аудиофайл</b> — поддерживаются форматы: MP3, WAV, FLAC, OGG, M4A</li>
            <li><b>Введите User ID</b> — произвольная метка владельца (например, «artist_01»).
                Паттерн генерируется из этой строки через SHA-256 и встраивается в файл.</li>
            <li><b>Выберите метод</b>:
                <ul>
                    <li><i>frequency</i> — FSK-модуляция в диапазоне 22–25 кГц</li>
                    <li><i>spread_spectrum</i> — DSSS, более устойчив к модификациям</li>
                </ul>
            </li>
            <li><b>Нажмите «Встроить водяной знак»</b></li>
            <li><b>Сохраните результат</b> кнопкой «Сохранить файл»</li>
        </ol>

        <h2>🔍 Обнаружение водяного знака</h2>
        <ol>
            <li><b>Выберите файл для проверки</b></li>
            <li><b>User ID (необязательно):</b>
                <ul>
                    <li><i>Пустое поле</i> — слепое обнаружение: анализ энергии в диапазоне
                        водяного знака относительно соседних полос (SNR-метод)</li>
                    <li><i>Конкретный ID</i> — проверка принадлежности: корреляция
                        детектированного FSK-паттерна с эталоном данного владельца</li>
                </ul>
            </li>
            <li><b>Нажмите «Проверить наличие»</b></li>
            <li><b>Результат:</b>
                <ul>
                    <li>🟢 Зелёный = водяной знак ОБНАРУЖЕН</li>
                    <li>🟡 Жёлтый = возможно присутствует</li>
                    <li>🔴 Красный = водяной знак НЕ обнаружен</li>
                </ul>
            </li>
        </ol>

        <h2>⚙️ Настройки</h2>
        <p><b>Частотный диапазон:</b> 22000–25000 Гц — выше порога слышимости человека</p>
        <p><b>Длительность символа:</b> время на один бит FSK-паттерна</p>
        <p><b>Амплитуда:</b> уровень сигнала водяного знака (0.01 = 1 % от основного)</p>
        <p><b>Длина паттерна:</b> количество бит уникальной подписи</p>

        <h2>❓ Частые вопросы</h2>
        <p><b>Q: Слышен ли водяной знак?</b><br>
        A: Нет. Рабочий диапазон 22–25 кГц недоступен человеческому уху.</p>
        <p><b>Q: Для чего нужен User ID при встраивании?</b><br>
        A: Это метка владельца. Из неё через хеш генерируется уникальный битовый паттерн,
           который встраивается в файл. При детекции с тем же ID система проверит,
           совпадает ли паттерн в файле с эталоном.</p>
        <p><b>Q: Можно проверить файл без знания User ID?</b><br>
        A: Да — оставьте поле пустым. Система определит наличие знака по энергетическому
           анализу спектра, не зная, чей именно паттерн встроен.</p>
        <p><b>Q: Сохранится ли знак после конвертации?</b><br>
        A: В WAV/FLAC — да. В MP3 с битрейтом ≥192 kbps — обычно сохраняется.</p>

        <h2>📞 Версия</h2>
        <p>Дипломная работа — версия 1.1.0 (2024)</p>
        """)
        layout.addWidget(help_text)
        self.tabs.addTab(tab, "📖 Справка")

    # ─────────────────────────────────────────────────────────────────────────
    # Выбор файлов
    # ─────────────────────────────────────────────────────────────────────────
    def select_file_for_embedding(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите аудиофайл", "",
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a);;All Files (*)"
        )
        if file_path:
            self.embed_input_file = file_path
            self.embed_file_label.setText(os.path.basename(file_path))

            # Сбрасываем оба графика встраивания
            self.embed_canvas_before.clear_plot("Оригинальный спектр (До)")
            self.embed_canvas_after.clear_plot("Спектр с водяным знаком (После)")
            self.embed_info_text.clear()
            self.embed_progress.setValue(0)
            self.embed_status.setText("Ожидание...")

            # Авто-подстройка freq_range под формат и sample rate файла
            try:
                new_min, new_max, msg = self._auto_freq_range(file_path)
                self.watermark_config = WatermarkConfig(
                    freq_range=(new_min, new_max),
                    duration_ms=self.watermark_config.duration_ms,
                    amplitude=self.watermark_config.amplitude,
                    pattern_length=self.watermark_config.pattern_length,
                )
                self.freq_min_spin.setValue(new_min)
                self.freq_max_spin.setValue(new_max)
                self.update_settings_info()
                self.statusBar().showMessage(msg)
            except Exception:
                self.statusBar().showMessage(f'Выбран файл: {file_path}')

            self.visualize_original_audio(file_path)

    def _auto_freq_range(self, file_path: str):
        """
        Автоматически выбирает freq_range в зависимости от формата и sample rate.

        Логика:
          MP3/OGG/M4A (lossy) — кодек срезает высокие частоты:
            128 kbps → ~16 кГц, 192 kbps → ~18 кГц, 320 kbps → ~20 кГц
            Безопасный диапазон: 14000–16000 Hz (выживает при любом битрейте)
          WAV/FLAC/AIFF (lossless) — ограничение только Nyquist:
            sr=44100 → max 22050, sr=48000 → max 24000
            Используем 20000–21000 Hz (выше порога слышимости)
        """
        ext = Path(file_path).suffix.lower()
        raw = Path(file_path).read_bytes()
        _, sr_file = librosa.load(io.BytesIO(raw), sr=None, mono=True, duration=1.0)
        nyquist = sr_file // 2

        LOSSY = {'.mp3', '.ogg', '.m4a', '.aac', '.wma', '.opus'}
        LOSSLESS = {'.wav', '.flac', '.aiff', '.aif', '.pcm', '.w64'}

        if ext in LOSSY:
            # Безопасный диапазон для lossy — 14–16 кГц
            new_min, new_max = 16000, 17000
            fmt_note = f"lossy ({ext.upper()}) — диапазон снижен для надёжности"
        else:
            # Lossless или неизвестный — по Nyquist
            if nyquist >= 21000:
                new_min, new_max = 20000, 21000
            elif nyquist >= 18000:
                new_min, new_max = 16000, min(18000, nyquist - 200)
            else:
                new_min, new_max = max(nyquist - 3000, 1000), nyquist - 200
            fmt_note = f"lossless ({ext.upper()}, {sr_file} Hz)"

        msg = (f"✓ {os.path.basename(file_path)} — {fmt_note} → "
               f"диапазон WM: {new_min}–{new_max} Hz")
        return new_min, new_max, msg

    def select_file_for_detection(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл для проверки", "",
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a);;All Files (*)"
        )
        if file_path:
            self.detect_input_file = file_path
            self.detect_file_label.setText(os.path.basename(file_path))

            # Авто-подстройка диапазона под формат проверяемого файла
            try:
                new_min, new_max, msg = self._auto_freq_range(file_path)
                self.watermark_config = WatermarkConfig(
                    freq_range=(new_min, new_max),
                    duration_ms=self.watermark_config.duration_ms,
                    amplitude=self.watermark_config.amplitude,
                    pattern_length=self.watermark_config.pattern_length,
                )
                self.freq_min_spin.setValue(new_min)
                self.freq_max_spin.setValue(new_max)
                self.update_settings_info()
                self.statusBar().showMessage(msg)
            except Exception:
                self.statusBar().showMessage(f'Выбран файл: {file_path}')

            # Сбрасываем всё на вкладке детекции
            self._reset_detect_tab()

    def _reset_detect_tab(self):
        """Полный сброс вкладки обнаружения"""
        self.detect_canvas.clear_plot("Ожидание анализа...")
        self.detect_info_text.clear()
        self.detect_progress.setValue(0)
        self.detect_status.setText("Ожидание...")
        self.detect_result_label.setText("Водяной знак: НЕ ПРОВЕРЕНО")
        self.detect_result_label.setStyleSheet(
            "padding: 20px; background-color: #e0e0e0; border-radius: 8px; font-size: 18px;"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Визуализация оригинала
    # ─────────────────────────────────────────────────────────────────────────
    def visualize_original_audio(self, file_path):
        try:
            audio, sr = _load_audio(file_path, sr=None, duration=30)
            self.embed_canvas_before.fig.clear()
            ax = self.embed_canvas_before.fig.add_subplot(111)
            self.embed_canvas_before.axes = ax

            D = librosa.stft(audio)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            img = librosa.display.specshow(
                S_db, sr=sr, x_axis='time', y_axis='hz',
                ax=ax, cmap='viridis'
            )
            ax.set_title('Оригинальный спектр (До)')
            ax.set_ylim([0, min(sr / 2, 26000)])

            ax.axhline(y=self.watermark_config.freq_range[0],
                       color='r', linestyle='--', alpha=0.7, label='Диапазон водяного знака')
            ax.axhline(y=self.watermark_config.freq_range[1],
                       color='r', linestyle='--', alpha=0.7)
            ax.legend()

            self.embed_canvas_before.fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self.embed_canvas_before.draw()

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка визуализации: {str(e)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Встраивание
    # ─────────────────────────────────────────────────────────────────────────
    def embed_watermark(self):
        if not self.embed_input_file:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите файл!")
            return

        user_id = self.user_id_input.currentText().strip()
        if not user_id:
            QMessageBox.warning(self, "Ошибка", "Введите User ID!")
            return

        # Сбрасываем график «после» и статусы перед новым запуском
        self.embed_canvas_after.clear_plot("Спектр с водяным знаком (После)")
        self.embed_info_text.clear()
        self.embed_progress.setValue(0)
        self.embed_status.setText("Запуск...")

        input_path = Path(self.embed_input_file)
        output_dir = Path("watermarked")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.embed_output_file = str(
            output_dir / f"watermarked_{timestamp}_{input_path.name}"
        )

        method = self.embed_method_combo.currentText()

        self.embed_thread = WatermarkThread(
            'embed', self.embed_input_file, self.embed_output_file,
            user_id, self.watermark_config, method
        )
        self.embed_thread.progress.connect(self.embed_progress.setValue)
        self.embed_thread.status.connect(self.embed_status.setText)
        self.embed_thread.finished.connect(self.on_embed_finished)
        self.embed_thread.error.connect(self.on_error)
        self.embed_thread.start()
        self.statusBar().showMessage('Встраивание водяного знака...')

    def on_embed_finished(self, result):
        self.statusBar().showMessage('Водяной знак успешно встроен!')

        info_text = f"""
<b>✓ Водяной знак успешно встроен!</b><br><br>
<b>User ID:</b> {result['user_id']}<br>
<b>Метод:</b> {result['method']}<br>
<b>Sample Rate:</b> {result['sample_rate']} Hz<br>
<b>Длительность:</b> {result['duration']:.2f} сек<br>
<b>Паттерн (первые 16 бит):</b> {''.join(map(str, result['pattern'][:16]))}...<br>
<b>Выходной файл:</b> {os.path.basename(result['output_file'])}
        """
        self.embed_info_text.setHtml(info_text)
        self.visualize_watermarked_audio(result['output_file'])

        QMessageBox.information(
            self, "Успех",
            f"Водяной знак встроен!\n\nФайл сохранен:\n{result['output_file']}"
        )

    def visualize_watermarked_audio(self, file_path):
        try:
            audio, sr = _load_audio(file_path, sr=None, duration=30)

            self.embed_canvas_after.fig.clear()
            ax = self.embed_canvas_after.fig.add_subplot(111)
            self.embed_canvas_after.axes = ax

            D = librosa.stft(audio)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            img = librosa.display.specshow(
                S_db, sr=sr, x_axis='time', y_axis='hz',
                ax=ax, cmap='viridis'
            )
            ax.set_title('Спектр с водяным знаком (После)', color='green')
            ax.set_ylim([0, min(sr / 2, 26000)])

            ax.axhline(y=self.watermark_config.freq_range[0],
                       color='lime', linestyle='--', linewidth=2, alpha=0.8,
                       label='Водяной знак здесь')
            ax.axhline(y=self.watermark_config.freq_range[1],
                       color='lime', linestyle='--', linewidth=2, alpha=0.8)
            ax.axhspan(self.watermark_config.freq_range[0],
                       self.watermark_config.freq_range[1],
                       alpha=0.2, color='green')
            ax.legend()

            self.embed_canvas_after.fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self.embed_canvas_after.draw()

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка визуализации: {str(e)}")

    def save_watermarked_file(self):
        if not self.embed_output_file or not os.path.exists(self.embed_output_file):
            QMessageBox.warning(self, "Ошибка", "Нет файла для сохранения!")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить файл",
            os.path.basename(self.embed_output_file),
            "Audio Files (*.wav *.mp3 *.flac)"
        )
        if save_path:
            import shutil
            shutil.copy(self.embed_output_file, save_path)
            QMessageBox.information(self, "Успех", f"Файл сохранен:\n{save_path}")
            self.statusBar().showMessage(f'Файл сохранен: {save_path}')

    # ─────────────────────────────────────────────────────────────────────────
    # Обнаружение
    # ─────────────────────────────────────────────────────────────────────────
    def detect_watermark(self):
        if not self.detect_input_file:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите файл!")
            return

        # Всегда сбрасываем перед каждым новым запуском
        self._reset_detect_tab()
        self.detect_status.setText("Запуск...")

        user_id = self.detect_user_id.currentText().strip()
        if not user_id:
            user_id = None  # слепое обнаружение

        self.detect_thread = WatermarkThread(
            'detect', self.detect_input_file,
            user_id=user_id,
            config=self.watermark_config
        )
        self.detect_thread.progress.connect(self.detect_progress.setValue)
        self.detect_thread.status.connect(self.detect_status.setText)
        self.detect_thread.finished.connect(self.on_detect_finished)
        self.detect_thread.error.connect(self.on_error)
        self.detect_thread.start()
        self.statusBar().showMessage('Поиск водяного знака...')

    def on_detect_finished(self, result):
        detected = result.get('detected', False)
        correlation = result.get('correlation', 0.0)
        confidence = result.get('confidence', 0.0)
        snr_db = result.get('snr_db', 0.0)

        # Определяем визуальный статус
        if 'error' in result:
            status_text = f"⚠ ОШИБКА АНАЛИЗА"
            color = "#FF9800"
        elif detected and confidence > 0.5:
            status_text = "✓ ВОДЯНОЙ ЗНАК ОБНАРУЖЕН"
            color = "#4CAF50"
        elif detected:
            status_text = "⚠ ВОЗМОЖНО ПРИСУТСТВУЕТ"
            color = "#FF9800"
        else:
            status_text = "✗ НЕ ОБНАРУЖЕН"
            color = "#f44336"

        self.detect_result_label.setText(status_text)
        self.detect_result_label.setStyleSheet(
            f"padding: 20px; background-color: {color}; color: white; "
            f"border-radius: 8px; font-size: 20px; font-weight: bold;"
        )

        # Детальная информация
        info_html = "<h3>Результаты анализа:</h3>"

        if 'error' in result:
            info_html += f"<p style='color:red'><b>Ошибка:</b> {result['error']}</p>"
        else:
            method_label = "проверка по User ID" if result.get('user_id') else "слепое обнаружение (SNR)"
            info_html += f"<p><b>Метод:</b> {method_label}</p>"

            if result.get('user_id'):
                info_html += f"<p><b>User ID:</b> {result['user_id']}</p>"
                info_html += f"<p><b>Корреляция паттерна:</b> {correlation:.3f} " \
                              f"(порог: {result.get('threshold', 0.6):.2f})</p>"

            info_html += f"<p><b>SNR в диапазоне {self.watermark_config.freq_range[0]}–" \
                         f"{self.watermark_config.freq_range[1]} Гц:</b> {snr_db:.2f} дБ " \
                         f"<i>(>3 дБ = знак есть)</i></p>"
            info_html += f"<p><b>Достоверность:</b> {confidence:.1%}</p>"

            if 'pattern' in result:
                pattern_str = ''.join(map(str, result['pattern'][:32]))
                info_html += f"<p><b>Извлечённый паттерн (32 бит):</b><br>" \
                              f"<code>{pattern_str}</code></p>"

        self.detect_info_text.setHtml(info_html)

        # Визуализация
        if 'error' not in result:
            self.visualize_detection_result(result)

        self.statusBar().showMessage(f'Анализ завершен: {status_text}')

    def visualize_detection_result(self, result):
        try:
            audio, sr = _load_audio(result['input_file'], sr=None, duration=30)

            # Полностью пересоздаём figure — никакого наслоения
            self.detect_canvas.fig.clear()
            ax = self.detect_canvas.fig.add_subplot(111)
            self.detect_canvas.axes = ax

            D = librosa.stft(audio)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            img = librosa.display.specshow(
                S_db, sr=sr, x_axis='time', y_axis='hz',
                ax=ax, cmap='viridis'
            )
            ax.set_ylim([0, min(sr / 2, 26000)])

            detected = result.get('detected', False)
            line_color = 'lime' if detected else 'red'
            label = 'Водяной знак ОБНАРУЖЕН' if detected else 'Водяной знак НЕ ОБНАРУЖЕН'

            ax.axhline(y=self.watermark_config.freq_range[0],
                       color=line_color, linestyle='--', linewidth=2.5, alpha=0.9,
                       label=label)
            ax.axhline(y=self.watermark_config.freq_range[1],
                       color=line_color, linestyle='--', linewidth=2.5, alpha=0.9)

            ax.axhspan(
                self.watermark_config.freq_range[0],
                self.watermark_config.freq_range[1],
                alpha=0.3 if detected else 0.1,
                color=line_color
            )

            if detected and 'pattern' in result:
                mid_freq = (self.watermark_config.freq_range[0] +
                            self.watermark_config.freq_range[1]) / 2
                ax.text(
                    0.5, mid_freq, '← ВОДЯНОЙ ЗНАК ЗДЕСЬ',
                    fontsize=12, color='white',
                    bbox=dict(boxstyle='round', facecolor=line_color, alpha=0.8),
                    verticalalignment='center'
                )

            snr_db = result.get('snr_db', 0.0)
            ax.set_title(
                f'{label}  |  SNR: {snr_db:.1f} дБ',
                fontsize=13, color=line_color, fontweight='bold'
            )
            ax.legend(loc='upper right')

            self.detect_canvas.fig.colorbar(img, ax=ax, format='%+2.0f dB')
            self.detect_canvas.draw()

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка визуализации: {str(e)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Настройки
    # ─────────────────────────────────────────────────────────────────────────
    def apply_settings(self):
        self.watermark_config = WatermarkConfig(
            freq_range=(self.freq_min_spin.value(), self.freq_max_spin.value()),
            duration_ms=self.duration_spin.value(),
            amplitude=self.amplitude_spin.value(),
            pattern_length=self.pattern_length_spin.value()
        )
        self.update_settings_info()
        QMessageBox.information(self, "Успех", "Настройки применены!")
        self.statusBar().showMessage('Настройки обновлены')

    def update_settings_info(self):
        info = f"""
<h3>Текущая конфигурация:</h3>
<table style="width:100%">
<tr><td><b>Частотный диапазон:</b></td>
    <td>{self.watermark_config.freq_range[0]} – {self.watermark_config.freq_range[1]} Гц</td></tr>
<tr><td><b>Длительность символа:</b></td>
    <td>{self.watermark_config.duration_ms} мс</td></tr>
<tr><td><b>Амплитуда:</b></td>
    <td>{self.watermark_config.amplitude} ({self.watermark_config.amplitude * 100:.2f} %)</td></tr>
<tr><td><b>Длина паттерна:</b></td>
    <td>{self.watermark_config.pattern_length} бит</td></tr>
</table>
<p style="color: #666; font-size: 12px;">
<i>Частоты 22–25 кГц находятся выше порога слышимости человека (20 Гц – 20 кГц)</i>
</p>
        """
        self.settings_info.setHtml(info)

    # ─────────────────────────────────────────────────────────────────────────
    # Ошибки
    # ─────────────────────────────────────────────────────────────────────────
    def on_error(self, error_msg):
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_msg}")
        self.statusBar().showMessage(f'Ошибка: {error_msg}')


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = AudioWatermarkGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()