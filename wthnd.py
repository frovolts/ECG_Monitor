import sys
import time
import serial
import threading
import numpy as np
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg

# ==========================================
# CONFIGURATION
# ==========================================
SERIAL_PORT = "COM4"  # CHANGE THIS IF NEEDED
BAUD_RATE = 115200
WINDOW_SIZE = 1000    # How many points to show on screen

# ==========================================
# 1. SIGNAL PROCESSING (The Math)
# ==========================================
class HeartSignalProcessor:
    def __init__(self):
        self.last_peak_time = 0
        self.peaks_buffer = deque(maxlen=20) 
        self.threshold = 2000 

    def detect(self, window_data):
        if len(window_data) < 50: return 0, False, 0.0

        signal = np.array(window_data)[-100:] 
        diff_sig = np.diff(signal)
        squared_sig = diff_sig ** 2
        
        current_max = np.max(squared_sig)
        if current_max > self.threshold:
            self.threshold = 0.6 * current_max
        else:
            self.threshold = self.threshold * 0.995 
            
        recent_val = squared_sig[-1]
        current_time = time.time()
        is_beat = False
        
        if recent_val > self.threshold and (current_time - self.last_peak_time) > 0.25:
            self.last_peak_time = current_time
            is_beat = True
            if len(self.peaks_buffer) > 0:
                delta = current_time - self.peaks_buffer[-1]
                bpm = 60.0 / delta
                if 40 < bpm < 220: self.peaks_buffer.append(current_time)
            else:
                self.peaks_buffer.append(current_time)

        if len(self.peaks_buffer) < 2: return 0, is_beat, self.threshold
        avg_bpm = int(60.0 / np.mean(np.diff(list(self.peaks_buffer))))
        return avg_bpm, is_beat, self.threshold

# ==========================================
# 2. SERIAL THREAD
# ==========================================
data_buffer = deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE)
serial_connected = False
processor = HeartSignalProcessor()

# Shared variables for UI to read
current_bpm = 0
status_text = "DISCONNECTED"
beat_detected = False

def serial_worker():
    global serial_connected, current_bpm, status_text, beat_detected
    print(f"🔌 Connecting to {SERIAL_PORT}...")
    
    while True:
        try:
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
                serial_connected = True
                status_text = "USB LINKED"
                print("✅ Connected!")
                
                while True:
                    if ser.in_waiting:
                        try:
                            line = ser.readline().decode('utf-8', errors='ignore').strip()
                            # Parse {"t":..., "v": 2000} or just raw numbers
                            # Let's handle the JSON format we made earlier
                            if "v" in line: 
                                import json
                                val = json.loads(line)["v"]
                            elif line.isdigit():
                                val = int(line)
                            else:
                                continue

                            data_buffer.append(val)
                            
                            # Run Math
                            bpm, beat, _ = processor.detect(list(data_buffer))
                            current_bpm = bpm
                            if beat: beat_detected = True
                            
                        except:
                            pass
        except Exception as e:
            serial_connected = False
            status_text = "NO USB"
            time.sleep(1)

# Start Serial in background
t = threading.Thread(target=serial_worker, daemon=True)
t.start()

# ==========================================
# 3. THE GUI (PyQtGraph)
# ==========================================
class ECGDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Monitor [PYTHON NATIVE]")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #0b0b10; color: #e0e0e0;")

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Header Stats ---
        stats_layout = QHBoxLayout()
        
        # BPM Box
        self.bpm_label = QLabel("--")
        self.bpm_label.setFont(QFont("Consolas", 72, QFont.Bold))
        self.bpm_label.setStyleSheet("color: #00ff88;") # Green
        
        bpm_box = self.create_stat_box("HEART RATE", self.bpm_label, "BPM")
        stats_layout.addWidget(bpm_box)

        # Status Box
        self.status_label = QLabel("INIT...")
        self.status_label.setFont(QFont("Consolas", 36, QFont.Bold))
        self.status_label.setStyleSheet("color: #ffbb00;") # Yellow
        
        status_box = self.create_stat_box("SYSTEM STATUS", self.status_label, "")
        stats_layout.addWidget(status_box)
        
        main_layout.addLayout(stats_layout)

        # --- The Graph ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1a1a20')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        
        # Style the line
        self.ecg_curve = self.plot_widget.plot(pen=pg.mkPen(color='#00ff88', width=2))
        
        # Remove axis numbers for cleaner look
        self.plot_widget.getPlotItem().hideAxis('bottom')
        self.plot_widget.getPlotItem().hideAxis('left')
        
        main_layout.addWidget(self.plot_widget)

        # --- Timer Loop (60 FPS) ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(16) # 16ms = ~60 FPS

    def create_stat_box(self, title, value_widget, unit):
        frame = QFrame()
        frame.setStyleSheet("background-color: #1a1a20; border-radius: 10px; border: 1px solid #333;")
        layout = QVBoxLayout(frame)
        
        title_lbl = QLabel(title)
        title_lbl.setFont(QFont("Arial", 10))
        title_lbl.setStyleSheet("color: #888;")
        
        unit_lbl = QLabel(unit)
        unit_lbl.setFont(QFont("Arial", 12))
        unit_lbl.setStyleSheet("color: #555; margin-top: 10px;")
        
        layout.addWidget(title_lbl)
        layout.addWidget(value_widget)
        layout.addWidget(unit_lbl)
        return frame

    def update_ui(self):
        global beat_detected
        
        # Update Plot
        self.ecg_curve.setData(list(data_buffer))
        
        # Update Labels
        self.bpm_label.setText(str(current_bpm))
        self.status_label.setText(status_text)
        
        # Color Logic
        if status_text == "NO USB":
            self.status_label.setStyleSheet("color: #ff4444;") # Red
        else:
            self.status_label.setStyleSheet("color: #00aaff;") # Blue
            
        # Beat Animation (Flash BPM color)
        if beat_detected:
            self.bpm_label.setStyleSheet("color: #ff0055;") # Pink flash
            beat_detected = False
        elif not beat_detected:
             # Fade back to green (simple toggle for now)
             if current_bpm > 0:
                 self.bpm_label.setStyleSheet("color: #00ff88;") 

# ==========================================
# 4. RUN APP
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ECGDashboard()
    window.show()
    sys.exit(app.exec_())