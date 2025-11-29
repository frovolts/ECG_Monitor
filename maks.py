import sys
import time
import threading
import numpy as np
import joblib 
import math # Needed for demo wave
from collections import deque
import serial
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFrame, QGridLayout, QGraphicsDropShadowEffect)
from PyQt5.QtCore import QTimer, Qt, QRectF
from PyQt5.QtGui import QFont, QColor, QPainter, QPen
import pyqtgraph as pg

# ==========================================
# CONFIGURATION
# ==========================================
SERIAL_PORT = "COM4" 
BAUD_RATE = 115200
WINDOW_SIZE = 1000    

# ==========================================
# 1. VISUAL COMPONENT: MODERN CARD
# ==========================================
class ModernCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            ModernCard {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 20))
        self.setGraphicsEffect(shadow)

# ==========================================
# 2. VISUAL COMPONENT: GAUGE
# ==========================================
class CircularGauge(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.value = 0
        self.primary_color = QColor("#00B8D9")
        self.setMinimumSize(120, 150)

    def set_value(self, val, color_hex):
        self.value = max(0, min(100, val))
        self.primary_color = QColor(color_hex)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        side = min(w, h) - 30
        rect = QRectF((w - side) / 2, 10, side, side)

        painter.setPen(QPen(QColor("#e6e6e6"), 10, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, 270 * 16, 360 * 16) 

        painter.setPen(QPen(self.primary_color, 10, Qt.SolidLine, Qt.RoundCap))
        span = int(-360 * (self.value / 100) * 16)
        painter.drawArc(rect, 270 * 16, span)

        painter.setPen(QColor("#2c3e50")) 
        painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, f"{int(self.value)}%")

        painter.setPen(QColor("#7f8c8d"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(QRectF(0, h - 25, w, 25), Qt.AlignCenter, self.title)
        painter.end()

# ==========================================
# 3. BACKEND LOGIC
# ==========================================
class DiagnosticEngine:
    def __init__(self):
        self.model = None
        self.classes = {
            0: "NORMAL RHYTHM", 1: "ATRIAL FIB", 2: "PVC DETECTED",
            3: "STEMI WARNING", 4: "LONG QT",
            5: "BRADYCARDIA", 6: "TACHYCARDIA"
        }
        try:
            self.model = joblib.load("ecg_brain_advanced.pkl")
            print("🧠 AI Model Loaded.")
        except:
            print("⚠️ MODEL MISSING. Run train_ai.py")

    def analyze(self, peaks_buffer, avg_qrs_width):
        if len(peaks_buffer) < 5:
            return {"condition": "Scanning...", "confidence": 0, "pred_idx": -1}
        rr_list_ms = np.diff(list(peaks_buffer)) * 1000 
        avg_bpm = 60000 / np.mean(rr_list_ms)
        diff_rr = np.diff(rr_list_ms)
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        sdnn = np.std(rr_list_ms)
        qt_interval, st_elev = 400, 0.0 
        
        condition, confidence, pred_idx = "ANALYZING...", 0, -1
        if self.model:
            features = np.array([[avg_bpm, rmssd, sdnn, avg_qrs_width, qt_interval, st_elev]])
            pred_idx = self.model.predict(features)[0]
            condition = self.classes.get(pred_idx, "UNKNOWN")
            probs = self.model.predict_proba(features)[0]
            confidence = int(probs[int(pred_idx)] * 100)

        return {"condition": condition, "confidence": confidence, "pred_idx": pred_idx, "rmssd": rmssd}

class HeartSignalProcessor:
    def __init__(self):
        self.last_peak_time = 0
        self.peaks_buffer = deque(maxlen=30) 
        self.threshold = 2000 
        self.diagnostic = DiagnosticEngine()
        self.qrs_start_time = 0

    def detect(self, window_data):
        if len(window_data) < 50: return 0, False, {}
        signal = np.array(window_data)[-100:] 
        squared_sig = (np.diff(signal) ** 2)
        current_max = np.max(squared_sig)
        if current_max > self.threshold: self.threshold = 0.6 * current_max
        else: self.threshold = self.threshold * 0.995 
        recent_val = squared_sig[-1]
        current_time = time.time()
        is_beat = False
        qrs_width = 80 
        
        if recent_val > self.threshold:
            if self.qrs_start_time == 0: self.qrs_start_time = current_time
            if (current_time - self.last_peak_time) > 0.25:
                self.last_peak_time = current_time
                is_beat = True
                if self.qrs_start_time != 0:
                    qrs_width = (current_time - self.qrs_start_time) * 1000
                    qrs_width = max(40, min(200, qrs_width))
                    self.qrs_start_time = 0
                if len(self.peaks_buffer) > 0:
                    delta = current_time - self.peaks_buffer[-1]
                    bpm = 60.0 / delta
                    if 30 < bpm < 220: self.peaks_buffer.append(current_time)
                else: self.peaks_buffer.append(current_time)
        else: self.qrs_start_time = 0
        
        bpm = 0
        if len(self.peaks_buffer) >= 2: bpm = int(60.0 / np.mean(np.diff(list(self.peaks_buffer))))
        ml_stats = self.diagnostic.analyze(self.peaks_buffer, qrs_width)
        return bpm, is_beat, ml_stats

# ==========================================
# 4. HYBRID WORKER (USB + DEMO FALLBACK)
# ==========================================
data_buffer = deque([2048]*WINDOW_SIZE, maxlen=WINDOW_SIZE)
processor = HeartSignalProcessor()
ui_state = { "bpm": 0, "status": "Initializing...", "beat": False, "ml": {} }

def hybrid_worker():
    """Tries USB first. If fails, runs Demo Mode."""
    print(f"🔌 Attempting connection to {SERIAL_PORT}...")
    
    usb_connected = False
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        usb_connected = True
        print(f"✅ USB Connected! Listening to heart...")
        ui_state["status"] = "LIVE MONITORING"
        
        while True:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if "v" in line: 
                        import json
                        val = int(json.loads(line)["v"])
                    elif line.isdigit(): val = int(line)
                    else: continue
                    
                    data_buffer.append(val)
                    bpm, beat, ml_stats = processor.detect(list(data_buffer))
                    ui_state["bpm"] = bpm
                    if beat: ui_state["beat"] = True
                    ui_state["ml"] = ml_stats
                except: pass
    except:
        print("❌ USB Connection Failed. Starting DEMO MODE.")
        ui_state["status"] = "DEMO SIMULATION"

    # --- DEMO MODE LOOP (Only runs if USB failed) ---
    if not usb_connected:
        t_sim = 0
        while True:
            time.sleep(0.004) # 250Hz
            t_sim += 0.004
            
            # Generate fake heartbeat
            val = 2048 
            # P-wave
            if 0.1 < (t_sim % 0.8) < 0.2: val += 100 * math.sin(((t_sim % 0.8)-0.1)*31.4)
            # QRS
            if 0.23 < (t_sim % 0.8) < 0.27: val -= 300 
            if 0.27 < (t_sim % 0.8) < 0.30: val += 1500 
            if 0.30 < (t_sim % 0.8) < 0.33: val -= 400 
            # T-wave
            if 0.45 < (t_sim % 0.8) < 0.6: val += 200 * math.sin(((t_sim % 0.8)-0.45)*20)
            
            # Add noise
            val += np.random.normal(0, 10)
            
            data_buffer.append(int(val))
            bpm, beat, ml_stats = processor.detect(list(data_buffer))
            ui_state["bpm"] = bpm
            if beat: ui_state["beat"] = True
            ui_state["ml"] = ml_stats

t = threading.Thread(target=hybrid_worker, daemon=True)
t.start()

# ==========================================
# 5. UI IMPLEMENTATION
# ==========================================
class MedicalDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patient Vitals Analytics")
        self.setGeometry(100, 100, 1280, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #f3f4f6; }
            QLabel { font-family: 'Segoe UI', sans-serif; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Sidebar
        sidebar = QFrame()
        sidebar.setStyleSheet("background-color: #ffffff; border-radius: 10px; border: 1px solid #e0e0e0;")
        sidebar.setFixedWidth(80)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(10, 30, 10, 30)
        for icon in ["📊", "❤️", "📁", "⚙️"]:
            l = QLabel(icon)
            l.setFont(QFont("Segoe UI", 20))
            l.setAlignment(Qt.AlignCenter)
            sb_layout.addWidget(l)
        sb_layout.addStretch()
        main_layout.addWidget(sidebar)

        # Content
        content_layout = QVBoxLayout()
        
        # Header with Status
        header_layout = QHBoxLayout()
        header_lbl = QLabel("SR 520 Patient Monitoring")
        header_lbl.setStyleSheet("color: #374151; font-size: 24px; font-weight: bold;")
        self.status_lbl = QLabel("Initializing...")
        self.status_lbl.setStyleSheet("color: #6b7280; font-size: 14px; font-style: italic;")
        header_layout.addWidget(header_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.status_lbl)
        content_layout.addLayout(header_layout)

        # KPIs
        kpi_layout = QHBoxLayout()
        kpi_layout.setSpacing(15)
        self.kpi_bpm = self.create_kpi_card("Avg Heart Rate", "--", "BPM", "#00B8D9")
        kpi_layout.addWidget(self.kpi_bpm)
        self.kpi_status = self.create_kpi_card("AI Diagnosis", "Initializing", "", "#36B37E")
        kpi_layout.addWidget(self.kpi_status)
        self.kpi_rmssd = self.create_kpi_card("Stress Metric", "0", "ms", "#6554C0")
        kpi_layout.addWidget(self.kpi_rmssd)
        content_layout.addLayout(kpi_layout)

        # Main Split
        split_layout = QHBoxLayout()
        
        graph_card = ModernCard()
        gc_layout = QVBoxLayout(graph_card)
        g_title = QLabel("Real-time Lead I Waveform")
        g_title.setStyleSheet("color: #6b7280; font-weight: bold;")
        gc_layout.addWidget(g_title)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w') 
        self.plot_widget.getPlotItem().hideAxis('bottom')
        self.plot_widget.getPlotItem().hideAxis('left')
        self.ecg_curve = self.plot_widget.plot(pen=pg.mkPen(color='#0052CC', width=2))
        gc_layout.addWidget(self.plot_widget)
        split_layout.addWidget(graph_card, stretch=2)

        gauge_card = ModernCard()
        gauge_layout = QGridLayout(gauge_card)
        self.g_afib = CircularGauge("AFib Risk")
        self.g_stemi = CircularGauge("STEMI Risk")
        self.g_longqt = CircularGauge("Long QT")
        self.g_brady = CircularGauge("Bradycardia")
        gauge_layout.addWidget(self.g_afib, 0, 0)
        gauge_layout.addWidget(self.g_stemi, 0, 1)
        gauge_layout.addWidget(self.g_longqt, 1, 0)
        gauge_layout.addWidget(self.g_brady, 1, 1)
        split_layout.addWidget(gauge_card, stretch=1)

        content_layout.addLayout(split_layout, stretch=1)
        main_layout.addLayout(content_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_dashboard)
        self.timer.start(30)

    def create_kpi_card(self, title, value, unit, accent_color):
        card = ModernCard()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        t_lbl = QLabel(title)
        t_lbl.setStyleSheet("color: #6b7280; font-size: 14px;")
        layout.addWidget(t_lbl)
        v_layout = QHBoxLayout()
        v_lbl = QLabel(value)
        v_lbl.setObjectName("value_lbl") 
        v_lbl.setStyleSheet(f"color: {accent_color}; font-size: 32px; font-weight: bold;")
        u_lbl = QLabel(unit)
        u_lbl.setStyleSheet("color: #9ca3af; font-size: 14px; margin-top: 10px;")
        v_layout.addWidget(v_lbl)
        v_layout.addWidget(u_lbl)
        v_layout.addStretch()
        layout.addLayout(v_layout)
        return card

    def update_dashboard(self):
        self.ecg_curve.setData(list(data_buffer))
        self.status_lbl.setText(f"Status: {ui_state['status']}")
        
        bpm_val_lbl = self.kpi_bpm.findChild(QLabel, "value_lbl")
        bpm_val_lbl.setText(str(ui_state["bpm"]))
        
        ml = ui_state.get("ml", {})
        cond = ml.get("condition", "Scanning...")
        status_lbl = self.kpi_status.findChild(QLabel, "value_lbl")
        status_lbl.setText(cond.split(" ")[0]) 
        
        rmssd_val = int(ml.get("rmssd", 0))
        rmssd_lbl = self.kpi_rmssd.findChild(QLabel, "value_lbl")
        rmssd_lbl.setText(str(rmssd_val))

        if not ml: return
        conf = ml.get("confidence", 0)
        idx = ml.get("pred_idx", -1)

        def update_g(gauge, target_idx, warning=False):
            val = conf if idx == target_idx else 5
            color = "#FF5630" if warning else "#FFAB00" 
            gauge.set_value(val, color if val > 20 else "#dfe1e6")

        update_g(self.g_afib, 1)
        update_g(self.g_stemi, 3, warning=True)
        update_g(self.g_longqt, 4)
        update_g(self.g_brady, 5)

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    window = MedicalDashboard()
    window.show()
    sys.exit(app.exec_())