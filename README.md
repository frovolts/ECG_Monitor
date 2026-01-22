# 🫀ECG Monitoring System 

A comprehensive real-time ECG monitoring dashboard with AI-powered heart condition classification.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> [!TIP]
> **Don't just look at the code—see it in action!**  
> Check out the [**Live Interactive Demo**](https://ecgmonitor.streamlit.app/) to test the AI diagnostics with simulated data.

## 🌟 Features

- **Real-time ECG Visualization**: Live ECG waveform display with smooth animations
- **Heart Rate Monitoring**: Continuous BPM calculation and display
- **AI-Powered Diagnostics**: Machine learning model for detecting:
  - Normal Sinus Rhythm
  - Atrial Fibrillation (AFib)
  - Premature Ventricular Contractions (PVC)
  - STEMI Warning
  - Long QT Syndrome
  - Bradycardia
  - Tachycardia
- **Heart Rate Variability (HRV) Analysis**: RMSSD and SDNN metrics
- **Demo Mode**: Simulated ECG data when no hardware is connected
- **CSV Data Playback**: Use recorded ECG data files
- **ESP32 Support**: Connect to ESP32-based ECG hardware via serial port
- **Professional Medical UI**: Clean, dark-themed medical dashboard

> [!TIP]
> **Don't just look at the code—see it in action!**  
> Check out the [**Live Interactive Demo**](https://ecgmonitor.streamlit.app/) to test the AI diagnostics with simulated data

## 📸 Screenshots
<img width="1907" height="748" alt="image" src="https://github.com/user-attachments/assets/a36b266d-331c-4891-bc94-5ab8ddb256e0" />
<img width="1463" height="653" alt="image" src="https://github.com/user-attachments/assets/e98805dd-87c4-48f0-9888-69e2f112e165" />
<img width="1919" height="896" alt="image" src="https://github.com/user-attachments/assets/7073b7b9-a830-416b-ab12-cef8458f775f" />

The dashboard features:
- Real-time ECG waveform with green trace on dark background
- Heart rate gauge with color-coded zones
- AI diagnosis panel with confidence scores
- Risk assessment visualization
- HRV metrics display


## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Sanjidh090/Low-cost-ECG-Monitoring.git
cd Low-cost-ECG-Monitoring
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run ecg_dashboard.py
```

4. Open your browser to `http://localhost:8501`

### Running Offline

The Streamlit dashboard can run completely offline:
```bash
streamlit run ecg_dashboard.py --server.headless true
```

### Real-time ECG from ESP32

For a simple, lightweight real-time ECG visualization directly from an ESP32:

```bash
python realtime_ecg.py
```

**Configuration:**
1. Edit `realtime_ecg.py` and set `SERIAL_PORT` to match your ESP32 port:
   - Windows: `"COM4"` (or similar)
   - Linux: `"/dev/ttyUSB0"` (or similar)
   - macOS: `"/dev/cu.usbserial-XXXX"` (or similar)

2. The ESP32 should send raw ADC values (0-4095) as plain text, one value per line.

3. Press `Ctrl+C` to stop the visualization.

## 📁 Project Structure

```
Low-cost-ECG-Monitoring/
├── ecg_dashboard.py      # Main Streamlit dashboard application
├── realtime_ecg.py       # Simple real-time ECG from ESP32 (matplotlib)
├── requirements.txt      # Python dependencies
├── ecg_brain_advanced.pkl # Trained ML model for diagnosis
├── train_ai.py          # Script to train the AI model
├── maks.py              # PyQt5 desktop dashboard (alternative)
├── wthnd.py             # Basic PyQt5 ECG viewer
├── app2.py              # Legacy Streamlit app
├── combined_1.csv       # Sample ECG data
├── combined_2.csv       # Sample ECG data
├── combined_3.csv       # Sample ECG data
├── combined_4.csv       # Sample ECG data
└── README.md            # This file
```

## 🔧 Configuration

### Data Sources

1. **Demo Mode**: Generates realistic simulated ECG data
   - Adjustable heart rate (40-150 BPM)
   - Includes P-wave, QRS complex, and T-wave

2. **CSV Files**: Load pre-recorded ECG data
   - Expected format: `timestamp_ms,ecg_value`
   - Automatically loads all `combined_*.csv` files

3. **ESP32 Hardware**: (Requires PySerial)
   - Configure COM port in settings
   - Baud rate: 115200
   - JSON format: `{"t": timestamp, "v": value}`

### AI Model

The diagnostic model (`ecg_brain_advanced.pkl`) is trained using `train_ai.py`:
```bash
python train_ai.py
```

Features used for classification:
- Heart rate (BPM)
- RMSSD (Root Mean Square of Successive Differences)
- SDNN (Standard Deviation of NN intervals)
- QRS width
- QT interval
- ST elevation

## 🏥 Medical Conditions Detected

| Condition | Description | Risk Level |
|-----------|-------------|------------|
| Normal Sinus Rhythm | Healthy heart rhythm | ✅ Normal |
| Atrial Fibrillation | Irregular heartbeat | ⚠️ Warning |
| PVC | Premature ventricular contractions | ⚠️ Warning |
| STEMI | ST-elevation myocardial infarction | 🔴 Danger |
| Long QT Syndrome | Extended QT interval | ⚠️ Warning |
| Bradycardia | Heart rate < 60 BPM | ⚠️ Warning |
| Tachycardia | Heart rate > 100 BPM | ⚠️ Warning |

## ⚠️ Medical Disclaimer

**This software is for educational and research purposes only.**

This ECG monitoring system is NOT a medical device and should NOT be used for:
- Diagnosing medical conditions
- Making treatment decisions
- Replacing professional medical advice

Always consult a qualified healthcare professional for any medical concerns.

## 🛠️ Hardware Setup (Optional)

### ESP32 Connection

1. Flash your ESP32 with the ECG reading firmware
2. Connect the AD8232 ECG sensor module
3. Configure the serial port in the dashboard settings
4. Switch data source to "ESP32 Serial"

### Recommended Hardware:
- ESP32 Development Board
- AD8232 ECG Sensor Module
- ECG electrodes (3-lead)
- Jumper wires

## 📸 Setup Images 

<p align="center">
  <img src="https://github.com/user-attachments/assets/0a73b7af-5381-4ae5-a34e-6de11a21b399" width="48%" />
  <img src="https://github.com/user-attachments/assets/4d5c89db-f939-4c28-8a12-7f41c1398d85" width="48%" />
</p>

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you have any questions or issues, please open an issue on GitHub.
