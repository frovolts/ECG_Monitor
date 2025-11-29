"""
Real-time ECG Visualization from ESP32
=======================================
A simple matplotlib-based script for visualizing live ECG data
from an ESP32 microcontroller via serial port.

Hardware Requirements:
- ESP32 Development Board
- AD8232 ECG Sensor Module
- ECG electrodes (3-lead)

Usage:
    python realtime_ecg.py

Configuration:
    Modify SERIAL_PORT and BAUD_RATE constants below to match your setup.
"""

import serial
import matplotlib.pyplot as plt
from collections import deque

# -------------------------------
# CONFIGURE YOUR SERIAL PORT HERE
# -------------------------------
SERIAL_PORT = "COM4"  # Change to your ESP32 port (e.g., "COM4" on Windows, "/dev/ttyUSB0" on Linux)
BAUD_RATE = 115200
MAX_POINTS = 250       # How many points to show on the graph


def main():
    """Main function to run the real-time ECG visualization."""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE}")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        print(f"Please check that {SERIAL_PORT} is correct and the device is connected.")
        return

    plt.ion()  # Interactive mode ON
    fig, ax = plt.subplots(figsize=(12, 4))  # Wider figure
    data = deque([0] * MAX_POINTS, maxlen=MAX_POINTS)
    line, = ax.plot(data)
    ax.set_ylim(0, 4095)  # ESP32 ADC range
    ax.set_xlim(0, MAX_POINTS)  # Fix X-axis width
    ax.set_xlabel("Sample")
    ax.set_ylabel("ECG Value")
    ax.set_title("Live ECG from ESP32")

    try:
        while True:
            raw = ser.readline()
            line_raw = raw.decode(errors="ignore").strip()

            if not line_raw:
                continue

            # Parse ECG value - handle both integer and float formats
            try:
                ecg_value = int(float(line_raw))
            except ValueError:
                continue

            data.append(ecg_value)

            line.set_ydata(data)
            line.set_xdata(range(len(data)))

            plt.pause(0.01)  # pause to update plot
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
