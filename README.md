# Adaptive-EMG-Control

Adaptive EMG-driven robotic hand control using Residual CNN and ESP32



\## Overview

Single-channel sEMG classification using a Residual CNN achieving 

93.8% cross-validation accuracy, with MVC-normalised adaptive 

feedforward control of a 5-DOF InMoov robotic hand.



\## System Architecture

PC (Python) → Master ESP32 → ESP-NOW → Slave ESP32 → PCA9685 → Servos



\## Requirements

pip install torch scikit-learn numpy pandas scipy matplotlib seaborn pyserial



\## Run Order

1\. python training/emg\_train\_final.py

2\. Upload arduino/master\_esp32\_final.ino to master ESP32

3\. Upload arduino/slave\_esp32\_final.ino to slave ESP32

4\. python inference/emg\_live\_final.py

