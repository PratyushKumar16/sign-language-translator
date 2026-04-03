# Sign-Language-Translator (Python Edition)

An interactive, real-time sign language translation system built using Python, Mediapipe, and Flask. This project enables seamless communication by translating hand gestures into text and speech, providing an accessible bridge between sign language and spoken languages.

---

## 🛠 Project Overview
This repository features a refined implementation of a sign language recognition engine. Inspired by earlier research in hand-pose estimation, this version has been optimized with modern libraries and a robust Flask-based web interface. It uses **Mediapipe** for high-precision landmark detection and a **Random Forest** model for gesture classification.

## 🚀 Key Features
- **Real-Time Translation**: Translates ASL characters into text and speech with low latency.
- **Web Dashboard**: An integrated Flask-based interface for easy interaction and display.
- **Multi-Modal Output**: Supports both visual text feedback and **Text-to-Speech** (via pyttsx3).
- **Improved Compatibility**: Updated dependency handling to support a wider range of Python environments.

## 🏗 Tech Stack
- **Python 3.9+**
- **Mediapipe**: For hand landmark tracking.
- **Flask**: To serve the application interface.
- **OpenCV**: For camera feed processing.
- **Scikit-learn**: For the classification model.
- **pyttsx3**: For integrated text-to-speech output.

## 🚦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PratyushKumar16/sign-language-translator.git
   cd sign-language-translator
   ```

2. **Set up a virtual environment** (Highly recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   *Note: To ensure compatibility with Python 3.9 environments, use the following sequence:*
   ```bash
   pip install "jax==0.4.13" "jaxlib==0.4.13"
   pip install -r requirements.txt --no-deps
   ```

## 🖥 How to Run
1. **Launch the Web App**:
   ```bash
   python app.py
   ```
2. **Access the Interface**: Open your browser and navigate to `http://127.0.0.1:5000`.
3. **Standalone Inference (Local Demo)**:
   ```bash
   python inference_classifier.py
   ```

## 📄 License
This project is open-source and intended for educational and personal use. Special thanks to the community for providing the foundational concepts and research that inspired this implementation.

---
*Modified and maintained by Pratyush Kumar.*
