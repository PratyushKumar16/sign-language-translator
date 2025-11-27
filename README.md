# sign-language-detector-flask-python
This project aims to create a sign language translator using machine learning techniques and Python programming. The application utilizes various modules, primarily Mediapipe, Landmark, and Random Forest algorithms to interpret and translate sign language gestures into text or spoken language.

## Project Overview
Sign language is a crucial form of communication for individuals with hearing impairments. This project focuses on bridging the communication gap by creating a tool that can interpret sign language gestures in real-time and convert them into understandable text or speech.
  
## Features
 - Real-time sign language recognition: Captures hand gestures using the Mediapipe library to track landmarks and movements.
 - Landmark analysis: Utilizes Landmark module to extract key points and gestures from hand movements.
 - Machine learning translation: Employs Random Forest algorithm to classify and interpret gestures into corresponding text or spoken language.
  
## Usage
 - Note that the app uses Python 9.3 Interpreter, since this version is quite outdated, it is recommended to use this program with the interpreter in a Virtual Environment (.venv).
  1. Installation:
  ```
   #Clone the repository
   git clone https://github.com/PratyushKumar16/sign-language-translator.git
   
   #Navigate to the project directory
   cd sign-language-translator
  ```
  
  2. **Install the required dependencies** using the following command:

  ```bash
    pip install -r requirements.txt
  ```
   2.1 **Note that there might be errors in installing packages
    - The errors are likely going to be syntax based/syntax errors.
    - The error is caused by JAX 0.4.30, which uses the match / case pattern matching syntax.
    - That syntax requires Python 3.10+, but your venv is using Python 3.9, so it crashes.
   2.2 **Fix
    - You need to install an older version of jax that supports Python 3.9.
    - Here’s what to do, inside your (.venv), in your bash terminal, run:
     ```
       pip install "jax==0.4.13" "jaxlib==0.4.13"
      ```
    - this version supports python 3.9
    - Now install the remaining dependencies-
    ```
      pip install -r requirements.txt --no-deps
    ```
    - this tells pip to install your listed packages without trying to upgrade jax again.
     
        
     
  3. Run the application:
  ```
   python sign-language-detector-flask-python.py
  ```
   
  4. Interact with the translator :
   - Activate the camera for real-time gesture recognition.
   - Perform sign language gestures in front of the camera.

![hand-signs-of-the-ASL-Language.png](hand-signs-of-the-ASL-Language.png)

## Contributing
 Contributions are welcome! If you'd like to contribute to this project, feel free to open issues, create pull requests, or reach out to discuss potential improvements.
