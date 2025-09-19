# Crowd Density & Stampede Prevention

A Streamlit demo to detect crowd density and prevent stampedes using YOLOv8 this is my  first project on opencv and vast amount of libraries

---

## Project Description
This project uses computer vision to monitor crowd density in real-time. It provides alerts to prevent overcrowding or stampede situations. The detection is powered by the YOLOv8 model, and the interface is built using Streamlit.

---

## Features
- Detects people in images or live camera feeds
- Calculates crowd density
- Visualizes results in a friendly Streamlit interface
- Easy to deploy locally

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/chavansrikar/crowd-density-prevention.git
cd crowd-density-prevention
Create a virtual environment (recommended):

bash
Copy code
python -m venv venv
venv\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Note: Make sure your requirements.txt includes at least:

nginx
Copy code
streamlit
ultralytics
opencv-python
numpy
Usage
Run the Streamlit app:

bash
Copy code
streamlit run app.py
The app interface will open in your browser.

Upload a crowd image or use your camera feed to detect people and get crowd density alerts.

Project Structure
bash
Copy code
strampade_project/
│
├── app.py          # Main Streamlit app
├── file.py         # Utility functions
├── yolov8n.pt      # Pre-trained YOLOv8 model
├── requirements.txt
└── README.md
Contribution
Feel free to fork this repo and create pull requests. Please ensure all new features are well-tested before submitting.

License
This project is for educational/demo purposes.

pgsql
Copy code

---

If you want, I can also **make a ready-to-copy `requirements.txt`** with all packages needed to run your project without errors — then your teammates can literally clone + install + run.  

Do you want me to do that too?
