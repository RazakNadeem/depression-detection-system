# Depression Detection System using Multi-Modal Analysis

This project detects depression signals from video uploads by analyzing:
1. **Facial Expressions** (Visual Modality) - Using FER-2013 based models.
2. **Speech Sentiment** (Audio Modality) - Using OpenAI Whisper and TextBlob.

## Features
- **Doctor Login/Registration**: secure authentication.
- **Video Analysis**: Upload patient videos for automated analysis.
- **Reports**: Generates downloadable PDF reports with score and breakdown.
- **Dashboard**: Track patient history.
- **Attractive UI**: Modern, responsive design.

## Prerequisites
- Python 3.8+
- Active Internet connection (for Whisper model download)
- **FFmpeg** installed on your system (Required for video/audio processing)

## Setup Instructions

1. **Create Virtual Environment**:
   Open a terminal in this directory and run:
   ```bash
   python -m venv venv
   ```

2. **Activate Virtual Environment**:
   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies**:
   With the virtual environment activated, run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Database**:
   The database will be created automatically on the first run.

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the App**:
   Open your browser and go to: `http://127.0.0.1:5000`

## Usage Guide
1. **Register** a new Doctor account.
2. **Login** to access the Dashboard.
3. Click "Analyze New Video" or upload from the Dashboard.
4. **Upload a video file** (mp4, avi, mov) where the patient's face is visible and they are speaking.
5. Wait for the analysis to complete (this may take a minute depending on video length).
6. View the results and **Download PDF Report**.

## Training on FER-2013 (Optional)
This application uses a pre-trained model for facial expression recognition to work out-of-the-box.
If you wish to train your own model on the FER-2013 dataset as per project requirements:
1. Download the FER-2013 dataset from Kaggle.
2. Organize it into `data/fer2013/train` and `data/fer2013/test`.
3. Run:
   ```bash
   python train_model.py
   ```
4. This will save `custom_fer_model.h5` which you can then load in `utils_ml.py`.

## troubleshooting
- **FFmpeg Error**: If audio extraction fails, ensure FFmpeg is installed or `imageio-ffmpeg` is updated.
- **No Face Detected**: Ensure the video has good lighting and the face is clearly visible.
