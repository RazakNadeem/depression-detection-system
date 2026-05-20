from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    name = db.Column(db.String(150))
    secret_question = db.Column(db.String(255))
    secret_answer = db.Column(db.String(255))
    reports = db.relationship('MedicalReport', backref='doctor', lazy=True)

class MedicalReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), nullable=False, default="Unknown") # Added Patient ID
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_filename = db.Column(db.String(255))
    depression_score = db.Column(db.Float)
    dominant_emotion = db.Column(db.String(50))
    audio_sentiment = db.Column(db.String(50))
    audio_text = db.Column(db.Text)
    result_pdf = db.Column(db.String(255))
    face_percentage = db.Column(db.Float) # % of frames with faces detected
    emotions_json = db.Column(db.Text) # Store Python dict as JSON string

    def get_emotions(self):
        try:
            return json.loads(self.emotions_json) if self.emotions_json else {}
        except:
            return {}

    def __repr__(self):
        return f"<Report {self.id} for User {self.user_id}>"
