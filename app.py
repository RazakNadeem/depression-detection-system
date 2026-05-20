import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, MedicalReport
from utils_ml import DepressionAnalyzer
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_12345'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///depression.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['REPORT_FOLDER'] = 'static/reports'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name')
        secret_question = request.form.get('secret_question')
        secret_answer = request.form.get('secret_answer')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        new_user = User(
            username=username, 
            password=generate_password_hash(password), 
            name=name,
            secret_question=secret_question,
            secret_answer=secret_answer
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        form_step = request.form.get('step')
        username = request.form.get('username')
        user = User.query.filter_by(username=username).first()
        
        if not user:
            flash('Username not found')
            return render_template('forgot_password.html', step=1)

        if form_step == '1':
            return render_template('forgot_password.html', step=2, user=user)
        
        elif form_step == '2':
            answer = request.form.get('answer')
            if user.secret_answer and user.secret_answer.lower() == answer.lower():
                return render_template('forgot_password.html', step=3, user=user)
            else:
                flash('Incorrect answer to security question')
                return render_template('forgot_password.html', step=2, user=user)
        
        elif form_step == '3':
            new_password = request.form.get('new_password')
            user.password = generate_password_hash(new_password)
            db.session.commit()
            flash('Password reset successful! Please login.')
            return redirect(url_for('login'))

    return render_template('forgot_password.html', step=1)

@app.route('/dashboard')
@login_required
def dashboard():
    all_reports = MedicalReport.query.filter_by(user_id=current_user.id).order_by(MedicalReport.timestamp.desc()).all()
    
    # Group by Patient ID
    patient_groups = {}
    for report in all_reports:
        pid = report.patient_id
        if pid not in patient_groups:
            patient_groups[pid] = []
        patient_groups[pid].append(report)
        
    return render_template('dashboard.html', user=current_user, patient_groups=patient_groups)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['video']
        patient_id = request.form.get('patient_id') # Get Patient ID

        if not patient_id:
            flash('Patient ID is required!')
            return redirect(request.url)

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file:
            # Create Patient Specific Folders
            patient_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], patient_id)
            patient_report_dir = os.path.join(app.config['REPORT_FOLDER'], patient_id)
            os.makedirs(patient_upload_dir, exist_ok=True)
            os.makedirs(patient_report_dir, exist_ok=True)

            filename = secure_filename(file.filename)
            filepath = os.path.join(patient_upload_dir, filename)
            file.save(filepath)
            
            # Start Analysis - Output to patient folder
            analyzer = DepressionAnalyzer(filepath, patient_upload_dir)
            
            # 1. Extraction
            try:
                audio_path = analyzer.extract_audio()
                if not audio_path:
                    # flash("Failed to extract audio. Ensure video has sound.")
                    print("Warn: Audio extraction failed or returned None")
            except Exception as e:
                print(f"Extraction Error: {e}")
                audio_path = None
            
            # 2. Audio Analysis
            if audio_path:
                text_content, sentiment = analyzer.analyze_audio_text(audio_path)
                # Cleanup audio file after transcription to prevent reuse
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except:
                        pass
            else:
                text_content = "[No Audio Track]"
                sentiment = 0.0 # Neutral
            
            # 3. Video Analysis
            emotion_counts, face_pct, total_frames = analyzer.analyze_video_frames()
            
            is_no_speech = (text_content == "[No Audio Track]" or not text_content.strip())
            if face_pct == 0 or is_no_speech:
                try:
                    os.remove(filepath)
                except:
                    pass
                flash('Error: Could not detect face or speech in the video. Please upload a video with a clear face and audible speech.')
                return redirect(request.url)

            # 4. Calculate Score
            score = analyzer.calculate_depression_score(emotion_counts, sentiment)
            
            # Cleanup: Close video handles to release files
            try:
                if 'analyzer' in locals():
                    # If we added a close method to analyzer or just close clips inside upload_video
                    pass 
            except:
                pass

            # 5. Generate Report
            timestamp_str = secure_filename(str(datetime.now().timestamp())) 
            report_filename = f"report_{patient_id}_{timestamp_str}.pdf"
            report_path = os.path.join(patient_report_dir, report_filename)
            
            # Relative path for DB storage (folder/filename)
            relative_report_path = os.path.join(patient_id, report_filename).replace("\\", "/")

            report_data = {
                'doctor_name': current_user.name,
                'filename': filename,
                'score': score,
                'face_percentage': round(face_pct, 2),
                'emotions': emotion_counts,
                'text': text_content,
                'sentiment': sentiment
            }
            try:
                analyzer.generate_pdf_report(report_data, report_path)
            except Exception as e:
                print(f"Error generating PDF: {e}")
                
            # Save to DB
            json_emotions = json.dumps(emotion_counts)
            
            new_report = MedicalReport(
                patient_id=patient_id,
                user_id=current_user.id,
                video_filename=filename,
                depression_score=score,
                dominant_emotion=max(emotion_counts, key=emotion_counts.get) if emotion_counts else "None",
                audio_sentiment=f"{sentiment:.2f}",
                audio_text=text_content,
                result_pdf=relative_report_path, # Store relative path including folder
                face_percentage=face_pct,
                emotions_json=json_emotions
            )
            db.session.add(new_report)
            db.session.commit()
            
            return redirect(url_for('result', report_id=new_report.id))
            
    return render_template('upload.html')

@app.route('/result/<int:report_id>')
@login_required
def result(report_id):
    report = MedicalReport.query.get_or_404(report_id)
    if report.user_id != current_user.id:
        flash('Access denied')
        return redirect(url_for('dashboard'))
    emotions_data = report.get_emotions()
    return render_template('result.html', report=report, emotions=emotions_data)

@app.route('/download/<int:report_id>')
@login_required
def download_report(report_id):
    report = MedicalReport.query.get_or_404(report_id)
    if report.user_id != current_user.id:
        flash('Access denied')
        return redirect(url_for('dashboard'))
    
    # Path is stored as "patient_id/filename.pdf"
    return send_file(os.path.join(app.config['REPORT_FOLDER'], report.result_pdf), as_attachment=True)

@app.route('/rename_patient/<patient_id>', methods=['POST'])
@login_required
def rename_patient(patient_id):
    new_patient_id = request.form.get('new_patient_id')
    if not new_patient_id or new_patient_id == patient_id:
        flash('Invalid or identical new patient ID.')
        return redirect(url_for('dashboard'))
        
    reports = MedicalReport.query.filter_by(patient_id=patient_id, user_id=current_user.id).all()
    if not reports:
        flash('Patient not found or access denied.')
        return redirect(url_for('dashboard'))
        
    old_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], patient_id)
    new_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], new_patient_id)
    old_report_dir = os.path.join(app.config['REPORT_FOLDER'], patient_id)
    new_report_dir = os.path.join(app.config['REPORT_FOLDER'], new_patient_id)
    
    try:
        if os.path.exists(new_upload_dir) or os.path.exists(new_report_dir):
            flash(f'Patient ID {new_patient_id} already exists. Merge not supported.')
            return redirect(url_for('dashboard'))

        if os.path.exists(old_upload_dir):
            os.rename(old_upload_dir, new_upload_dir)
        if os.path.exists(old_report_dir):
            os.rename(old_report_dir, new_report_dir)
            
        for report in reports:
            report.patient_id = new_patient_id
            if report.result_pdf and report.result_pdf.startswith(f"{patient_id}/"):
                report.result_pdf = report.result_pdf.replace(f"{patient_id}/", f"{new_patient_id}/", 1)
        db.session.commit()
        flash(f'Patient ID renamed to {new_patient_id} successfully.')
    except Exception as e:
        db.session.rollback()
        flash(f'Error renaming patient: {e}')
        
    return redirect(url_for('dashboard'))

@app.route('/delete_patient/<patient_id>', methods=['POST'])
@login_required
def delete_patient(patient_id):
    # secure_filename to prevent directory traversal attack
    safe_patient_id = secure_filename(patient_id)
    
    # 1. Delete records from DB
    reports = MedicalReport.query.filter_by(patient_id=patient_id, user_id=current_user.id).all()
    if not reports:
        flash('Patient not found or access denied.')
        return redirect(url_for('dashboard'))
        
    for report in reports:
        db.session.delete(report)
    db.session.commit()
    
    # 2. Delete folders
    import shutil
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], safe_patient_id)
    report_dir = os.path.join(app.config['REPORT_FOLDER'], safe_patient_id)
    
    try:
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        if os.path.exists(report_dir):
            shutil.rmtree(report_dir)
        flash(f'Patient {patient_id} and all associated data deleted successfully.')
    except Exception as e:
        flash(f'Error deleting files: {e}')
        
    return redirect(url_for('dashboard'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        # Create tables if they don't exist
        db.create_all()
    app.run(debug=True)
