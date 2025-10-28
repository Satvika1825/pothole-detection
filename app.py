from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from dotenv import load_dotenv
load_dotenv()
import sqlite3
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import cv2
import base64
from pathlib import Path

# ========================
# Flask Configuration
# ========================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback_secret_key")

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
VIDEO_FOLDER = 'static/videos'
DETECTED_FRAMES_FOLDER = 'static/detected_frames'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER
app.config['DETECTED_FRAMES_FOLDER'] = DETECTED_FRAMES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Create directories
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, VIDEO_FOLDER, DETECTED_FRAMES_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(RESULT_FOLDER, 'detected'), exist_ok=True)

# Allowed extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

def allowed_file(filename, file_type='image'):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    return False

# ========================
# Database Setup
# ========================
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            detection_type TEXT,
            location TEXT,
            file_path TEXT,
            result_path TEXT,
            pothole_count INTEGER DEFAULT 0,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            alert_sent BOOLEAN DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ========================
# Model Setup
# ========================
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'pothole_yolov11_best.pt')
model = None

def load_model():
    global model
    if model is not None:
        return model
    if not os.path.isfile(MODEL_PATH):
        app.logger.error(f"Model file not found at {MODEL_PATH}")
        return None
    try:
        app.logger.info(f"Loading model from {MODEL_PATH} ...")
        model = YOLO(MODEL_PATH)
        app.logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        app.logger.exception(f"Failed to load model: {e}")
        model = None
        return None

load_model()

# ========================
# Email Notification with Images
# ========================
def notify_authorities(detection_data):
    """
    Send email to authorities with pothole images
    detection_data = {
        'images': [list of image paths],
        'location': 'location string',
        'count': number of potholes,
        'timestamp': datetime
    }
    """
    sender = os.getenv("NOTIFY_SENDER_EMAIL")
    app_password = os.getenv("NOTIFY_APP_PASSWORD")
    recipient = os.getenv("NOTIFY_RECIPIENT")

    app.logger.info(f"📧 Email notification triggered for {detection_data['count']} pothole(s)")
    app.logger.info(f"   Sender: {sender}")
    app.logger.info(f"   Recipient: {recipient}")
    app.logger.info(f"   Images to attach: {len(detection_data['images'])}")

    if not sender or not app_password or not recipient:
        app.logger.error("❌ Email credentials missing in .env")
        app.logger.error(f"   NOTIFY_SENDER_EMAIL: {'SET' if sender else 'MISSING'}")
        app.logger.error(f"   NOTIFY_APP_PASSWORD: {'SET' if app_password else 'MISSING'}")
        app.logger.error(f"   NOTIFY_RECIPIENT: {'SET' if recipient else 'MISSING'}")
        return False

    try:
        msg = MIMEMultipart()
        msg["Subject"] = f"🚨 URGENT: {detection_data['count']} Pothole(s) Detected!"
        msg["From"] = sender
        msg["To"] = recipient

        # Email body
        body = f"""
POTHOLE DETECTION ALERT
=======================

⚠️ IMMEDIATE ATTENTION REQUIRED ⚠️

Location: {detection_data['location']}
Number of Potholes: {detection_data['count']}
Detected At: {detection_data['timestamp']}
Detection Type: {detection_data.get('type', 'Image')}

Please take immediate action to repair the detected road damage.
This is an automated alert from the Pothole Detection System.

Attached: {len(detection_data['images'])} detection image(s) showing pothole locations

---
Automated Pothole Detection System
Contact: {sender}
        """
        
        msg.attach(MIMEText(body, 'plain'))

        # Attach images with better error handling
        attached_count = 0
        for idx, img_path in enumerate(detection_data['images'][:5]):  # Limit to 5 images
            if os.path.exists(img_path):
                try:
                    with open(img_path, 'rb') as f:
                        img_data = f.read()
                        image = MIMEImage(img_data, name=f"pothole_detection_{idx+1}.jpg")
                        msg.attach(image)
                        attached_count += 1
                        app.logger.info(f"   ✓ Attached image {idx+1}: {os.path.basename(img_path)}")
                except Exception as img_error:
                    app.logger.error(f"   ✗ Failed to attach image {idx+1}: {img_error}")
            else:
                app.logger.warning(f"   ✗ Image not found: {img_path}")

        app.logger.info(f"📧 Connecting to Gmail SMTP server (smtp.gmail.com:587)...")
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            app.logger.info("🔐 Starting TLS encryption...")
            server.starttls()
            
            app.logger.info(f"🔑 Logging in as {sender}...")
            server.login(sender, app_password)
            
            app.logger.info("📤 Sending email with attachments...")
            server.send_message(msg)
        
        app.logger.info("=" * 60)
        app.logger.info(f"✅ EMAIL SENT SUCCESSFULLY!")
        app.logger.info(f"   To: {recipient}")
        app.logger.info(f"   Attachments: {attached_count} image(s)")
        app.logger.info(f"   Subject: {msg['Subject']}")
        app.logger.info("=" * 60)
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        app.logger.error("=" * 60)
        app.logger.error("❌ SMTP AUTHENTICATION FAILED!")
        app.logger.error("=" * 60)
        app.logger.error("   Possible issues:")
        app.logger.error("   1. App Password is incorrect")
        app.logger.error("   2. 2-Step Verification not enabled")
        app.logger.error("   3. Need to generate new App Password")
        app.logger.error(f"   Error: {e}")
        app.logger.error("=" * 60)
        return False
        
    except smtplib.SMTPException as e:
        app.logger.error(f"❌ SMTP Error: {e}")
        return False
        
    except Exception as e:
        app.logger.exception(f"❌ Email sending failed with unexpected error: {e}")
        return False

# ========================
# Video Processing Function
# ========================
def process_video(video_path, location, user_id):
    """Process video and extract frames with potholes"""
    m = load_model()
    if m is None:
        return None, []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // 2)  # Process 2 frames per second
    
    frame_count = 0
    detected_frames = []
    pothole_images = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(app.config['DETECTED_FRAMES_FOLDER'], timestamp)
    os.makedirs(output_folder, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Run detection on frame
            results = m.predict(source=frame, save=False, verbose=False)
            
            # If potholes detected, save frame
            if len(results[0].boxes) > 0:
                annotated = results[0].plot()
                frame_filename = f"frame_{frame_count}_potholes_{len(results[0].boxes)}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, annotated)
                
                # FIX: Convert path to forward slashes
                rel_path = frame_path.replace('\\', '/').replace('static/', '')
                
                detected_frames.append({
                    'frame_number': frame_count,
                    'pothole_count': len(results[0].boxes),
                    'path': frame_path,
                    'rel_path': rel_path
                })
                pothole_images.append(frame_path)

        frame_count += 1

    cap.release()
    
    return detected_frames, pothole_images

# ========================
# Routes
# ========================
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form.get('email', '')
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)', 
                         (username, password, email))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!', 'error')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user'] = username
            session['user_id'] = user[0]
            return redirect(url_for('upload'))
        else:
            flash('Invalid credentials!', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('user_id', None)
    return redirect(url_for('login'))

# ========================
# Upload Route (Image/Video/Camera)
# ========================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        upload_type = request.form.get('upload_type', 'image')
        location = request.form.get('location', 'Unknown')

        # Handle camera capture
        if upload_type == 'camera':
            image_data = request.form.get('camera_image')
            if not image_data:
                flash('No camera image captured.', 'error')
                return redirect(request.url)

            # Decode base64 image
            try:
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"camera_{timestamp}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                
                # Process the captured image
                return process_image_detection(filepath, location, 'camera')
                
            except Exception as e:
                flash(f'Error processing camera image: {e}', 'error')
                return redirect(request.url)

        # Handle file upload (image or video)
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if it's video or image
        if allowed_file(filename, 'video'):
            upload_filename = f"{timestamp}_{filename}"
            upload_path = os.path.join(app.config['VIDEO_FOLDER'], upload_filename)
            file.save(upload_path)
            return process_video_detection(upload_path, location)
        
        elif allowed_file(filename, 'image'):
            upload_filename = f"{timestamp}_{filename}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
            file.save(upload_path)
            return process_image_detection(upload_path, location, 'image')
        
        else:
            flash('Invalid file type. Please upload an image or video.', 'error')
            return redirect(request.url)

    return render_template('upload.html')

def process_image_detection(image_path, location, detection_type):
    """Process single image detection"""
    m = load_model()
    if m is None:
        flash("Model not loaded. Check server logs.", 'error')
        return redirect(url_for('upload'))

    try:
        results = m.predict(source=image_path, save=False, verbose=False)
        annotated_image = results[0].plot()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"detected_{timestamp}.jpg"
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'detected', result_filename)
        cv2.imwrite(result_path, annotated_image)

        pothole_count = len(results[0].boxes)
        pothole_detected = pothole_count > 0

        # Save to database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO detections 
                         (user_id, detection_type, location, file_path, result_path, pothole_count, alert_sent)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                      (session['user_id'], detection_type, location, image_path, result_path, pothole_count, pothole_detected))
        conn.commit()
        conn.close()

        # Send alert if pothole detected
        if pothole_detected:
            detection_data = {
                'images': [result_path],
                'location': location,
                'count': pothole_count,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'type': detection_type.capitalize()
            }
            notify_authorities(detection_data)

        # FIX: Convert Windows backslashes to forward slashes for URL
        rel_path = result_path.replace('\\', '/').replace('static/', '')
        
        return render_template('results.html',
                             result="Pothole Detected!" if pothole_detected else "No Pothole Detected",
                             location=location,
                             image_path=url_for('static', filename=rel_path),
                             pothole_count=pothole_count,
                             detection_type=detection_type,
                             pothole_detected=pothole_detected)

    except Exception as e:
        app.logger.exception(f"Detection error: {e}")
        flash(f"Error: {e}", 'error')
        return redirect(url_for('upload'))

def process_video_detection(video_path, location):
    """Process video detection"""
    detected_frames, pothole_images = process_video(video_path, location, session['user_id'])
    
    if detected_frames is None:
        flash("Error processing video.", 'error')
        return redirect(url_for('upload'))

    total_potholes = sum(frame['pothole_count'] for frame in detected_frames)
    
    # Save to database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO detections 
                     (user_id, detection_type, location, file_path, pothole_count, alert_sent)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (session['user_id'], 'video', location, video_path, total_potholes, total_potholes > 0))
    conn.commit()
    conn.close()

    # Send alert if potholes detected
    if total_potholes > 0:
        detection_data = {
            'images': pothole_images,
            'location': location,
            'count': total_potholes,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': 'Video'
        }
        notify_authorities(detection_data)

    return render_template('video_results.html',
                         detected_frames=detected_frames,
                         location=location,
                         total_potholes=total_potholes,
                         frame_count=len(detected_frames))

# ========================
# Run Flask App
# ========================
if __name__ == '__main__':
    app.run(debug=True, threaded=True)