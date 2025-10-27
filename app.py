from flask import Flask, render_template, request, redirect, url_for, session, flash
from dotenv import load_dotenv
load_dotenv()
import sqlite3
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import glob

# ========================
# Flask Configuration
# ========================
app = Flask(__name__)
# Flask secret key from .env
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback_secret_key")

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

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
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ========================
# Model Setup (correct path & robust loading)
# ========================
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'pothole_yolov11_best.pt')
model = None  # global variable

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

# Try load model on startup (best-effort)
load_model()

# ========================
# Email Notification
# ========================
def notify_authorities(image_path, location="Unknown"):
    sender = os.getenv("NOTIFY_SENDER_EMAIL")
    app_password = os.getenv("NOTIFY_APP_PASSWORD")
    recipient = os.getenv("NOTIFY_RECIPIENT")

    if not sender or not app_password or not recipient:
        app.logger.error("❌ Email credentials missing in .env")
        return

    msg = MIMEText(f"A pothole has been detected at {location}.\n\nImage path: {image_path}")
    msg["Subject"] = "🚨 Pothole Alert!"
    msg["From"] = sender
    msg["To"] = recipient

    try:
        app.logger.info(f"📧 Connecting to Gmail SMTP as {sender}")
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, app_password)
            server.send_message(msg)
        app.logger.info("✅ Email sent successfully to municipality.")
    except Exception as e:
        app.logger.exception(f"❌ Email sending failed: {e}")

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
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!')
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
            return redirect(url_for('upload'))
        else:
            flash('Invalid credentials!')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ========================
# /upload Route
# ========================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('image')
        location = request.form.get('location', 'Unknown')

        if not file or file.filename == '':
            flash('No file selected.')
            return redirect(request.url)

        # Save uploaded file
        original = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_filename = f"{timestamp}_{original}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
        file.save(upload_path)

        # Load YOLO model
        m = load_model()
        if m is None:
            flash("Model not loaded. Check server logs.")
            return redirect(request.url)

        try:
            # Run detection
            results = m.predict(
                source=upload_path,
                save=False  # save manually
            )

            # Annotate image
            annotated_image = results[0].plot()
            detected_folder = os.path.join(app.config['RESULT_FOLDER'], 'detected')
            os.makedirs(detected_folder, exist_ok=True)
            result_image_filename = f"detected_{upload_filename}"
            result_image_path = os.path.join(detected_folder, result_image_filename)
            from cv2 import imwrite
            imwrite(result_image_path, annotated_image)

            # Pothole detection
            pothole_detected = len(results[0].boxes) > 0
            result_text = "🚧 Pothole Detected!" if pothole_detected else "✅ No Pothole Detected."

            # Notify authorities
            if pothole_detected:
                notify_authorities(result_image_path, location)

            # Static URL
            rel_path = result_image_path.split('static/')[1].replace('\\', '/')
            image_url = url_for('static', filename=rel_path)

            return render_template(
                'results.html',
                result=result_text,
                location=location,
                image_path=image_url
            )

        except Exception as e:
            flash(f"Error: {e}")
            return redirect(request.url)

    return render_template('upload.html')

# ========================
# Run Flask App
# ========================
if __name__ == '__main__':
    app.logger.info(f"MODEL_PATH = {MODEL_PATH}")
    app.logger.info(f"UPLOAD_FOLDER = {UPLOAD_FOLDER}")
    app.logger.info(f"RESULT_FOLDER = {RESULT_FOLDER}")
    app.run(debug=True)
