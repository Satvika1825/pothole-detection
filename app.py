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
from datetime import datetime, timedelta
import cv2
import base64
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
import json

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
# Priority & Severity Configuration
# ========================
SEVERITY_LEVELS = {
    'CRITICAL': {'min_area': 5000, 'priority': 1, 'response_time': 24},  # 24 hours
    'HIGH': {'min_area': 2000, 'priority': 2, 'response_time': 72},      # 3 days
    'MEDIUM': {'min_area': 500, 'priority': 3, 'response_time': 168},    # 1 week
    'LOW': {'min_area': 0, 'priority': 4, 'response_time': 720}          # 30 days
}

ESCALATION_THRESHOLDS = {
    'days_overdue': 2,
    'severity_upgrade_days': 7,
    'repeat_detection_count': 3
}

# ========================
# Enhanced Database Setup
# ========================
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Enhanced detections table with severity and priority
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            detection_type TEXT,
            location TEXT,
            latitude REAL,
            longitude REAL,
            file_path TEXT,
            result_path TEXT,
            pothole_count INTEGER DEFAULT 0,
            severity TEXT DEFAULT 'LOW',
            priority INTEGER DEFAULT 4,
            status TEXT DEFAULT 'PENDING',
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            alert_sent BOOLEAN DEFAULT 0,
            last_alert_sent TIMESTAMP,
            response_due_date TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Individual potholes table for tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS potholes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            pothole_index INTEGER,
            x_center REAL,
            y_center REAL,
            width REAL,
            height REAL,
            area REAL,
            confidence REAL,
            severity TEXT,
            FOREIGN KEY (detection_id) REFERENCES detections (id)
        )
    ''')
    
    # Pothole clusters for route optimization
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pothole_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_name TEXT,
            center_latitude REAL,
            center_longitude REAL,
            radius_meters REAL,
            pothole_count INTEGER,
            total_severity_score REAL,
            recommended_route TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Cluster members (which detections belong to which cluster)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cluster_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id INTEGER NOT NULL,
            detection_id INTEGER NOT NULL,
            distance_from_center REAL,
            FOREIGN KEY (cluster_id) REFERENCES pothole_clusters (id),
            FOREIGN KEY (detection_id) REFERENCES detections (id)
        )
    ''')
    
    # Repair history and tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS repair_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            status TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_by TEXT,
            notes TEXT,
            FOREIGN KEY (detection_id) REFERENCES detections (id)
        )
    ''')
    
    # Escalation log
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS escalations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            previous_severity TEXT,
            new_severity TEXT,
            previous_priority INTEGER,
            new_priority INTEGER,
            reason TEXT,
            escalated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (detection_id) REFERENCES detections (id)
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
        
        existing_classes = model.names
        
        if len(existing_classes) == 1:
            model.names = {0: 'pothole'}
        elif len(existing_classes) == 2:
            model.names = {0: 'pothole', 1: 'normal'}
        elif len(existing_classes) >= 3:
            app.logger.warning(f"⚠️ Model has {len(existing_classes)} classes")
        
        max_class_id = max(existing_classes.keys()) if existing_classes else 0
        for i in range(max_class_id + 1):
            if i not in model.names:
                model.names[i] = f'unknown_class_{i}'
        
        app.logger.info(f"✅ Final model classes: {model.names}")
        return model

    except Exception as e:
        app.logger.exception(f"Failed to load model: {e}")
        model = None
        return None

# ========================
# Priority & Severity Functions
# ========================
def calculate_pothole_severity(box_data):
    """
    Calculate severity based on pothole size
    box_data: YOLO box object with xyxy coordinates
    Returns: severity level and area
    """
    # Extract bounding box coordinates
    x1, y1, x2, y2 = box_data.xyxy[0].cpu().numpy()
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    # Determine severity based on area
    if area >= SEVERITY_LEVELS['CRITICAL']['min_area']:
        severity = 'CRITICAL'
    elif area >= SEVERITY_LEVELS['HIGH']['min_area']:
        severity = 'HIGH'
    elif area >= SEVERITY_LEVELS['MEDIUM']['min_area']:
        severity = 'MEDIUM'
    else:
        severity = 'LOW'
    
    return severity, area

def get_overall_severity(potholes_data):
    """Get the highest severity level from multiple potholes"""
    if not potholes_data:
        return 'LOW', 4
    
    severities = [p['severity'] for p in potholes_data]
    
    if 'CRITICAL' in severities:
        return 'CRITICAL', 1
    elif 'HIGH' in severities:
        return 'HIGH', 2
    elif 'MEDIUM' in severities:
        return 'MEDIUM', 3
    else:
        return 'LOW', 4

def calculate_response_due_date(severity):
    """Calculate when response is due based on severity"""
    response_hours = SEVERITY_LEVELS[severity]['response_time']
    return datetime.now() + timedelta(hours=response_hours)

def check_and_escalate_detections():
    """
    Background task to check for overdue repairs and escalate severity
    Should be called periodically (e.g., daily cron job)
    """
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Find overdue detections
    cursor.execute('''
        SELECT id, severity, priority, response_due_date, detected_at
        FROM detections
        WHERE status IN ('PENDING', 'IN_PROGRESS')
        AND response_due_date < ?
    ''', (datetime.now(),))
    
    overdue = cursor.fetchall()
    
    for detection_id, current_severity, current_priority, due_date, detected_at in overdue:
        # Calculate days overdue
        days_overdue = (datetime.now() - datetime.strptime(due_date, '%Y-%m-%d %H:%M:%S')).days
        
        # Escalate if needed
        if days_overdue >= ESCALATION_THRESHOLDS['days_overdue']:
            new_severity, new_priority = escalate_severity(current_severity, current_priority)
            
            if new_severity != current_severity:
                # Update detection
                new_due_date = calculate_response_due_date(new_severity)
                cursor.execute('''
                    UPDATE detections
                    SET severity = ?, priority = ?, response_due_date = ?
                    WHERE id = ?
                ''', (new_severity, new_priority, new_due_date, detection_id))
                
                # Log escalation
                cursor.execute('''
                    INSERT INTO escalations 
                    (detection_id, previous_severity, new_severity, previous_priority, new_priority, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (detection_id, current_severity, new_severity, current_priority, new_priority,
                      f'Overdue by {days_overdue} days'))
                
                app.logger.warning(f"🚨 Escalated detection #{detection_id}: {current_severity} → {new_severity}")
    
    conn.commit()
    conn.close()

def escalate_severity(current_severity, current_priority):
    """Move to next higher severity level"""
    severity_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    try:
        current_index = severity_order.index(current_severity)
        if current_index < len(severity_order) - 1:
            new_severity = severity_order[current_index + 1]
            new_priority = SEVERITY_LEVELS[new_severity]['priority']
            return new_severity, new_priority
    except ValueError:
        pass
    
    return current_severity, current_priority

# ========================
# Route Optimization Functions
# ========================
def cluster_nearby_potholes(detections_data, epsilon_km=0.5):
    """
    Group nearby potholes using DBSCAN clustering
    epsilon_km: maximum distance in kilometers to consider potholes as neighbors
    Returns: list of clusters with their members
    """
    if not detections_data:
        return []
    
    # Extract coordinates
    coords = []
    detection_ids = []
    
    for det in detections_data:
        if det['latitude'] and det['longitude']:
            coords.append([det['latitude'], det['longitude']])
            detection_ids.append(det['id'])
    
    if len(coords) < 2:
        return []
    
    # Convert to numpy array
    coords = np.array(coords)
    
    # DBSCAN clustering (epsilon in degrees, roughly 0.01 degrees ≈ 1 km)
    epsilon_degrees = epsilon_km / 111.0  # rough conversion
    clustering = DBSCAN(eps=epsilon_degrees, min_samples=2).fit(coords)
    
    # Organize clusters
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label == -1:  # noise point
            continue
        
        if label not in clusters:
            clusters[label] = []
        
        clusters[label].append({
            'detection_id': detection_ids[idx],
            'coords': coords[idx]
        })
    
    return clusters

def save_clusters_to_db(clusters, detections_data):
    """Save identified clusters to database"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    for cluster_id, members in clusters.items():
        # Calculate cluster center
        coords = np.array([m['coords'] for m in members])
        center_lat = np.mean(coords[:, 0])
        center_lon = np.mean(coords[:, 1])
        
        # Calculate radius (max distance from center)
        distances = [np.linalg.norm(coord - [center_lat, center_lon]) * 111 for coord in coords]
        radius = max(distances) if distances else 0
        
        # Calculate severity score
        severity_scores = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        total_score = 0
        
        for member in members:
            det = next((d for d in detections_data if d['id'] == member['detection_id']), None)
            if det:
                total_score += severity_scores.get(det['severity'], 1)
        
        # Create cluster
        cursor.execute('''
            INSERT INTO pothole_clusters 
            (cluster_name, center_latitude, center_longitude, radius_meters, 
             pothole_count, total_severity_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (f'Cluster_{cluster_id}', center_lat, center_lon, radius * 1000,
              len(members), total_score))
        
        db_cluster_id = cursor.lastrowid
        
        # Add cluster members
        for member in members:
            distance = np.linalg.norm(member['coords'] - [center_lat, center_lon]) * 111 * 1000
            cursor.execute('''
                INSERT INTO cluster_members (cluster_id, detection_id, distance_from_center)
                VALUES (?, ?, ?)
            ''', (db_cluster_id, member['detection_id'], distance))
    
    conn.commit()
    conn.close()

def generate_optimal_route(cluster_id):
    """
    Generate optimal repair route for a cluster
    Simple nearest neighbor algorithm
    """
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get all members of cluster
    cursor.execute('''
        SELECT cm.detection_id, d.latitude, d.longitude, d.severity
        FROM cluster_members cm
        JOIN detections d ON cm.detection_id = d.id
        WHERE cm.cluster_id = ?
    ''', (cluster_id,))
    
    members = cursor.fetchall()
    conn.close()
    
    if not members:
        return []
    
    # Sort by severity (CRITICAL first)
    severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    members_sorted = sorted(members, key=lambda x: severity_order.get(x[3], 4))
    
    # Build route (for now, just ordered by severity)
    route = [
        {
            'detection_id': m[0],
            'latitude': m[1],
            'longitude': m[2],
            'severity': m[3],
            'order': idx + 1
        }
        for idx, m in enumerate(members_sorted)
    ]
    
    return route

# ========================
# Enhanced Email Notification with Priority
# ========================
def notify_authorities(detection_data):
    """
    Send email to authorities with pothole images and priority information
    """
    sender = os.getenv("NOTIFY_SENDER_EMAIL")
    app_password = os.getenv("NOTIFY_APP_PASSWORD")
    recipient = os.getenv("NOTIFY_RECIPIENT")

    if not sender or not app_password or not recipient:
        app.logger.error("❌ Email credentials missing in .env")
        return False

    severity = detection_data.get('severity', 'LOW')
    priority = detection_data.get('priority', 4)
    due_date = detection_data.get('due_date', 'Unknown')
    
    # Set urgency level in subject
    urgency_markers = {
        'CRITICAL': '🚨🚨🚨 CRITICAL URGENT',
        'HIGH': '🚨 HIGH PRIORITY',
        'MEDIUM': '⚠️ MEDIUM PRIORITY',
        'LOW': 'ℹ️ LOW PRIORITY'
    }
    
    urgency = urgency_markers.get(severity, 'ℹ️')

    try:
        msg = MIMEMultipart()
        msg["Subject"] = f"{urgency}: {detection_data['count']} Pothole(s) - Action Required by {due_date}"
        msg["From"] = sender
        msg["To"] = recipient

        # Enhanced email body with priority info
        body = f"""
POTHOLE DETECTION ALERT
{'=' * 60}

🚨 SEVERITY: {severity}
⚡ PRIORITY: Level {priority}
📅 RESPONSE DUE: {due_date}
{'=' * 60}

📍 Location: {detection_data['location']}
🔢 Number of Potholes: {detection_data['count']}
🕒 Detected At: {detection_data['timestamp']}
📹 Detection Type: {detection_data.get('type', 'Image')}

SEVERITY BREAKDOWN:
{detection_data.get('severity_breakdown', 'N/A')}

⚠️ ACTION REQUIRED ⚠️
This detection requires immediate attention based on its severity level.
Failure to respond by the due date will result in automatic escalation.

Please assign a repair crew and update the status in the system.

Attached: {len(detection_data['images'])} detection image(s)

{'=' * 60}
Automated Pothole Detection System
Contact: {sender}
        """
        
        msg.attach(MIMEText(body, 'plain'))

        # Attach images
        attached_count = 0
        for idx, img_path in enumerate(detection_data['images'][:5]):
            if os.path.exists(img_path):
                try:
                    with open(img_path, 'rb') as f:
                        img_data = f.read()
                        image = MIMEImage(img_data, name=f"pothole_{severity}_{idx+1}.jpg")
                        msg.attach(image)
                        attached_count += 1
                except Exception as img_error:
                    app.logger.error(f"   ✗ Failed to attach image: {img_error}")

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(sender, app_password)
            server.send_message(msg)
        
        app.logger.info(f"✅ {severity} priority email sent successfully!")
        return True
        
    except Exception as e:
        app.logger.exception(f"❌ Email sending failed: {e}")
        return False

# ========================
# Video Processing Function
# ========================
def process_video(video_path, location, user_id, latitude=None, longitude=None):
    """Process video and extract frames with potholes"""
    m = load_model()
    if m is None:
        return None, [], []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, [], []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // 2)
    
    frame_count = 0
    detected_frames = []
    pothole_images = []
    all_potholes_data = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(app.config['DETECTED_FRAMES_FOLDER'], timestamp)
    os.makedirs(output_folder, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            results = m.predict(source=frame, save=False, verbose=False)
            
            if len(results[0].boxes) > 0:
                # Analyze each pothole
                frame_potholes = []
                for box in results[0].boxes:
                    severity, area = calculate_pothole_severity(box)
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    pothole_data = {
                        'severity': severity,
                        'area': area,
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    }
                    frame_potholes.append(pothole_data)
                    all_potholes_data.append(pothole_data)
                
                # Save annotated frame
                annotated = results[0].plot()
                frame_filename = f"frame_{frame_count}_potholes_{len(results[0].boxes)}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, annotated)
                
                rel_path = frame_path.replace('\\', '/').replace('static/', '')
                
                detected_frames.append({
                    'frame_number': frame_count,
                    'pothole_count': len(results[0].boxes),
                    'path': frame_path,
                    'rel_path': rel_path,
                    'potholes': frame_potholes
                })
                pothole_images.append(frame_path)

        frame_count += 1

    cap.release()
    
    return detected_frames, pothole_images, all_potholes_data

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
# Upload Route (Enhanced with GPS)
# ========================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        upload_type = request.form.get('upload_type', 'image')
        location = request.form.get('location', 'Unknown')
        latitude = request.form.get('latitude', type=float)
        longitude = request.form.get('longitude', type=float)

        # Handle camera capture
        if upload_type == 'camera':
            image_data = request.form.get('camera_image')
            if not image_data:
                flash('No camera image captured.', 'error')
                return redirect(request.url)

            try:
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"camera_{timestamp}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                
                return process_image_detection(filepath, location, 'camera', latitude, longitude)
                
            except Exception as e:
                flash(f'Error processing camera image: {e}', 'error')
                return redirect(request.url)

        # Handle file upload
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if allowed_file(filename, 'video'):
            upload_filename = f"{timestamp}_{filename}"
            upload_path = os.path.join(app.config['VIDEO_FOLDER'], upload_filename)
            file.save(upload_path)
            return process_video_detection(upload_path, location, latitude, longitude)
        
        elif allowed_file(filename, 'image'):
            upload_filename = f"{timestamp}_{filename}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
            file.save(upload_path)
            return process_image_detection(upload_path, location, 'image', latitude, longitude)
        
        else:
            flash('Invalid file type.', 'error')
            return redirect(request.url)

    return render_template('upload.html')

def process_image_detection(image_path, location, detection_type, latitude=None, longitude=None):
    """Process single image detection with priority and severity"""
    m = load_model()
    if m is None:
        flash("Model not loaded.", 'error')
        return redirect(url_for('upload'))

    try:
        results = m.predict(source=image_path, save=False, verbose=False)
        annotated_image = results[0].plot()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"detected_{timestamp}.jpg"
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'detected', result_filename)
        cv2.imwrite(result_path, annotated_image)

        # Analyze each pothole for severity
        potholes_data = []
        for idx, box in enumerate(results[0].boxes):
            severity, area = calculate_pothole_severity(box)
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            potholes_data.append({
                'index': idx,
                'severity': severity,
                'area': area,
                'confidence': confidence,
                'x_center': (x1 + x2) / 2,
                'y_center': (y1 + y2) / 2,
                'width': x2 - x1,
                'height': y2 - y1
            })

        pothole_count = len(potholes_data)
        pothole_detected = pothole_count > 0
        
        # Get overall severity
        overall_severity, priority = get_overall_severity(potholes_data)
        response_due = calculate_response_due_date(overall_severity)

        # Save to database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO detections 
                         (user_id, detection_type, location, latitude, longitude, 
                          file_path, result_path, pothole_count, severity, priority, 
                          response_due_date, alert_sent, last_alert_sent)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (session['user_id'], detection_type, location, latitude, longitude,
                       image_path, result_path, pothole_count, overall_severity, priority,
                       response_due, pothole_detected, datetime.now() if pothole_detected else None))
        
        detection_id = cursor.lastrowid
        
        # Save individual pothole data
        for pothole in potholes_data:
            cursor.execute('''INSERT INTO potholes 
                             (detection_id, pothole_index, x_center, y_center, width, height, 
                              area, confidence, severity)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (detection_id, pothole['index'], pothole['x_center'], pothole['y_center'],
                           pothole['width'], pothole['height'], pothole['area'], 
                           pothole['confidence'], pothole['severity']))
        
        conn.commit()
        conn.close()

        # Send alert if pothole detected
        if pothole_detected:
            severity_breakdown = '\n'.join([
                f"  - Pothole {i+1}: {p['severity']} ({p['area']:.0f} px²)"
                for i, p in enumerate(potholes_data)
            ])
            
            detection_data = {
                'images': [result_path],
                'location': location,
                'count': pothole_count,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'type': detection_type.capitalize(),
                'severity': overall_severity,
                'priority': priority,
                'due_date': response_due.strftime("%Y-%m-%d %H:%M"),
                'severity_breakdown': severity_breakdown
            }
            notify_authorities(detection_data)

        rel_path = result_path.replace('\\', '/').replace('static/', '')
        
        return render_template('results.html',
                             result="Pothole Detected!" if pothole_detected else "No Pothole Detected",
                             location=location,
                             image_path=url_for('static', filename=rel_path),
                             pothole_count=pothole_count,
                             detection_type=detection_type,
                             pothole_detected=pothole_detected,
                             severity=overall_severity,
                             priority=priority,
                             due_date=response_due.strftime("%Y-%m-%d %H:%M"),
                             potholes_data=potholes_data)

    except Exception as e:
        app.logger.exception(f"Detection error: {e}")
        flash(f"Error: {e}", 'error')
        return redirect(url_for('upload'))

def process_video_detection(video_path, location, latitude=None, longitude=None):
    """Process video detection with priority tracking"""
    detected_frames, pothole_images, all_potholes_data = process_video(
        video_path, location, session['user_id'], latitude, longitude
    )
    
    if detected_frames is None:
        flash("Error processing video.", 'error')
        return redirect(url_for('upload'))

    total_potholes = sum(frame['pothole_count'] for frame in detected_frames)
    
    # Get overall severity
    overall_severity, priority = get_overall_severity(all_potholes_data)
    response_due = calculate_response_due_date(overall_severity)
    
    # Save to database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO detections 
                     (user_id, detection_type, location, latitude, longitude, file_path, 
                      pothole_count, severity, priority, response_due_date, alert_sent, last_alert_sent)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (session['user_id'], 'video', location, latitude, longitude, video_path, 
                   total_potholes, overall_severity, priority, response_due,
                   total_potholes > 0, datetime.now() if total_potholes > 0 else None))
    conn.commit()
    conn.close()

    # Send alert
    if total_potholes > 0:
        severity_counts = {}
        for pothole in all_potholes_data:
            sev = pothole['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        severity_breakdown = '\n'.join([
            f"  - {sev}: {count} pothole(s)"
            for sev, count in sorted(severity_counts.items(), 
                                    key=lambda x: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].index(x[0]))
        ])
        
        detection_data = {
            'images': pothole_images,
            'location': location,
            'count': total_potholes,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': 'Video',
            'severity': overall_severity,
            'priority': priority,
            'due_date': response_due.strftime("%Y-%m-%d %H:%M"),
            'severity_breakdown': severity_breakdown
        }
        notify_authorities(detection_data)

    return render_template('video_results.html',
                         detected_frames=detected_frames,
                         location=location,
                         total_potholes=total_potholes,
                         frame_count=len(detected_frames),
                         severity=overall_severity,
                         priority=priority,
                         due_date=response_due.strftime("%Y-%m-%d %H:%M"))

# ========================
# Dashboard & Analytics Routes
# ========================
@app.route('/dashboard')
def dashboard():
    """Main dashboard showing priority queue and statistics"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get all pending detections ordered by priority
    cursor.execute('''
        SELECT id, location, pothole_count, severity, priority, 
               response_due_date, detected_at, latitude, longitude
        FROM detections
        WHERE status = 'PENDING'
        ORDER BY priority ASC, detected_at ASC
    ''')
    pending_detections = cursor.fetchall()
    
    # Get statistics
    cursor.execute('SELECT COUNT(*) FROM detections WHERE status = "PENDING"')
    pending_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM detections WHERE severity = "CRITICAL" AND status = "PENDING"')
    critical_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM detections WHERE response_due_date < ? AND status = "PENDING"',
                  (datetime.now(),))
    overdue_count = cursor.fetchone()[0]
    
    conn.close()
    
    # Format detections for display
    detections_list = []
    for det in pending_detections:
        due_date = datetime.strptime(det[5], '%Y-%m-%d %H:%M:%S')
        is_overdue = due_date < datetime.now()
        
        detections_list.append({
            'id': det[0],
            'location': det[1],
            'count': det[2],
            'severity': det[3],
            'priority': det[4],
            'due_date': det[5],
            'detected_at': det[6],
            'is_overdue': is_overdue,
            'latitude': det[7],
            'longitude': det[8]
        })
    
    return render_template('dashboard.html',
                         detections=detections_list,
                         pending_count=pending_count,
                         critical_count=critical_count,
                         overdue_count=overdue_count)

@app.route('/clusters')
def view_clusters():
    """View pothole clusters for route optimization"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Get all pending detections with coordinates
    cursor.execute('''
        SELECT id, location, latitude, longitude, severity, pothole_count
        FROM detections
        WHERE status = 'PENDING' AND latitude IS NOT NULL AND longitude IS NOT NULL
    ''')
    
    detections = cursor.fetchall()
    conn.close()
    
    if len(detections) < 2:
        flash('Not enough detections with GPS coordinates for clustering.', 'info')
        return redirect(url_for('dashboard'))
    
    # Format for clustering
    detections_data = [
        {
            'id': d[0],
            'location': d[1],
            'latitude': d[2],
            'longitude': d[3],
            'severity': d[4],
            'count': d[5]
        }
        for d in detections
    ]
    
    # Perform clustering
    clusters = cluster_nearby_potholes(detections_data)
    
    # Save to database
    save_clusters_to_db(clusters, detections_data)
    
    # Get clusters from database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, cluster_name, center_latitude, center_longitude, 
               pothole_count, total_severity_score, radius_meters
        FROM pothole_clusters
        ORDER BY total_severity_score DESC
    ''')
    
    cluster_list = []
    for cluster in cursor.fetchall():
        cluster_id = cluster[0]
        
        # Get route for this cluster
        route = generate_optimal_route(cluster_id)
        
        cluster_list.append({
            'id': cluster_id,
            'name': cluster[1],
            'center_lat': cluster[2],
            'center_lon': cluster[3],
            'pothole_count': cluster[4],
            'severity_score': cluster[5],
            'radius': cluster[6],
            'route': route
        })
    
    conn.close()
    
    return render_template('clusters.html', clusters=cluster_list)

@app.route('/update_status/<int:detection_id>', methods=['POST'])
def update_status(detection_id):
    """Update detection status (for repair crews)"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    new_status = request.json.get('status')
    notes = request.json.get('notes', '')
    
    if new_status not in ['PENDING', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED']:
        return jsonify({'error': 'Invalid status'}), 400
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Update detection
    cursor.execute('UPDATE detections SET status = ? WHERE id = ?', (new_status, detection_id))
    
    # Add to repair history
    cursor.execute('''
        INSERT INTO repair_history (detection_id, status, updated_by, notes)
        VALUES (?, ?, ?, ?)
    ''', (detection_id, new_status, session['user'], notes))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': f'Status updated to {new_status}'})

@app.route('/escalate_check')
def escalate_check():
    """Manual trigger for escalation check (admin only)"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    check_and_escalate_detections()
    flash('Escalation check completed.', 'success')
    return redirect(url_for('dashboard'))

# ========================
# Run Flask App
# ========================
if __name__ == '__main__':
    app.run(debug=True, threaded=True)