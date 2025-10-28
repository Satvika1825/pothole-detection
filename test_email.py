from dotenv import load_dotenv
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

load_dotenv()

def test_email_connection():
    """Test if email credentials work"""
    
    sender = os.getenv("NOTIFY_SENDER_EMAIL")
    app_password = os.getenv("NOTIFY_APP_PASSWORD")
    recipient = os.getenv("NOTIFY_RECIPIENT")
    
    print("=" * 60)
    print("TESTING EMAIL CONFIGURATION")
    print("=" * 60)
    
    # Check if credentials exist
    print(f"\n✓ Sender Email: {sender}")
    print(f"✓ Recipient Email: {recipient}")
    print(f"✓ App Password: {'*' * len(app_password) if app_password else 'MISSING'}")
    
    if not sender or not app_password or not recipient:
        print("\n❌ ERROR: Email credentials missing in .env file!")
        return False
    
    print("\n📧 Attempting to send test email...")
    
    try:
        # Create test message
        msg = MIMEMultipart()
        msg["Subject"] = "🧪 TEST: Pothole Detection System"
        msg["From"] = sender
        msg["To"] = recipient
        
        body = """
        TEST EMAIL FROM POTHOLE DETECTION SYSTEM
        =========================================
        
        If you receive this email, your email configuration is working correctly!
        
        This is a test message to verify that the pothole detection system
        can successfully send alerts to authorities.
        
        Configuration Details:
        - Sender: {}
        - Recipient: {}
        - SMTP Server: smtp.gmail.com:587
        
        Next Step: Upload an image with potholes to test automatic alerts.
        
        ---
        Automated Pothole Detection System
        """.format(sender, recipient)
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Try to add a sample image if exists
        sample_image_path = 'static/results/detected'
        if os.path.exists(sample_image_path):
            images = [f for f in os.listdir(sample_image_path) if f.endswith(('.jpg', '.png'))]
            if images:
                img_path = os.path.join(sample_image_path, images[0])
                print(f"📎 Attaching sample image: {images[0]}")
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data, name=images[0])
                    msg.attach(image)
        
        # Connect and send
        print("🔌 Connecting to Gmail SMTP server...")
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            print("🔐 Starting TLS encryption...")
            server.starttls()
            
            print("🔑 Logging in...")
            server.login(sender, app_password)
            
            print("📤 Sending email...")
            server.send_message(msg)
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Test email sent successfully!")
        print("=" * 60)
        print(f"\n📬 Check your inbox at: {recipient}")
        print("   (Check spam folder if not in inbox)\n")
        
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print("\n" + "=" * 60)
        print("❌ AUTHENTICATION FAILED!")
        print("=" * 60)
        print("\nPossible issues:")
        print("1. App Password is incorrect")
        print("2. 2-Step Verification not enabled on Gmail")
        print("3. App Password not generated correctly")
        print("\nHow to fix:")
        print("1. Go to: https://myaccount.google.com/apppasswords")
        print("2. Generate new App Password for 'Mail'")
        print("3. Update NOTIFY_APP_PASSWORD in .env file")
        print(f"\nError details: {e}\n")
        return False
        
    except smtplib.SMTPException as e:
        print("\n" + "=" * 60)
        print("❌ SMTP ERROR!")
        print("=" * 60)
        print(f"\nError: {e}\n")
        return False
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ UNEXPECTED ERROR!")
        print("=" * 60)
        print(f"\nError: {e}\n")
        return False

if __name__ == "__main__":
    test_email_connection()