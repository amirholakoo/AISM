"""
Database models for SMS verification system
"""
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
import random
import string

db = SQLAlchemy()

class SMSVerification(db.Model):
    """SMS verification code model"""
    id = db.Column(db.Integer, primary_key=True)
    phone_number = db.Column(db.String(20), nullable=False, index=True)
    code = db.Column(db.String(6), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    is_used = db.Column(db.Boolean, default=False)
    attempts = db.Column(db.Integer, default=0)
    
    @classmethod
    def generate_code(cls, phone_number, expiry_minutes=10):
        """Generate a new verification code"""
        code = ''.join(random.choices(string.digits, k=6))
        expires_at = datetime.utcnow() + timedelta(minutes=expiry_minutes)
        
        verification = cls(
            phone_number=phone_number,
            code=code,
            expires_at=expires_at
        )
        db.session.add(verification)
        db.session.commit()
        return verification
    
    def is_valid(self):
        """Check if verification code is still valid"""
        return (
            not self.is_used and 
            datetime.utcnow() <= self.expires_at and
            self.attempts < 3
        )
    
    def increment_attempts(self):
        """Increment failed attempts counter"""
        self.attempts += 1
        db.session.commit()
    
    def mark_as_used(self):
        """Mark verification code as used"""
        self.is_used = True
        db.session.commit()
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'phone_number': self.phone_number,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'is_used': self.is_used,
            'attempts': self.attempts
        }

class SMSLog(db.Model):
    """SMS sending log model"""
    id = db.Column(db.Integer, primary_key=True)
    phone_number = db.Column(db.String(20), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), nullable=False)  # success, failed
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    @classmethod
    def log_sms(cls, phone_number, message, status, error_message=None):
        """Create SMS log entry"""
        log = cls(
            phone_number=phone_number,
            message=message,
            status=status,
            error_message=error_message
        )
        db.session.add(log)
        db.session.commit()
        return log 