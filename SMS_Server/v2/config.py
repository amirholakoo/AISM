"""
Configuration settings for SMS Server
"""
import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'ioms_secret_key_2025')
    API_KEY = os.getenv('API_KEY', 'ioms_sms_server_2025')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    SIM800C_PORT = os.getenv('SIM800C_PORT', '/dev/ttyAMA0')
    SIM800C_BAUDRATE = int(os.getenv('SIM800C_BAUDRATE', '115200'))
    MAX_VERIFICATION_ATTEMPTS = 3
    VERIFICATION_CODE_EXPIRY = 10  # minutes

class DevelopmentConfig(Config):
    """Development configuration"""
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///sms_server.db')
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    SQLALCHEMY_DATABASE_URI = 'sqlite:///test.db'
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    """Production configuration"""
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///sms_server.db')
    DEBUG = False
    TESTING = False

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': ProductionConfig  # Changed default to production
} 