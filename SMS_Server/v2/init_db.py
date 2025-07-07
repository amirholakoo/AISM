"""
Database initialization script
"""

from sms_server import app, db

def init_db():
    """Initialize the database"""
    with app.app_context():
        # Create all tables
        db.create_all()
        print("Database initialized successfully!")

if __name__ == "__main__":
    init_db() 