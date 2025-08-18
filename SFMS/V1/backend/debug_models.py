from flask import Flask
from models.db import db

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    print("Checking models...")
    
    # Check if models are accessible
    try:
        from models.db import Warehouse, VisionServer, OperationType, Loading
        print("✅ All models imported successfully")
        
        # Check model metadata
        print(f"Warehouse table name: {Warehouse.__tablename__}")
        print(f"VisionServer table name: {VisionServer.__tablename__}")
        print(f"OperationType table name: {OperationType.__tablename__}")
        print(f"Loading table name: {Loading.__tablename__}")
        
        # Check if tables exist in metadata
        print(f"\nTables in metadata: {list(db.metadata.tables.keys())}")
        
    except Exception as e:
        print(f"❌ Error importing models: {e}")
        import traceback
        traceback.print_exc()
