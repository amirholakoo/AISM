from flask import Flask
from models.db import db, Unloading, UnloadingItem, Product, Warehouse, VisionServer, OperationType
import os

app = Flask(__name__)

# Use absolute path for database
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db.sqlite3')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

print(f"Database path: {db_path}")
print(f"Database exists: {os.path.exists(db_path)}")

db.init_app(app)

with app.app_context():
    print("Creating all tables...")
    
    # Check metadata before creation
    print(f"Tables in metadata before creation: {list(db.metadata.tables.keys())}")
    
    try:
        db.create_all()
        print("✅ All tables created successfully!")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()
    
    # Check metadata after creation
    print(f"Tables in metadata after creation: {list(db.metadata.tables.keys())}")
    
    # Check if database file was created
    print(f"Database exists after creation: {os.path.exists(db_path)}")
    if os.path.exists(db_path):
        print(f"Database size: {os.path.getsize(db_path)} bytes")
    
    # Check if loadings table exists
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print("\nTables in database:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Check important tables specifically
        important_tables = ['loadings', 'vision_servers', 'warehouse_vision_server', 'operation_types']
        for table_name in important_tables:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            table_exists = cursor.fetchone()
            if table_exists:
                print(f"✅ {table_name} table exists")
            else:
                print(f"❌ {table_name} table does not exist")
        
        conn.close()
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc() 