from flask import Flask
from models import db, Unloading, UnloadingItem, Product, Warehouse, OperationType, VisionServer
from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from datetime import datetime, timedelta
from models.database import get_warehouse_tables_from_database
import sqlite3
from pathlib import Path

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
db.init_app(app)

def add_shipment_id_column():
    """Add shipment_id column to loadings table if it doesn't exist"""
    try:
        # Get database path from SQLAlchemy URI
        db_path = SQLALCHEMY_DATABASE_URI.replace('sqlite:///', '')
        if db_path.startswith('/'):
            db_path = db_path[1:]  # Remove leading slash on Windows
        # Handle Windows path with backslashes
        if '\\' in db_path:
            db_path = db_path.replace('\\', '/')
        
        print(f"Checking database at: {db_path}")
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if loadings table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='loadings'")
        if not cursor.fetchone():
            print("⚠️  loadings table doesn't exist yet, will be created by SQLAlchemy")
            return True
        
        # Check if shipment_id column already exists
        cursor.execute("PRAGMA table_info(loadings)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'shipment_id' in columns:
            print("✅ shipment_id column already exists in loadings table")
            return True
        
        # Add shipment_id column
        print("Adding shipment_id column to loadings table...")
        cursor.execute("ALTER TABLE loadings ADD COLUMN shipment_id INTEGER")
        
        # Commit changes
        conn.commit()
        print("✅ Successfully added shipment_id column to loadings table")
        
        return True
        
    except Exception as e:
        print(f"❌ Error adding shipment_id column: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def seed_database():
    with app.app_context():
        # Create tables if they don't exist
        print("Creating tables if they don't exist...")
        db.create_all()
        
        # Clear loadings and loading_items tables
        print("Clearing loadings and loading_items tables...")
        try:
            # Delete all loading items first (due to foreign key constraint)
            UnloadingItem.query.delete()
            print("  - Cleared loading_items table")
            
            # Delete all loadings
            Unloading.query.delete()
            print("  - Cleared loadings table")
            
            # Commit the changes
            db.session.commit()
            print("✅ Successfully cleared loadings and loading_items tables!")
            
        except Exception as e:
            print(f"❌ Error clearing tables: {e}")
            db.session.rollback()
        
        # Add shipment_id column to loadings table
        print("Checking for shipment_id column...")
        add_shipment_id_column()
        
        # Sync warehouses from external database
        print("Syncing warehouses from external database...")
        try:
            external_warehouses = get_warehouse_tables_from_database()
            print(f"Found {len(external_warehouses)} warehouses in external database:")
            
            added_count = 0
            for external_name in external_warehouses:
                warehouse_name = external_name.replace('Anbar_', '')
                
                # بررسی اینکه آیا انبار قبلاً وجود داره (بر اساس id و name)
                existing_warehouse_by_id = Warehouse.query.filter_by(id=external_name).first()
                existing_warehouse_by_name = Warehouse.query.filter_by(name=warehouse_name).first()
                
                if existing_warehouse_by_id or existing_warehouse_by_name:
                    print(f"  - Skipped: {warehouse_name} (already exists)")
                    continue
                
                # اضافه کردن انبار جدید
                warehouse = Warehouse(
                    id=external_name,
                    name=warehouse_name,
                    persian_name=f"انبار {warehouse_name}",
                    is_active=True
                )
                db.session.add(warehouse)
                added_count += 1
                print(f"  - Added: {warehouse_name} (from {external_name})")
            
            db.session.commit()
            print(f"✅ Successfully synced {added_count} new warehouses!")
            
        except Exception as e:
            print(f"❌ Error syncing warehouses: {e}")
            db.session.rollback()  # Rollback the session on error
            print("Creating default warehouses...")
            
            # Fallback to default warehouses
            default_warehouses = [
                {"name": "انبار ۱", "id": "Anbar_1", "persian_name": "انبار ۱"},
                {"name": "انبار ۲", "id": "Anbar_2", "persian_name": "انبار ۲"},
                {"name": "انبار ۳", "id": "Anbar_3", "persian_name": "انبار ۳"},
                {"name": "انبار ۴", "id": "Anbar_4", "persian_name": "انبار ۴"}
            ]
            
            added_count = 0
            for warehouse_data in default_warehouses:
                # بررسی اینکه آیا انبار قبلاً وجود داره (بر اساس id و name)
                existing_warehouse_by_id = Warehouse.query.filter_by(id=warehouse_data["id"]).first()
                existing_warehouse_by_name = Warehouse.query.filter_by(name=warehouse_data["name"]).first()
                
                if existing_warehouse_by_id or existing_warehouse_by_name:
                    print(f"  - Skipped: {warehouse_data['name']} (already exists)")
                    continue
                
                warehouse = Warehouse(
                    id=warehouse_data["id"],
                    name=warehouse_data["name"],
                    persian_name=warehouse_data["persian_name"],
                    is_active=True
                )
                db.session.add(warehouse)
                added_count += 1
                print(f"  - Added: {warehouse_data['name']}")
            
            db.session.commit()
            print(f"✅ Created {added_count} default warehouses!")
        
        # Always ensure test warehouse exists
        print("Ensuring test warehouse exists...")
        test_warehouse = Warehouse.query.filter_by(id="Anbar_Test").first()
        if not test_warehouse:
            test_warehouse = Warehouse(
                id="Anbar_Test",
                name="Test",
                persian_name="تست",
                is_active=True
            )
            db.session.add(test_warehouse)
            db.session.commit()
            print("  - Added: Test warehouse")
        else:
            print("  - Skipped: Test warehouse (already exists)")
        
        # Create operation types
        print("Creating operation types...")
        operation_types_data = [
            {
                "name": "unloading",
                "persian_name": "تخلیه",
                "icon": "Download",
                "color": "red",
                "is_enabled": True,
                "is_available": True,
                "description": "شروع عملیات تخلیه محموله",
                "order": 1
            },
            {
                "name": "loading",
                "persian_name": "بارگیری",
                "icon": "Upload",
                "color": "blue",
                "is_enabled": True,
                "is_available": True,
                "description": "شروع عملیات بارگیری محموله",
                "order": 2
            },
            {
                "name": "consumption",
                "persian_name": "مصرف",
                "icon": "ShoppingCart",
                "color": "gray",
                "is_enabled": True,
                "is_available": False,
                "description": "به زودی در دسترس خواهد بود",
                "order": 3
            },
            {
                "name": "transfer",
                "persian_name": "انتقال",
                "icon": "ArrowRightLeft",
                "color": "gray",
                "is_enabled": True,
                "is_available": False,
                "description": "به زودی در دسترس خواهد بود",
                "order": 4
            },
            {
                "name": "return",
                "persian_name": "مرجوع",
                "icon": "RotateCcw",
                "color": "gray",
                "is_enabled": True,
                "is_available": False,
                "description": "به زودی در دسترس خواهد بود",
                "order": 5
            }
        ]
        
        added_count = 0
        for operation_data in operation_types_data:
            # بررسی اینکه آیا نوع عملیات قبلاً وجود داره
            existing_operation = OperationType.query.filter_by(name=operation_data["name"]).first()
            
            if existing_operation:
                print(f"  - Skipped: {operation_data['persian_name']} (already exists)")
                continue
            
            operation = OperationType(**operation_data)
            db.session.add(operation)
            added_count += 1
            print(f"  - Added: {operation_data['persian_name']} ({operation_data['name']})")
        
        db.session.commit()
        print(f"✅ Created {added_count} operation types!")
        
        # Create vision servers
        print("Creating vision servers...")
        vision_servers_data = [
            {
                "name": "unloading_vision_server",
                "persian_name": "سرور بینایی تخلیه",
                "url": "http://localhost:5001",
                "type": "unloading",
                "is_active": True,
                "video_source": "picamera"
            },
            {
                "name": "loading_vision_server",
                "persian_name": "سرور بینایی بارگیری",
                "url": "http://localhost:5002",
                "type": "loading",
                "is_active": True,
                "video_source": "picamera"
            },
            {
                "name": "both_vision_server",
                "persian_name": "سرور بینایی ترکیبی",
                "url": "http://localhost:5003",
                "type": "both",
                "is_active": True,
                "video_source": "picamera"
            }
        ]
        
        added_count = 0
        for server_data in vision_servers_data:
            # بررسی اینکه آیا سرور بینایی قبلاً وجود داره
            existing_server = VisionServer.query.filter_by(name=server_data["name"]).first()
            
            if existing_server:
                print(f"  - Skipped: {server_data['persian_name']} (already exists)")
                continue
            
            server = VisionServer(**server_data)
            db.session.add(server)
            added_count += 1
            print(f"  - Added: {server_data['persian_name']} ({server_data['name']})")
        
        db.session.commit()
        print(f"✅ Created {added_count} vision servers!")
        
        # Assign vision servers to warehouses by default
        print("Assigning vision servers to warehouses...")
        try:
            # Get the first warehouse (usually "انبار سنگین")
            first_warehouse = Warehouse.query.first()
            if first_warehouse:
                # Get the unloading, loading, and both vision servers
                unloading_server = VisionServer.query.filter_by(type="unloading").first()
                loading_server = VisionServer.query.filter_by(type="loading").first()
                both_server = VisionServer.query.filter_by(type="both").first()
                
                if unloading_server and loading_server and both_server:
                    # Assign all three servers to the first warehouse
                    first_warehouse.vision_servers.append(unloading_server)
                    first_warehouse.vision_servers.append(loading_server)
                    first_warehouse.vision_servers.append(both_server)
                    db.session.commit()
                    print(f"✅ Assigned all vision servers to warehouse: {first_warehouse.persian_name}")
                else:
                    print("⚠️  Some vision servers not found for assignment")
            else:
                print("⚠️  No warehouses found for assignment")
        except Exception as e:
            print(f"⚠️  Error assigning vision servers to warehouses: {e}")
            db.session.rollback()
        
        # Create products
        print("Creating products...")
        products_data = [
            # محصولات پرکاربرد سیستم ویژن
            {"name": "sodium_hydroxide", "persian_name": "هیدروکسید سدیم"},
            {"name": "titanium_dioxide", "persian_name": "دی‌اکسید تیتانیوم"},
            {"name": "calcium_carbonate", "persian_name": "کربنات کلسیم"},
            {"name": "magnesium_oxide", "persian_name": "اکسید منیزیم"},
            {"name": "aluminum_oxide", "persian_name": "اکسید آلومینیوم"},
            {"name": "iron_oxide", "persian_name": "اکسید آهن"},
            {"name": "zinc_oxide", "persian_name": "اکسید روی"},

        ]
        
        added_count = 0
        for product_data in products_data:
            # بررسی اینکه آیا محصول قبلاً وجود داره (بر اساس name و persian_name)
            existing_product_by_name = Product.query.filter_by(name=product_data["name"]).first()
            existing_product_by_persian = Product.query.filter_by(persian_name=product_data["persian_name"]).first()
            
            if existing_product_by_name or existing_product_by_persian:
                print(f"  - Skipped: {product_data['persian_name']} (already exists)")
                continue
            
            product = Product(
                name=product_data["name"],
                persian_name=product_data["persian_name"],
                vision_name=product_data["name"].lower()
            )
            db.session.add(product)
            added_count += 1
            print(f"  - Added: {product_data['persian_name']} ({product_data['name']})")
        
        db.session.commit()
        print(f"✅ Created {added_count} products!")
        
        # Note: Cameras should be added manually through the admin interface
        # No sample cameras are created automatically
        print("Skipping camera creation - cameras should be added manually")
        
        print("Database seeded successfully!")

if __name__ == "__main__":
    seed_database() 