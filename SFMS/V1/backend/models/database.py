from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from .external_db import Base
import os

from config import EXTERNAL_DATABASE_URL


# Create engine with SQLite configuration
engine = create_engine(
    EXTERNAL_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False  # Set to True for SQL query logging
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Session:
    """
    Get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def close_database_connection():
    """
    Close all database connections and dispose the engine
    This should be called when you want to completely close the database
    to allow file replacement via SSH
    """
    try:
        # Close all sessions
        SessionLocal.close_all()
        
        # Dispose the engine to close all connections
        engine.dispose()
        
        print("✅ Database connections closed successfully")
        return True
    except Exception as e:
        print(f"❌ Error closing database connections: {e}")
        return False

def dispose_engine():
    """
    Dispose the engine to close all connections
    """
    try:
        engine.dispose()
        print("✅ Engine disposed successfully")
        return True
    except Exception as e:
        print(f"❌ Error disposing engine: {e}")
        return False

def get_db_with_manual_close() -> Session:
    """
    Get database session that requires manual closing
    Use this when you want to control when the session is closed
    """
    return SessionLocal()

def close_session(session: Session):
    """
    Manually close a database session
    """
    try:
        if session:
            session.close()
            print("✅ Session closed successfully")
    except Exception as e:
        print(f"❌ Error closing session: {e}")

def create_tables():
    """
    Create all tables in the database
    """
    Base.metadata.create_all(bind=engine)

def drop_tables():
    """
    Drop all tables from the database
    """
    Base.metadata.drop_all(bind=engine)

# Example usage functions
def get_suppliers(db: Session, skip: int = 0, limit: int = 100):
    """
    Get suppliers with pagination
    """
    from .external_db import Supplier
    return db.query(Supplier).offset(skip).limit(limit).all()

def get_customers(db: Session, skip: int = 0, limit: int = 100):
    """
    Get customers with pagination
    """
    from .external_db import Customer
    return db.query(Customer).offset(skip).limit(limit).all()

def get_shipments(db: Session, skip: int = 0, limit: int = 100):
    """
    Get shipments with pagination
    """
    from .external_db import Shipments
    return db.query(Shipments).offset(skip).limit(limit).all()

def get_latest_loaded_unloaded_shipments(db: Session, limit: int = 10):
    """
    Get latest shipments with Loaded or Unloaded status ordered by date (newest first)
    """
    from .external_db import Shipments
    return db.query(Shipments).filter(
        Shipments.status.in_(['Loaded', 'Unloaded'])
    ).order_by(Shipments.date.desc()).limit(limit).all()

def get_shipments_for_unloading(db: Session, limit: int = 50):
    """
    Get shipments for unloading operation (Incoming shipments with LoadingUnloading status)
    """
    from .external_db import Shipments
    return db.query(Shipments).filter(
        Shipments.shipment_type == 'Incoming',
        Shipments.status == 'LoadingUnloading'
    ).order_by(Shipments.date.desc()).limit(limit).all()

def get_shipments_for_loading(db: Session, limit: int = 50):
    """
    Get shipments for loading operation (Outgoing shipments with LoadingUnloading status)
    """
    from .external_db import Shipments
    return db.query(Shipments).filter(
        Shipments.shipment_type == 'Outgoing',
        Shipments.status == 'LoadingUnloading'
    ).order_by(Shipments.date.desc()).limit(limit).all()

def get_shipments_by_date_range(db: Session, start_date=None, end_date=None, limit: int = 100):
    """
    Get shipments within a date range
    """
    from .external_db import Shipments
    from datetime import datetime
    
    query = db.query(Shipments)
    
    if start_date:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        query = query.filter(Shipments.date >= start_date)
    
    if end_date:
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        query = query.filter(Shipments.date <= end_date)
    
    return query.order_by(Shipments.date.desc()).limit(limit).all()

def get_shipments_count(db: Session):
    """
    Get total count of shipments
    """
    from .external_db import Shipments
    return db.query(Shipments).count()

def get_products(db: Session, skip: int = 0, limit: int = 100):
    """
    Get products with pagination
    """
    from .external_db import Products
    return db.query(Products).offset(skip).limit(limit).all()

def get_sales(db: Session, skip: int = 0, limit: int = 100):
    """
    Get sales with pagination
    """
    from .external_db import Sales
    return db.query(Sales).offset(skip).limit(limit).all()

def get_purchases(db: Session, skip: int = 0, limit: int = 100):
    """
    Get purchases with pagination
    """
    from .external_db import Purchases
    return db.query(Purchases).offset(skip).limit(limit).all()

def get_alerts(db: Session, skip: int = 0, limit: int = 100):
    """
    Get alerts with pagination
    """
    from .external_db import Alert
    return db.query(Alert).offset(skip).limit(limit).all()

def get_trucks(db: Session, skip: int = 0, limit: int = 100):
    """
    Get trucks with pagination
    """
    from .external_db import Truck
    return db.query(Truck).offset(skip).limit(limit).all()

# Warehouse functions
def get_anbar_sangin(db: Session, skip: int = 0, limit: int = 100):
    """
    Get Anbar Sangin items with pagination
    """
    from .external_db import AnbarSangin
    return db.query(AnbarSangin).offset(skip).limit(limit).all()

def get_anbar_salon_tolid(db: Session, skip: int = 0, limit: int = 100):
    """
    Get Anbar Salon Tolid items with pagination
    """
    from .external_db import AnbarSalonTolid
    return db.query(AnbarSalonTolid).offset(skip).limit(limit).all()

def get_anbar_muhvateh_kardan(db: Session, skip: int = 0, limit: int = 100):
    """
    Get Anbar Muhvateh Kardan items with pagination
    """
    from .external_db import AnbarMuhvatehKardan
    return db.query(AnbarMuhvatehKardan).offset(skip).limit(limit).all()

def get_anbar_koochak(db: Session, skip: int = 0, limit: int = 100):
    """
    Get Anbar Koochak items with pagination
    """
    from .external_db import AnbarKoochak
    return db.query(AnbarKoochak).offset(skip).limit(limit).all()

def get_anbar_khamir_kordan(db: Session, skip: int = 0, limit: int = 100):
    """
    Get Anbar Khamir Kordan items with pagination
    """
    from .external_db import AnbarKhamirKordan
    return db.query(AnbarKhamirKordan).offset(skip).limit(limit).all()

def get_anbar_khamir_ghadim(db: Session, skip: int = 0, limit: int = 100):
    """
    Get Anbar Khamir Ghadim items with pagination
    """
    from .external_db import AnbarKhamirGhadim
    return db.query(AnbarKhamirGhadim).offset(skip).limit(limit).all()

def get_anbar_akhal(db: Session, skip: int = 0, limit: int = 100):
    """
    Get Anbar Akhal items with pagination
    """
    from .external_db import AnbarAkhal
    return db.query(AnbarAkhal).offset(skip).limit(limit).all()

def get_anbar_muhavateh_homayoun(db: Session, skip: int = 0, limit: int = 100):
    """
    Get Anbar Muhavateh Homayoun items with pagination
    """
    from .external_db import AnbarMuhavatehHomayoun
    return db.query(AnbarMuhavatehHomayoun).offset(skip).limit(limit).all()

def get_anbar_pak(db: Session, skip: int = 0, limit: int = 100):
    """
    Get Anbar PAK items with pagination
    """
    from .external_db import AnbarPAK
    return db.query(AnbarPAK).offset(skip).limit(limit).all()

def get_anbar_parvandeh(db: Session, skip: int = 0, limit: int = 100):
    """
    Get Anbar Parvandeh items with pagination
    """
    from .external_db import AnbarParvandeh
    return db.query(AnbarParvandeh).offset(skip).limit(limit).all()

def get_warehouse_names():
    """
    Get list of warehouse names from external database
    Returns warehouse names without 'Anbar_' prefix
    """
    warehouse_names = [
        'Sangin',           # انبار سنگین
        'Salon_Tolid',      # انبار سالن تولید
        'Parvandeh',        # انبار پرونده
        'Muhvateh_Kardan',  # انبار محوله کردن
        'Koochak',          # انبار کوچک
        'Khamir_Kordan',    # انبار خمیر کردن
        'Khamir_Ghadim',    # انبار خمیر قدیم
        'Akhal',            # انبار آخال
        'Muhavateh_Homayoun', # انبار محوله همایون
        'PAK'               # انبار PAK
    ]
    return warehouse_names

def get_warehouse_table_names():
    """
    Get list of warehouse table names from external database
    Returns full table names with 'Anbar_' prefix
    """
    warehouse_table_names = [
        'Anbar_Sangin',
        'Anbar_Salon_Tolid', 
        'Anbar_Parvandeh',
        'Anbar_Muhvateh_Kardan',
        'Anbar_Koochak',
        'Anbar_Khamir_Kordan',
        'Anbar_Khamir_Ghadim',
        'Anbar_Akhal',
        'Anbar_Muhavateh_Homayoun',
        'Anbar_PAK'
    ]
    return warehouse_table_names

def get_warehouse_names_from_database():
    """
    Get warehouse names dynamically from database using SQLAlchemy
    Returns warehouse names without 'Anbar_' prefix
    """
    try:
        db_session = next(get_db())
        
        # Get all table names using SQLAlchemy inspect
        from sqlalchemy import inspect
        inspector = inspect(engine)
        all_tables = inspector.get_table_names()
        
        # Filter tables that start with 'Anbar_'
        warehouse_tables = [table for table in all_tables if table.startswith('Anbar_')]
        warehouse_tables.sort()  # Sort alphabetically
        
        # Remove 'Anbar_' prefix from table names
        warehouse_names = [table.replace('Anbar_', '') for table in warehouse_tables]
        
        db_session.close()
        return warehouse_names
        
    except Exception as e:
        print(f"Error getting warehouse names from database: {e}")
        # Fallback to static list
        return get_warehouse_names()

def get_warehouse_tables_from_database():
    """
    Get warehouse table names dynamically from database using SQLAlchemy
    Returns full table names with 'Anbar_' prefix
    """
    try:
        db_session = next(get_db())
        
        # Get all table names using SQLAlchemy inspect
        from sqlalchemy import inspect
        inspector = inspect(engine)
        all_tables = inspector.get_table_names()
        
        # Filter tables that start with 'Anbar_'
        warehouse_tables = [table for table in all_tables if table.startswith('Anbar_')]
        warehouse_tables.sort()  # Sort alphabetically
        
        db_session.close()
        return warehouse_tables
        
    except Exception as e:
        print(f"Error getting warehouse tables from database: {e}")
        # Fallback to static list
        return get_warehouse_table_names()

def get_all_tables_from_database():
    """
    Get all table names from database using SQLAlchemy
    """
    try:
        db_session = next(get_db())
        
        # Get all table names using SQLAlchemy inspect
        from sqlalchemy import inspect
        inspector = inspect(engine)
        all_tables = inspector.get_table_names()
        
        db_session.close()
        return all_tables
        
    except Exception as e:
        print(f"Error getting all tables from database: {e}")
        return []

def get_warehouse_info():
    """
    Get detailed warehouse information
    Returns list of dictionaries with warehouse details
    """
    warehouse_info = [
        {
            'name': 'Sangin',
            'table_name': 'Anbar_Sangin',
            'persian_name': 'انبار سنگین',
            'description': 'انبار برای مواد سنگین'
        },
        {
            'name': 'Salon_Tolid',
            'table_name': 'Anbar_Salon_Tolid', 
            'persian_name': 'انبار سالن تولید',
            'description': 'انبار در سالن تولید'
        },
        {
            'name': 'Parvandeh',
            'table_name': 'Anbar_Parvandeh',
            'persian_name': 'انبار پرونده',
            'description': 'انبار برای پرونده‌ها'
        },
        {
            'name': 'Muhvateh_Kardan',
            'table_name': 'Anbar_Muhvateh_Kardan',
            'persian_name': 'انبار محوله کردن',
            'description': 'انبار برای محوله کردن'
        },
        {
            'name': 'Koochak',
            'table_name': 'Anbar_Koochak',
            'persian_name': 'انبار کوچک',
            'description': 'انبار کوچک'
        },
        {
            'name': 'Khamir_Kordan',
            'table_name': 'Anbar_Khamir_Kordan',
            'persian_name': 'انبار خمیر کردن',
            'description': 'انبار برای خمیر کردن'
        },
        {
            'name': 'Khamir_Ghadim',
            'table_name': 'Anbar_Khamir_Ghadim',
            'persian_name': 'انبار خمیر قدیم',
            'description': 'انبار خمیر قدیم'
        },
        {
            'name': 'Akhal',
            'table_name': 'Anbar_Akhal',
            'persian_name': 'انبار آخال',
            'description': 'انبار برای آخال'
        },
        {
            'name': 'Muhavateh_Homayoun',
            'table_name': 'Anbar_Muhavateh_Homayoun',
            'persian_name': 'انبار محوله همایون',
            'description': 'انبار محوله همایون'
        },
        {
            'name': 'PAK',
            'table_name': 'Anbar_PAK',
            'persian_name': 'انبار PAK',
            'description': 'انبار PAK'
        }
    ]
    return warehouse_info 