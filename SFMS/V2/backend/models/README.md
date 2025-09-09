# SQLAlchemy Models for SFMS Database

This directory contains SQLAlchemy models for the SFMS (Supply and Factory Management System) database.

## Files

- `external_db.py` - All SQLAlchemy model definitions for external SFMS database
- `db.py` - Flask-SQLAlchemy models for main application
- `database.py` - Database connection and session management for external database
- `README.md` - This file

## Models Overview

### Core Business Models
- `Supplier` - تامین‌کنندگان
- `Customer` - مشتریان
- `MaterialType` - انواع مواد
- `RawMaterial` - مواد خام
- `Unit` - واحدها
- `Truck` - کامیون‌ها
- `Shipments` - محموله‌ها
- `Purchases` - خریدها
- `Sales` - فروش‌ها
- `Products` - محصولات
- `Consumption` - مصرف
- `ConsumptionProfile` - پروفایل مصرف
- `Alert` - هشدارها

### Warehouse Models
- `AnbarSangin` - انبار سنگین
- `AnbarSalonTolid` - انبار سالن تولید
- `AnbarParvandeh` - انبار پرونده
- `AnbarMuhvatehKardan` - انبار محوله کردن
- `AnbarKoochak` - انبار کوچک
- `AnbarKhamirKordan` - انبار خمیر کردن
- `AnbarKhamirGhadim` - انبار خمیر قدیم
- `AnbarAkhal` - انبار آخال
- `AnbarMuhavatehHomayoun` - انبار محوله همایون
- `AnbarPAK` - انبار PAK

### Financial Models
- `Invoice` - فاکتورها
- `Havaleh` - حواله‌ها
- `HavalehItem` - آیتم‌های حواله
- `WeightAdjustmentLog` - لاگ تنظیم وزن

## Usage

### Basic Setup

```python
# Method 1: Import from __init__.py (recommended)
from models import Supplier, Customer, Shipments, Loading, LoadingItem, Warehouse

# Method 2: Direct imports
from models.database import get_db, get_suppliers, get_customers
from models.external_db import Supplier, Customer

# Get database session
db = next(get_db())

# Use helper functions
suppliers = get_suppliers(db, limit=10)
customers = get_customers(db, limit=10)

# Or use direct queries
all_suppliers = db.query(Supplier).all()
active_suppliers = db.query(Supplier).filter(Supplier.status == 'Active').all()

# For main application models (Flask-SQLAlchemy)
from models.db import db, Loading, LoadingItem, Warehouse

db.close()
```

### Example Queries

```python
# Import all models from __init__.py
from models import (
    Supplier, Customer, Shipments, Sales, 
    Products as ExternalProducts,  # Note: renamed to avoid conflict
    Loading, LoadingItem, Warehouse
)

# Get suppliers by status
active_suppliers = db.query(Supplier).filter(Supplier.status == 'Active').all()

# Get shipments by date range
from datetime import datetime, timedelta
start_date = datetime.now() - timedelta(days=30)
recent_shipments = db.query(Shipments).filter(Shipments.date >= start_date).all()

# Get external products by location
products_in_location = db.query(ExternalProducts).filter(ExternalProducts.location == 'شرکت سالار ایرانیان').all()

# Get sales with customer info
sales_with_customers = db.query(Sales).filter(Sales.customer_name.isnot(None)).all()

# Count total items in warehouse
total_products = db.query(ExternalProducts).count()
total_shipments = db.query(Shipments).count()

# Main application models
loadings = Loading.query.all()
warehouses = Warehouse.query.all()
```

### Warehouse Operations

```python
# Get warehouse names and information
from models.database import (
    get_warehouse_names, 
    get_warehouse_table_names,
    get_warehouse_names_from_database,
    get_warehouse_info
)

# Get warehouse names (without 'Anbar_' prefix)
warehouse_names = get_warehouse_names()
# Returns: ['Sangin', 'Salon_Tolid', 'Parvandeh', 'Muhvateh_Kardan', ...]

# Get warehouse table names (with 'Anbar_' prefix)
table_names = get_warehouse_table_names()
# Returns: ['Anbar_Sangin', 'Anbar_Salon_Tolid', 'Anbar_Parvandeh', ...]

# Get warehouse names dynamically from database
db_names = get_warehouse_names_from_database()

# Get detailed warehouse information
warehouse_info = get_warehouse_info()
# Returns list of dictionaries with name, table_name, persian_name, description

# Get items from specific warehouse
from models.database import get_anbar_sangin, get_anbar_salon_tolid

sangin_items = get_anbar_sangin(db, limit=50)
tolid_items = get_anbar_salon_tolid(db, limit=50)

# Get items by status
in_stock_items = db.query(AnbarSangin).filter(AnbarSangin.status == 'In-stock').all()
used_items = db.query(AnbarSangin).filter(AnbarSangin.status == 'Used').all()
```

### Creating New Records

```python
# Create new supplier
new_supplier = Supplier(
    supplier_name="New Supplier",
    address="Tehran, Iran",
    phone="09123456789",
    status="Active",
    comments="Test supplier",
    username="admin",
    logs="Created by admin"
)
db.add(new_supplier)
db.commit()

# Create new shipment
new_shipment = Shipments(
    license_number="12ب365ایران11",
    supplier_name="آقای زعیم",
    material_type="آخال قهوه ای زعیم",
    material_name="آخال قهوه ای زعیم",
    status="Pending",
    username="admin",
    logs="Created by admin"
)
db.add(new_shipment)
db.commit()
```

### Updating Records

```python
# Update supplier status
supplier = db.query(Supplier).filter(Supplier.id == 1).first()
if supplier:
    supplier.status = "Inactive"
    supplier.logs += ", Updated by admin"
    db.commit()

# Update shipment weight
shipment = db.query(Shipments).filter(Shipments.id == 1).first()
if shipment:
    shipment.weight1 = 25000
    shipment.logs += ", Weight updated by admin"
    db.commit()
```

### Deleting Records

```python
# Delete supplier (soft delete by updating status)
supplier = db.query(Supplier).filter(Supplier.id == 1).first()
if supplier:
    supplier.status = "Deleted"
    supplier.logs += ", Deleted by admin"
    db.commit()

# Hard delete (be careful!)
# db.delete(supplier)
# db.commit()
```

## Database Configuration

The database connection is configured in `config.py`:

```python
EXTERNAL_DATABASE_URL = 'sqlite:///external_db/localnew.sqlite3'
```

To use a different database, update this URL in the config file.

## Testing

Run the test script to verify everything works:

```bash
python test_models.py
```

## Dependencies

Install required packages:

```bash
pip install -r requirements_sqlalchemy.txt
```

## Notes

- All models include audit fields like `username` and `logs` for tracking changes
- Most models have a `status` field for soft deletion and state management
- Date fields use `datetime.now` as default
- The database uses SQLite with proper configuration for multi-threading
- All warehouse models follow the same structure for consistency 