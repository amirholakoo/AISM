from flask_sqlalchemy import SQLAlchemy  # type: ignore

db = SQLAlchemy()

# Many-to-many relationship table between warehouses and vision servers
warehouse_vision_server = db.Table('warehouse_vision_server',
    db.Column('warehouse_id', db.String(100), db.ForeignKey('warehouses.id', ondelete='CASCADE'), primary_key=True),
    db.Column('vision_server_id', db.Integer, db.ForeignKey('vision_servers.id', ondelete='CASCADE'), primary_key=True)
)

class Warehouse(db.Model):
    __tablename__ = 'warehouses'
    id = db.Column(db.String(100), primary_key=True)  # نام جدول در external DB به عنوان primary key
    name = db.Column(db.String(100), nullable=False, unique=True)  # نام انگلیسی از external DB
    persian_name = db.Column(db.String(100), nullable=True)  # نام فارسی
    is_active = db.Column(db.Boolean, default=True)  # فعال/غیرفعال
    unloadings = db.relationship('Unloading', backref='warehouse', lazy=True)
    loadings = db.relationship('Loading', backref='warehouse', lazy=True)
    vision_servers = db.relationship('VisionServer', secondary=warehouse_vision_server, backref='warehouses', cascade='all, delete')

class Unloading(db.Model):
    __tablename__ = 'unloadings'
    id = db.Column(db.Integer, primary_key=True)
    warehouse_id = db.Column(db.String(100), db.ForeignKey('warehouses.id'), nullable=False)
    shipment_id = db.Column(db.Integer, nullable=True)  # شناسه محموله انتخاب شده
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=True)
    user_confirm_time = db.Column(db.DateTime, nullable=True)
    edit_time = db.Column(db.DateTime, nullable=True)
    version = db.Column(db.Integer, nullable=False, default=1)
    status = db.Column(db.String(32), nullable=False, default='started')
    vision_output = db.Column(db.Text, nullable=True)  # خروجی سیستم بینایی
    items = db.relationship('UnloadingItem', backref='unloading', cascade='all, delete-orphan', lazy=True)

class UnloadingItem(db.Model):
    __tablename__ = 'unloading_items'
    id = db.Column(db.Integer, primary_key=True)
    unloading_id = db.Column(db.Integer, db.ForeignKey('unloadings.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(16), nullable=False)
    count = db.Column(db.Integer, nullable=False) 
    source = db.Column(db.String(16), nullable=False, default='user')  # 'vision' or 'user'
    version = db.Column(db.Integer, nullable=False, default=1) 

class Loading(db.Model):
    __tablename__ = 'loadings'
    id = db.Column(db.Integer, primary_key=True)
    warehouse_id = db.Column(db.String(100), db.ForeignKey('warehouses.id'), nullable=False)
    shipment_id = db.Column(db.Integer, nullable=True)  # شناسه محموله انتخاب شده
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=True)
    user_confirm_time = db.Column(db.DateTime, nullable=True)
    edit_time = db.Column(db.DateTime, nullable=True)
    version = db.Column(db.Integer, nullable=False, default=1)
    status = db.Column(db.String(32), nullable=False, default='started')
    vision_output = db.Column(db.Text, nullable=True)  # خروجی سیستم بینایی
    items = db.relationship('LoadingItem', backref='loading', cascade='all, delete-orphan', lazy=True)

class LoadingItem(db.Model):
    __tablename__ = 'loading_items'
    id = db.Column(db.Integer, primary_key=True)
    loading_id = db.Column(db.Integer, db.ForeignKey('loadings.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(16), nullable=False)
    count = db.Column(db.Integer, nullable=False) 
    source = db.Column(db.String(16), nullable=False, default='user')  # 'vision' or 'user'
    version = db.Column(db.Integer, nullable=False, default=1)
    
    # New fields for loading items based on vision output
    reel_number = db.Column(db.String(50), nullable=True)  # شماره رول
    width = db.Column(db.Integer, nullable=True)  # عرض (mm)
    gsm = db.Column(db.Integer, nullable=True)  # گرماژ (gsm)
    length = db.Column(db.Integer, nullable=True)  # طول (m)
    breaks = db.Column(db.Integer, nullable=True)  # تعداد شکستگی‌ها
    grade = db.Column(db.String(100), nullable=True)  # درجه/کیفیت

class Product(db.Model):
    __tablename__ = 'products'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)  # نام انگلیسی
    persian_name = db.Column(db.String(100), nullable=False, unique=True)  # نام فارسی
    vision_name = db.Column(db.String(100), nullable=True)  # نام در سیستم بینایی
    width = db.Column(db.Integer, nullable=True)  # عرض (mm)
    gsm = db.Column(db.Integer, nullable=True)  # گرماژ (gsm)
    length = db.Column(db.Integer, nullable=True)  # طول (m) 

class VisionServer(db.Model):
    __tablename__ = 'vision_servers'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    persian_name = db.Column(db.String(100), nullable=True)  # نام فارسی
    url = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(50), nullable=False)  # نوع عملیات: unloading, loading, consumption, transfer, return
    is_active = db.Column(db.Boolean, default=True)
    video_source = db.Column(db.String(50), nullable=False, default='picamera')  # منبع ویدیو: picamera, webcam, file, etc.
    
    def __repr__(self):
        return f'<VisionServer {self.name}>'

class OperationType(db.Model):
    __tablename__ = 'operation_types'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)  # نام انگلیسی
    persian_name = db.Column(db.String(100), nullable=False)  # نام فارسی
    icon = db.Column(db.String(100), nullable=True)  # نام آیکون (مثل Truck, Upload, etc.)
    color = db.Column(db.String(50), nullable=True)  # رنگ آیکون (مثل red, blue, etc.)
    is_enabled = db.Column(db.Boolean, default=True)  # فعال/غیرفعال
    is_available = db.Column(db.Boolean, default=True)  # در دسترس/غیرفعال
    description = db.Column(db.String(255), nullable=True)  # توضیحات
    order = db.Column(db.Integer, default=0)  # ترتیب نمایش

 