# SFMS - Smart Factory Management System
# سیستم مدیریت هوشمند کارخانه

[![React](https://img.shields.io/badge/React-19.1.0-blue.svg)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.1-green.svg)](https://flask.palletsprojects.com/)
[![Python](https://img.shields.io/badge/Python-3.13-yellow.svg)](https://www.python.org/)
[![Vite](https://img.shields.io/badge/Vite-7.0.4-purple.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-4.1.11-38B2AC.svg)](https://tailwindcss.com/)

## 📋 فهرست مطالب / Table of Contents

- [معرفی / Overview](#معرفی--overview)
- [ویژگی‌ها / Features](#ویژگی‌ها--features)
- [معماری سیستم / System Architecture](#معماری-سیستم--system-architecture)
- [نصب و راه‌اندازی / Installation](#نصب-و-راه‌اندازی--installation)
- [استفاده / Usage](#استفاده--usage)
- [API Documentation](#api-documentation)
- [ساختار پروژه / Project Structure](#ساختار-پروژه--project-structure)
- [مشارکت / Contributing](#مشارکت--contributing)
- [لایسنس / License](#لایسنس--license)

---

## 🏭 معرفی / Overview

**SFMS (Smart Factory Management System)** یک سیستم جامع مدیریت کارخانه هوشمند است که از تکنولوژی‌های مدرن برای مدیریت عملیات بارگیری، تخلیه، انبارداری و بینایی کامپیوتری استفاده می‌کند.

**SFMS (Smart Factory Management System)** is a comprehensive smart factory management system that utilizes modern technologies for managing loading, unloading, warehousing operations, and computer vision.

### 🎯 اهداف اصلی / Main Objectives

- مدیریت هوشمند عملیات بارگیری و تخلیه
- سیستم بینایی کامپیوتری برای تشخیص محصولات
- مدیریت انبارها و محصولات
- رابط کاربری مدرن و کاربرپسند
- API های RESTful برای یکپارچه‌سازی

---

## ✨ ویژگی‌ها / Features

### 🔄 مدیریت عملیات / Operations Management
- **بارگیری (Loading)**: مدیریت کامل فرآیند بارگیری محصولات
- **تخلیه (Unloading)**: کنترل و نظارت بر عملیات تخلیه
- **انتخاب محموله**: سیستم هوشمند انتخاب محموله‌ها
- **انواع عملیات**: پشتیبانی از انواع مختلف عملیات

### 🏢 مدیریت انبار / Warehouse Management
- **مدیریت انبارها**: CRUD عملیات برای انبارها
- **انتساب سرور**: تخصیص سرورهای بینایی به انبارها
- **مدیریت محصولات**: ثبت و مدیریت محصولات

### 👁️ سیستم بینایی / Vision System
- **سرورهای بینایی**: مدیریت سرورهای پردازش تصویر
- **تشخیص محصولات**: استفاده از AI برای تشخیص محصولات
- **پردازش QR Code**: اسکن و پردازش کدهای QR

### 🖥️ رابط کاربری / User Interface
- **طراحی مدرن**: استفاده از Tailwind CSS و Radix UI
- **پاسخگو**: سازگار با تمام دستگاه‌ها
- **رابط فارسی**: پشتیبانی کامل از زبان فارسی
- **تجربه کاربری**: طراحی UX/UI بهینه

### 🔧 مدیریت سیستم / System Management
- **پنل ادمین**: مدیریت کامل سیستم
- **SSH Operations**: عملیات از راه دور
- **مدیریت دیتابیس**: ابزارهای مدیریت دیتابیس
- **API Health Check**: نظارت بر سلامت سیستم

---

## 🏗️ معماری سیستم / System Architecture

```
SFMS/
├── frontend/                 # React Frontend
│   ├── src/
│   │   ├── components/       # UI Components
│   │   ├── pages/           # Application Pages
│   │   ├── contexts/        # React Contexts
│   │   └── utils/           # Utility Functions
│   └── public/              # Static Assets
├── backend/                  # Flask Backend
│   ├── routes/              # API Routes
│   ├── models/              # Database Models
│   ├── config.py            # Configuration
│   └── main.py              # Main Application
├── qr_live_v2/              # QR Code Processing
└── Smart_Warehouse_Vision_V2/ # Computer Vision
```

### 🔄 جریان داده / Data Flow

1. **Frontend** ←→ **Backend API** ←→ **Database**
2. **Vision System** ←→ **Backend** ←→ **Frontend**
3. **QR Scanner** ←→ **Backend** ←→ **Frontend**

---

## 🚀 نصب و راه‌اندازی / Installation

### پیش‌نیازها / Prerequisites

- Python 3.13+
- Node.js 18+
- SQLite3
- Git

### نصب Backend / Backend Installation

```bash
# Clone the repository
git clone <repository-url>
cd SFMS

# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python create_tables.py

# Run the application
python main.py
```

### نصب Frontend / Frontend Installation

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### متغیرهای محیطی / Environment Variables

Create a `.env` file in the backend directory:

```env
FLASK_ENV=development
FLASK_DEBUG=True
SQLALCHEMY_DATABASE_URI=sqlite:///sfms.db
SECRET_KEY=your-secret-key
```

---

## 📖 استفاده / Usage

### شروع کار / Getting Started

1. **راه‌اندازی سرورها**: Backend و Frontend را اجرا کنید
2. **انتخاب انبار**: از صفحه اصلی انبار مورد نظر را انتخاب کنید
3. **انتخاب دوربین**: دوربین مناسب برای عملیات را انتخاب کنید
4. **شروع عملیات**: عملیات بارگیری یا تخلیه را شروع کنید

### صفحات اصلی / Main Pages

- **صفحه اصلی**: انتخاب نوع عملیات
- **مدیریت انبار**: مدیریت انبارها و محصولات
- **عملیات بارگیری**: مدیریت فرآیند بارگیری
- **عملیات تخلیه**: مدیریت فرآیند تخلیه
- **پنل ادمین**: مدیریت سیستم

---

## 🔌 API Documentation

### Endpoints اصلی / Main Endpoints

#### Health Check
```http
GET /api/health
```

#### Warehouse Management
```http
GET    /api/warehouses
POST   /api/warehouses
PUT    /api/warehouses/<id>
DELETE /api/warehouses/<id>
```

#### Loading Operations
```http
GET    /api/loadings
POST   /api/loadings
PUT    /api/loadings/<id>
DELETE /api/loadings/<id>
```

#### Unloading Operations
```http
GET    /api/unloadings
POST   /api/unloadings
PUT    /api/unloadings/<id>
DELETE /api/unloadings/<id>
```

#### Vision System
```http
POST   /api/vision/start
POST   /api/vision/stop
GET    /api/vision/status
```

### نمونه درخواست / Example Request

```javascript
// Get warehouses
fetch('/api/warehouses')
  .then(response => response.json())
  .then(data => console.log(data));

// Create loading operation
fetch('/api/loadings', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    warehouse_id: 1,
    shipment_id: 'SHIP001',
    items: [...]
  })
});
```

---

## 📁 ساختار پروژه / Project Structure

### Frontend Structure
```
frontend/src/
├── components/           # UI Components
│   ├── ui/              # Base UI Components
│   ├── loadings/        # Loading Components
│   ├── unloadings/      # Unloading Components
│   ├── warehouse/       # Warehouse Components
│   └── products/        # Product Components
├── pages/               # Application Pages
│   ├── admin/           # Admin Pages
│   └── ...              # Other Pages
├── contexts/            # React Contexts
├── utils/               # Utility Functions
└── lib/                 # Library Functions
```

### Backend Structure
```
backend/
├── routes/              # API Routes
│   ├── loading_routes.py
│   ├── unloading_routes.py
│   ├── warehouse_routes.py
│   ├── vision_routes.py
│   └── ...
├── models/              # Database Models
│   ├── database.py
│   └── external_db.py
├── config.py            # Configuration
├── main.py              # Main Application
└── requirements.txt     # Dependencies
```

---

## 🤝 مشارکت / Contributing

### راهنمای مشارکت / Contributing Guidelines

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### استانداردهای کدنویسی / Coding Standards

- استفاده از ESLint برای Frontend
- پیروی از PEP 8 برای Python
- مستندسازی کد با docstring
- تست‌نویسی برای API ها

---

## 📄 لایسنس / License

این پروژه تحت لایسنس MIT منتشر شده است. برای اطلاعات بیشتر فایل `LICENSE` را مطالعه کنید.

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## 📞 پشتیبانی / Support

برای پشتیبانی و سوالات:

- **ایمیل**: [your-email@example.com]
- **GitHub Issues**: [Repository Issues]
- **مستندات**: [Documentation Link]

---

## 🙏 تشکر / Acknowledgments

- **React Team** برای فریم‌ورک عالی
- **Flask Team** برای Backend Framework
- **Tailwind CSS** برای Styling
- **Radix UI** برای UI Components
- **تیم توسعه** برای تلاش‌های بی‌وقفه

---

<div align="center">

**SFMS - Smart Factory Management System**  
*ساخته شده با ❤️ برای صنعت هوشمند*

</div>
