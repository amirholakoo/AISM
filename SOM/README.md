# 📦 SOM: A Sales Order Management

A full-stack Django-based application that allows users to submit daily inventory and order requests using a simple form in **Persian (RTL)**. Admins can manage inventory, approve users, view reports, and control business hours.

---

## 📌 Features

### 👤 User Side
- Login with phone number & SMS password
- View confirmation and response

### 🛠️ Admin Panel
- Approve/reject users
- Manage inventory levels
- Update daily pricing (cash/term)
- View and export order reports
- Control business hours (open/close site)
- Track logs of all actions (💚)

---

## 🧰 Tech Stack

| Component       | Technology                             |
|-----------------|----------------------------------------|
| Backend         | Django (Python)                        |
| Database        | PostgreSQL                             |
| Frontend        | HTML, CSS (offline), RTL support       |
| Form Handling   | Django ModelForm                       |
| Admin Interface | Django Admin                           |

---

## 🗂️ Folder Structure

```

inventory\_system/
├── inventory\_app/
│   ├── models.py
│   ├── views.py
│   ├── forms.py
│   ├── urls.py
│   └── templates/
│       └── ...
├── static/
│   ├── css/
│   │   └── bootstrap.rtl.min.css
│   └── js/
│       └── bootstrap.bundle.min.js
├── .env
├── requirements.txt
├── manage.py
└── README.md

````

---

## 🚀 Setup Instructions

### 1. Clone the Project

```bash
git clone https:https://github.com/amirholakoo/SOMv4.git
cd SOMv4
````

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file:

```env
DEBUG=True
SECRET_KEY=your-django-secret
DATABASE_NAME=inventory_db
DATABASE_USER=your_user
DATABASE_PASSWORD=your_password
DATABASE_HOST=localhost
DATABASE_PORT=5432
```

### 5. Migrate & Run Server

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

---

## 🌐 Accessing the App

* User: `http://localhost:7000/form/`
* Admin: `http://localhost:7000/admin/`

---

## 🔒 Admin Login (Initial)

You must create an admin user:

```bash
python manage.py createsuperuser
```

Then log in at: `http://localhost:7000/admin/`

---

## 🌙 Offline Setup Notes

* Place all Bootstrap and JS files inside `/static/css/` and `/static/js/`.
* Load them in templates using Django's `{% static %}` tag.
* Set `dir="rtl"` and `lang="fa"` in the `<html>` tag.

---

## 📦 Deployment Notes

* Use `gunicorn` or `uvicorn` for production
* Optional: Setup `nginx` as a reverse proxy
* Secure with HTTPS using `certbot` (if public-facing)

---

## 📑 License

MIT License

---

## 🙋‍♂️ Author

Developed with love 💚 for the WORLD and Persian-speaking businesses.


