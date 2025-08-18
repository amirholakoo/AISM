import React from 'react';
import { Building2, Phone, Mail, MapPin, Calendar } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-slate-900 text-white py-6 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-4">
          {/* اطلاعات شرکت */}
          <div className="text-center md:text-right">
            <div className="flex items-center justify-center md:justify-start gap-2 mb-2">
              <Building2 className="w-5 h-5 text-blue-400" />
              <h3 className="text-lg font-bold text-white">
                شرکت صنایع تولیدی کاغذ و مقوای همایون
              </h3>
            </div>
            <p className="text-slate-300 text-sm">
              تولید کننده انواع کاغذ و مقوا با کیفیت برتر
            </p>
          </div>

          {/* اطلاعات تماس */}
          <div className="flex flex-col items-center md:items-end gap-1 text-sm text-slate-300">
            <div className="flex items-center gap-2">
              <Phone className="w-4 h-4 text-green-400" />
              <span>۰۲۱-۱۲۳۴۵۶۷۸</span>
            </div>
            <div className="flex items-center gap-2">
              <Mail className="w-4 h-4 text-blue-400" />
              <span>info@hamayunpaper.com</span>
            </div>
            <div className="flex items-center gap-2">
              <MapPin className="w-4 h-4 text-red-400" />
              <span>تهران، خیابان ولیعصر</span>
            </div>
          </div>
        </div>

        {/* خط جداکننده */}
        <div className="border-t border-slate-700 mt-4 pt-4">
          <div className="flex flex-col sm:flex-row justify-between items-center gap-2 text-xs text-slate-400">
            <div className="flex items-center gap-2">
              <Calendar className="w-3 h-3" />
              <span>© {currentYear} تمامی حقوق محفوظ است</span>
            </div>
            <div className="flex items-center gap-4">
              <span>سیستم مدیریت انبار</span>
              <span>•</span>
              <span>v1.0.0-rc1</span>
            </div>
          </div>
          
          {/* توضیحات نسخه بتا */}
          <div className="mt-3 text-center text-xs text-slate-400 max-w-2xl mx-auto">
            <p>
              این نسخه‌ی بتا شامل تمام قابلیت‌های موردنیاز است و آماده‌ی تست توسط کاربران منتخب می‌باشد. در صورت مشاهده هرگونه مشکل یا نیاز به بهبود، لطفاً بازخورد دهید.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer; 