import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  Settings, 
  Warehouse, 
  Package, 
  List, 
  Eye, 
  ArrowLeft,
  Menu,
  X,
  Home,
  BarChart3,
  Users,
  Database
} from 'lucide-react';
import HomeButton from '@/components/ui/home-button';

export default function AdminLayout({ children, title, subtitle }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Listen for sidebar toggle events from child components
  React.useEffect(() => {
    const handleToggleSidebar = () => {
      setSidebarOpen(true);
    };

    window.addEventListener('toggleSidebar', handleToggleSidebar);
    return () => {
      window.removeEventListener('toggleSidebar', handleToggleSidebar);
    };
  }, []);

  const adminMenuItems = [
    {
      id: 'dashboard',
      title: 'داشبورد',
      description: 'نمای کلی سیستم',
      icon: BarChart3,
      path: '/admin',
      color: 'blue'
    },
    {
      id: 'warehouses',
      title: 'مدیریت انبارها',
      description: 'افزودن، ویرایش و حذف انبارها',
      icon: Warehouse,
      path: '/warehouses',
      color: 'green'
    },
    {
      id: 'operation-types',
      title: 'انواع عملیات',
      description: 'تنظیم دکمه‌های صفحه اصلی',
      icon: Settings,
      path: '/admin/operation-types',
      color: 'purple'
    },
    {
      id: 'vision-servers',
      title: 'سرورهای بینایی',
      description: 'تنظیم سرورهای بینایی',
      icon: Eye,
      path: '/admin/vision-servers',
      color: 'indigo'
    },
    {
      id: 'warehouse-assignments',
      title: 'تخصیص سرورها',
      description: 'تعیین سرور برای هر انبار',
      icon: Database,
      path: '/admin/warehouse-assignments',
      color: 'orange'
    },
    {
      id: 'products',
      title: 'مدیریت محصولات',
      description: 'افزودن، ویرایش و حذف محصولات',
      icon: Package,
      path: '/products',
      color: 'teal'
    },
    {
      id: 'unloadings',
      title: 'لیست تخلیه‌ها',
      description: 'مشاهده و مدیریت عملیات',
      icon: List,
      path: '/unloadings',
      color: 'red'
    }
  ];

  const getColorClasses = (colorName) => {
    const colorMap = {
      'blue': {
        bg: 'bg-blue-100',
        text: 'text-blue-600',
        hover: 'hover:bg-blue-50',
        active: 'bg-blue-50 border-blue-200'
      },
      'green': {
        bg: 'bg-green-100',
        text: 'text-green-600',
        hover: 'hover:bg-green-50',
        active: 'bg-green-50 border-green-200'
      },
      'purple': {
        bg: 'bg-purple-100',
        text: 'text-purple-600',
        hover: 'hover:bg-purple-50',
        active: 'bg-purple-50 border-purple-200'
      },
      'indigo': {
        bg: 'bg-indigo-100',
        text: 'text-indigo-600',
        hover: 'hover:bg-indigo-50',
        active: 'bg-indigo-50 border-indigo-200'
      },
      'orange': {
        bg: 'bg-orange-100',
        text: 'text-orange-600',
        hover: 'hover:bg-orange-50',
        active: 'bg-orange-50 border-orange-200'
      },
      'teal': {
        bg: 'bg-teal-100',
        text: 'text-teal-600',
        hover: 'hover:bg-teal-50',
        active: 'bg-teal-50 border-teal-200'
      },
      'red': {
        bg: 'bg-red-100',
        text: 'text-red-600',
        hover: 'hover:bg-red-50',
        active: 'bg-red-50 border-red-200'
      }
    };
    
    return colorMap[colorName] || colorMap['blue'];
  };

  const handleMenuClick = (item) => {
    navigate(item.path);
    setSidebarOpen(false);
  };

  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <div className="min-h-screen bg-slate-50 flex">
      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 right-0 z-50 w-80 bg-white shadow-xl transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0
        ${sidebarOpen ? 'translate-x-0' : 'translate-x-full lg:translate-x-0'}
      `}>
        <div className="flex flex-col h-full">
          {/* Sidebar Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Settings className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900">پنل مدیریت</h2>
                <p className="text-xs text-gray-500">سیستم انبارداری</p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>

          {/* Navigation Menu */}
          <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
            {adminMenuItems.map((item) => {
              const IconComponent = item.icon;
              const { bg, text, hover, active } = getColorClasses(item.color);
              const activeClass = isActive(item.path) ? active : '';
              
              return (
                <button
                  key={item.id}
                  onClick={() => handleMenuClick(item)}
                  className={`
                    w-full flex items-center gap-3 p-3 rounded-lg text-right transition-all duration-200 border
                    ${isActive(item.path) 
                      ? `${activeClass} ${text} font-medium` 
                      : `${hover} text-gray-700 hover:${text} border-transparent`
                    }
                  `}
                >
                  <div className={`p-2 rounded-lg ${bg}`}>
                    <IconComponent className="w-5 h-5" />
                  </div>
                  <div className="flex-1 text-right">
                    <div className="font-medium">{item.title}</div>
                    <div className="text-xs text-gray-500 mt-1">{item.description}</div>
                  </div>
                </button>
              );
            })}
          </nav>

          {/* Sidebar Footer */}
          <div className="p-4 border-t border-gray-200">
            <Button
              variant="outline"
              onClick={() => navigate('/')}
              className="w-full justify-center gap-2"
            >
              <Home className="w-4 h-4" />
              بازگشت به خانه
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 lg:mr-0">
        {/* Page Content */}
        <main className="p-4">
          {children}
        </main>
      </div>
    </div>
  );
} 