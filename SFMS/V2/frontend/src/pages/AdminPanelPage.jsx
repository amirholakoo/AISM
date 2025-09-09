import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  Settings, 
  Warehouse, 
  Package, 
  List, 
  Eye, 
  Home,
  Database
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function AdminPanelPage() {
  const navigate = useNavigate();

  const adminMenuItems = [
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
      id: 'operations',
      title: 'لیست عملیات انجام شده',
      description: 'مشاهده و مدیریت عملیات',
      icon: List,
      path: '/operations',
      color: 'red'
    }
  ];

  const getColorClasses = (colorName) => {
    const colorMap = {
      'blue': {
        bg: 'bg-blue-100',
        text: 'text-blue-600',
        hover: 'hover:bg-blue-50'
      },
      'green': {
        bg: 'bg-green-100',
        text: 'text-green-600',
        hover: 'hover:bg-green-50'
      },
      'purple': {
        bg: 'bg-purple-100',
        text: 'text-purple-600',
        hover: 'hover:bg-purple-50'
      },
      'indigo': {
        bg: 'bg-indigo-100',
        text: 'text-indigo-600',
        hover: 'hover:bg-indigo-50'
      },
      'orange': {
        bg: 'bg-orange-100',
        text: 'text-orange-600',
        hover: 'hover:bg-orange-50'
      },
      'teal': {
        bg: 'bg-teal-100',
        text: 'text-teal-600',
        hover: 'hover:bg-teal-50'
      },
      'red': {
        bg: 'bg-red-100',
        text: 'text-red-600',
        hover: 'hover:bg-red-50'
      }
    };
    
    return colorMap[colorName] || colorMap['blue'];
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
            {/* دکمه پنل مدیریت در سطر اول وسط در موبایل */}
            <div className="flex justify-center sm:justify-end sm:hidden mb-2">
              <div 
                className="p-2 bg-blue-100 rounded-lg cursor-pointer hover:bg-blue-200 transition-colors duration-200"
                onClick={() => navigate('/')}
              >
                <Home className="w-6 h-6 text-blue-600" />
              </div>
            </div>
            
            {/* عنوان و زیرعنوان با دکمه خانه در دسکتاپ */}
            <div className="text-center sm:text-right sm:flex sm:items-center sm:gap-3">
              <div>
                <h1 className="text-2xl font-bold">
                  پنل مدیریت
                </h1>
                <p className="text-slate-600 text-sm">
                  سیستم انبارداری
                </p>
              </div>
              {/* دکمه پنل مدیریت در کنار عنوان در دسکتاپ */}
              <div className="hidden sm:block order-first">
                <div 
                  className="p-2 bg-blue-100 rounded-lg cursor-pointer hover:bg-blue-200 transition-colors duration-200"
                  onClick={() => navigate('/')}
                >
                  <Home className="w-6 h-6 text-blue-600" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-6">
          {/* Dashboard Header */}
          <div className="bg-white rounded-xl shadow-md p-6">
            <div className="text-center">
              <h2 className="text-xl font-semibold text-slate-900 mb-1">
                خوش آمدید به پنل مدیریت
              </h2>
              <p className="text-slate-600">
                مدیریت کامل سیستم بارگیری و انبارداری
              </p>
            </div>
          </div>

                 {/* Management Cards */}
         <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-w-3xl mx-auto">
           {adminMenuItems.map((item) => {
             const IconComponent = item.icon;
             const { bg, text } = getColorClasses(item.color);
             
             return (
               <Card 
                 key={item.id} 
                 className="bg-white shadow-md hover:shadow-lg transition-all duration-200 cursor-pointer"
                 onClick={() => navigate(item.path)}
               >
                 <CardContent className="px-3 py-0 text-center">
                   <div className="flex flex-col items-center space-y-0.5">
                     <div className={`p-3 rounded-full ${bg}`}>
                       <IconComponent className={`w-12 h-12 ${text}`} />
                     </div>
                     {/* Separator line between icon and title */}
                     <div className="w-16 h-px bg-gray-300 my-2"></div>
                     <div>
                       <h3 className="text-base font-semibold mb-0 text-slate-900">
                         {item.title}
                       </h3>
                       <p className="text-xs text-slate-600">
                         {item.description}
                       </p>
                     </div>
                   </div>
                 </CardContent>
               </Card>
             );
           })}
         </div>
      </div>
    </main>
  </div>
);
} 