import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Plus, Warehouse } from 'lucide-react';

const WarehouseHeader = ({ onAddClick, onSyncClick, syncing, warehouseCount }) => {
  const navigate = useNavigate();

  return (
    <>
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
        {/* آیکون در سطر اول وسط در موبایل */}
        <div className="flex justify-center sm:justify-end sm:hidden mb-2">
          <div 
            className="p-2 bg-green-100 rounded-lg cursor-pointer hover:bg-green-200 transition-colors duration-200"
            onClick={() => navigate('/admin')}
          >
            <Warehouse className="w-6 h-6 text-green-600" />
          </div>
        </div>
        
        {/* عنوان و زیرعنوان با آیکون در دسکتاپ */}
        <div className="text-center sm:text-right sm:flex sm:items-center sm:gap-3">
          <div>
            <h1 className="text-2xl font-bold">
              مدیریت انبارها
            </h1>
            <p className="text-slate-600 text-sm">
              {warehouseCount} انبار در سیستم ثبت شده
            </p>
          </div>
          {/* آیکون در کنار عنوان در دسکتاپ */}
          <div className="hidden sm:block order-first">
            <div 
              className="p-2 bg-green-100 rounded-lg cursor-pointer hover:bg-green-200 transition-colors duration-200"
              onClick={() => navigate('/admin')}
            >
              <Warehouse className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </div>
        
        {/* دکمه‌ها در دسکتاپ - سمت راست */}
        <div className="hidden sm:flex items-center gap-3">
          <Button 
            onClick={onAddClick}
            className="bg-green-600 hover:bg-green-700 border border-green-600 hover:border-green-700 transition-all duration-200"
          >
            <Plus className="h-4 w-4 ml-2" />
            افزودن انبار
          </Button>
          
          <Button 
            onClick={onSyncClick} 
            disabled={syncing}
            variant="outline"
            className="border-gray-300 hover:border-blue-500 hover:bg-blue-50 transition-all duration-200"
          >
            {syncing ? 'در حال همگام‌سازی...' : 'همگام‌سازی دیتابیس'}
          </Button>
        </div>
      </div>
      
      {/* دکمه‌ها در موبایل - در همان سطر عنوان */}
      <div className="flex justify-center gap-2 sm:hidden mt-4">
        <Button 
          onClick={onAddClick}
          className="bg-green-600 hover:bg-green-700 border border-green-600 hover:border-green-700 transition-all duration-200"
        >
          <Plus className="h-4 w-4 ml-2" />
          افزودن انبار
        </Button>
        
        <Button 
          onClick={onSyncClick} 
          disabled={syncing}
          variant="outline"
          className="border-gray-300 hover:border-blue-500 hover:bg-blue-50 transition-all duration-200"
        >
          {syncing ? 'در حال همگام‌سازی...' : 'همگام‌سازی دیتابیس'}
        </Button>
      </div>
    </>
  );
};

export default WarehouseHeader; 