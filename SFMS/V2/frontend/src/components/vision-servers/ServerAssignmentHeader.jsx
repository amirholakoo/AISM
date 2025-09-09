import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Database, Save, RefreshCw } from 'lucide-react';

const ServerAssignmentHeader = ({ onSaveClick, onRefreshClick, loading, saving, serversCount }) => {
  const navigate = useNavigate();

  return (
    <>
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
        {/* آیکون در سطر اول وسط در موبایل */}
        <div className="flex justify-center sm:justify-end sm:hidden mb-2">
          <div 
            className="p-2 bg-orange-100 rounded-lg cursor-pointer hover:bg-orange-200 transition-colors duration-200"
            onClick={() => navigate('/admin')}
          >
            <Database className="w-6 h-6 text-orange-600" />
          </div>
        </div>
        
        {/* عنوان و زیرعنوان با آیکون در دسکتاپ */}
        <div className="text-center sm:text-right sm:flex sm:items-center sm:gap-3">
          <div>
            <h1 className="text-2xl font-bold">
              تخصیص سرورهای بینایی
            </h1>
            <p className="text-slate-600 text-sm">
              تعیین انبارهای مربوط به {serversCount} سرور بینایی
            </p>
          </div>
          {/* آیکون در کنار عنوان در دسکتاپ */}
          <div className="hidden sm:block order-first">
            <div 
              className="p-2 bg-orange-100 rounded-lg cursor-pointer hover:bg-orange-200 transition-colors duration-200"
              onClick={() => navigate('/admin')}
            >
              <Database className="w-6 h-6 text-orange-600" />
            </div>
          </div>
        </div>
        
        {/* دکمه‌ها در دسکتاپ - سمت راست */}
        <div className="hidden sm:flex items-center gap-3">
          <Button
            variant="outline"
            onClick={onRefreshClick}
            disabled={loading}
            className="flex items-center gap-2"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            بارگذاری مجدد
          </Button>
          <Button
            onClick={onSaveClick}
            disabled={saving}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700"
          >
            <Save className="h-4 w-4" />
            {saving ? 'در حال ذخیره...' : 'ذخیره تخصیص‌ها'}
          </Button>
        </div>
      </div>
      
      {/* دکمه‌ها در موبایل - در همان سطر عنوان */}
      <div className="flex justify-center gap-2 sm:hidden mt-4">
        <Button
          variant="outline"
          onClick={onRefreshClick}
          disabled={loading}
          className="flex items-center gap-2"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          بارگذاری مجدد
        </Button>
        <Button
          onClick={onSaveClick}
          disabled={saving}
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700"
        >
          <Save className="h-4 w-4" />
          {saving ? 'در حال ذخیره...' : 'ذخیره تخصیص‌ها'}
        </Button>
      </div>
    </>
  );
};

export default ServerAssignmentHeader;
