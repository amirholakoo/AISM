import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { EditIcon, ClockIcon, AlertCircleIcon } from "lucide-react";
import Spinner from "@/components/Spinner";
import { API_ENDPOINTS } from "@/config";

const EditLastLoadingButton = ({ 
  onClick, 
  className = "w-full bg-amber-50 hover:bg-amber-100 border-amber-300 text-amber-700 hover:text-amber-800 hover:border-amber-400" 
}) => {
  const [lastOperation, setLastOperation] = useState(null);
  const [fetchingData, setFetchingData] = useState(false);
  const [buttonLoading, setButtonLoading] = useState(false);
  const [remainingTime, setRemainingTime] = useState(0);
  const [isHovered, setIsHovered] = useState(false);

  // دریافت آخرین عملیات (بارگیری یا تخلیه)
  const fetchLastOperation = async () => {
    setFetchingData(true);
    try {
      const res = await fetch(API_ENDPOINTS.OPERATIONS_LAST_COMPLETED);
      const data = await res.json();
      
      if (data.success) {
        setLastOperation(data);
        // مقداردهی اولیه remainingTime با مقدار backend
        setRemainingTime(data.remaining_minutes || 0);
      } else {
        setLastOperation(null);
        setRemainingTime(0);
      }
    } catch (error) {
      console.error('خطا در دریافت آخرین عملیات:', error);
      setLastOperation(null);
    } finally {
      setFetchingData(false);
    }
  };

  // بارگذاری اولیه داده‌ها
  useEffect(() => {
    fetchLastOperation();
  }, []);

  // Refresh داده‌ها هر 1 دقیقه
  useEffect(() => {
    const refreshInterval = setInterval(() => {
      fetchLastOperation();
    }, 60 * 1000); // 1 دقیقه

    return () => clearInterval(refreshInterval);
  }, []);

  // Timer برای آپدیت زمان باقی‌مانده
  useEffect(() => {
    if (!lastOperation || !lastOperation.can_edit) {
      setRemainingTime(0);
      return;
    }

    const updateRemainingTime = () => {
      const now = new Date().getTime();
      const editTime = new Date(lastOperation.edit_time || lastOperation.user_confirm_time).getTime();
      const deadline = editTime + (20 * 60 * 1000); // 20 دقیقه
      const remaining = Math.max(0, Math.floor((deadline - now) / (1000 * 60)));
      
      setRemainingTime(remaining);
    };

    // آپدیت اولیه
    updateRemainingTime();

    // آپدیت هر 1 دقیقه
    const interval = setInterval(updateRemainingTime, 60000);

    return () => clearInterval(interval);
  }, [lastOperation]);

  // handle click with loading state
  const handleClick = async () => {
    setButtonLoading(true);
    try {
      await onClick();
    } finally {
      setButtonLoading(false);
    }
  };

  // دریافت متن وضعیت
  const getStatusText = () => {
    if (!lastOperation) {
      return "آخرین عملیات یافت نشد";
    }
    
    return lastOperation.warehouse_name || "انبار نامشخص";
  };

  // دریافت متن زمان باقی‌مانده
  const getTimeText = () => {
    if (!lastOperation) {
      return "";
    }
    
    // استفاده از remainingTime برای نمایش real-time، اگر صفر باشه از backend استفاده کن
    const timeToShow = remainingTime > 0 ? remainingTime : lastOperation.remaining_minutes;
    
    if (timeToShow === 0) {
      return "زمان ویرایش منقضی شده";
    } else if (timeToShow > 0) {
      return `${timeToShow} دقیقه باقی مانده`;
    }
    
    return "";
  };

  // دریافت آیکون مناسب
  const getStatusIcon = () => {
    if (!lastOperation) {
      return <AlertCircleIcon className="w-4 h-4" />;
    }
    
    if (!lastOperation.can_edit) {
      return <ClockIcon className="w-4 h-4" />;
    }
    
    return <EditIcon className="w-4 h-4" />;
  };

  // دریافت کلاس‌های رنگ
  const getColorClasses = () => {
    // اگر در حال loading باشه، رنگ خاکستری
    if (buttonLoading || fetchingData) {
      return "bg-slate-50 text-slate-600 hover:bg-slate-100 shadow-sm hover:shadow-md";
    }
    
    if (!lastOperation) {
      return "bg-slate-50 text-slate-600 hover:bg-slate-100 shadow-sm hover:shadow-md";
    }
    
    if (!lastOperation.can_edit || (remainingTime === 0 && lastOperation.remaining_minutes === 0)) {
      return "bg-red-50 text-red-600 hover:bg-red-100 shadow-sm hover:shadow-md";
    }
    
    return "bg-amber-50 text-amber-700 hover:bg-amber-100 shadow-sm hover:shadow-md";
  };

  // دریافت عنوان دکمه بر اساس نوع عملیات
  const getButtonTitle = () => {
    if (!lastOperation) {
      return "ویرایش مجدد آخرین عملیات";
    }
    
    if (lastOperation.type === 'loading') {
      return "ویرایش مجدد آخرین بارگیری";
    } else if (lastOperation.type === 'unloading') {
      return "ویرایش مجدد آخرین تخلیه";
    }
    
    return "ویرایش مجدد آخرین عملیات";
  };

  return (
    <div className="relative group">
      <Button
        onClick={handleClick}
        disabled={buttonLoading || fetchingData || !lastOperation?.can_edit || (remainingTime === 0 && lastOperation?.remaining_minutes === 0)}
        variant="ghost"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className={`w-full h-32 flex flex-col items-center justify-center gap-2 transition-all duration-200 ease-in-out cursor-pointer ${getColorClasses()}`}
      >
        {/* خط اول - آیکون */}
        <div className="flex items-center justify-center">
          {buttonLoading || fetchingData ? (
            <Spinner className="w-5 h-5" />
          ) : (
            <div className={`${isHovered ? 'scale-110' : 'scale-100'} transition-transform duration-200`}>
              {getStatusIcon()}
            </div>
          )}
        </div>
        
        {/* خط دوم - عنوان */}
        <div className="text-center">
          <div className="font-semibold text-sm">{getButtonTitle()}</div>
        </div>
        
        {/* خط سوم - اسم انبار */}
        <div className="text-center">
          <div className="text-xs font-medium opacity-80">
            {buttonLoading || fetchingData ? 'در حال بررسی...' : (lastOperation?.warehouse_name || "انبار نامشخص")}
          </div>
        </div>
        
        {/* خط چهارم - زمان باقی‌مانده */}
        <div className="text-center">
          <div className="text-xs font-medium opacity-80">
            {buttonLoading || fetchingData ? '' : getTimeText()}
          </div>
        </div>
      </Button>
    </div>
  );
};

export default EditLastLoadingButton; 