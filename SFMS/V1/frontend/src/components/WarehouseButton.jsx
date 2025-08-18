import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useUnloadingContext } from "@/contexts/LoadingContext";
import { API_ENDPOINTS } from "@/config";

const WarehouseButton = ({
  warehouse,
  selectedWarehouseId,
  setSelectedWarehouseId,
  onStart,
  onEnd,
  isGlobalRefreshing = false
}) => {

  const navigate = useNavigate();
  const { setSelectedWarehouse, operationType } = useUnloadingContext();
  const [operationTypeData, setOperationTypeData] = useState(null);

  // دریافت اطلاعات نوع عملیات از دیتابیس
  useEffect(() => {
    const fetchOperationType = async () => {
      try {
        const response = await fetch(API_ENDPOINTS.OPERATION_TYPES);
        const data = await response.json();
        
        if (data.success) {
          const currentOperationType = data.data.find(op => op.name === operationType);
          if (currentOperationType) {
            setOperationTypeData(currentOperationType);
          }
        }
      } catch (error) {
        console.error('خطا در دریافت اطلاعات نوع عملیات:', error);
      }
    };

    if (operationType) {
      fetchOperationType();
    }
  }, [operationType]);

  const handleClick = () => {
    // انبار رو انتخاب کن و به صفحه انتخاب دوربین برو
    setSelectedWarehouseId(warehouse.id);
    
    // ذخیره انبار کامل در context
    setSelectedWarehouse(warehouse);
    
    // انتقال به صفحه انتخاب دوربین
    navigate(`/camera-select/${warehouse.id}`);
  };

  const isSelected = selectedWarehouseId === warehouse.id;

  // تعیین رنگ‌ها بر اساس رنگ ذخیره شده در دیتابیس
  const getButtonColors = () => {
    if (!operationTypeData || !operationTypeData.color) {
      // رنگ پیش‌فرض اگر اطلاعات نوع عملیات موجود نباشد
      return {
        selected: 'border-2 border-gray-400 bg-gray-50 text-gray-700 shadow-lg',
        default: 'bg-gradient-to-br from-gray-50 to-slate-50 hover:from-gray-100 hover:to-slate-200 text-gray-700 hover:text-gray-800 hover:shadow-lg border border-gray-200 hover:border-gray-300'
      };
    }

    const color = operationTypeData.color;
    
    // تعریف کلاس‌های رنگ بر اساس نام رنگ
    const colorClasses = {
      blue: {
        selected: 'border-2 border-blue-400 bg-blue-50 text-blue-700 shadow-lg',
        default: 'bg-gradient-to-br from-blue-50 to-indigo-50 hover:from-blue-100 hover:to-blue-200 text-blue-700 hover:text-blue-800 hover:shadow-lg border border-blue-200 hover:border-blue-300'
      },
      red: {
        selected: 'border-2 border-red-400 bg-red-50 text-red-700 shadow-lg',
        default: 'bg-gradient-to-br from-red-50 to-pink-50 hover:from-red-100 hover:to-pink-200 text-red-700 hover:text-red-800 hover:shadow-lg border border-red-200 hover:border-red-300'
      },
      green: {
        selected: 'border-2 border-green-400 bg-green-50 text-green-700 shadow-lg',
        default: 'bg-gradient-to-br from-green-50 to-emerald-50 hover:from-green-100 hover:to-emerald-200 text-green-700 hover:text-green-800 hover:shadow-lg border border-green-200 hover:border-green-300'
      },
      yellow: {
        selected: 'border-2 border-yellow-400 bg-yellow-50 text-yellow-700 shadow-lg',
        default: 'bg-gradient-to-br from-yellow-50 to-amber-50 hover:from-yellow-100 hover:to-amber-200 text-yellow-700 hover:text-yellow-800 hover:shadow-lg border border-yellow-200 hover:border-yellow-300'
      },
      purple: {
        selected: 'border-2 border-purple-400 bg-purple-50 text-purple-700 shadow-lg',
        default: 'bg-gradient-to-br from-purple-50 to-violet-50 hover:from-purple-100 hover:to-violet-200 text-purple-700 hover:text-purple-800 hover:shadow-lg border border-purple-200 hover:border-purple-300'
      },
      pink: {
        selected: 'border-2 border-pink-400 bg-pink-50 text-pink-700 shadow-lg',
        default: 'bg-gradient-to-br from-pink-50 to-rose-50 hover:from-pink-100 hover:to-rose-200 text-pink-700 hover:text-pink-800 hover:shadow-lg border border-pink-200 hover:border-pink-300'
      },
      indigo: {
        selected: 'border-2 border-indigo-400 bg-indigo-50 text-indigo-700 shadow-lg',
        default: 'bg-gradient-to-br from-indigo-50 to-blue-50 hover:from-indigo-100 hover:to-blue-200 text-indigo-700 hover:text-indigo-800 hover:shadow-lg border border-indigo-200 hover:border-indigo-300'
      },
      gray: {
        selected: 'border-2 border-gray-400 bg-gray-50 text-gray-700 shadow-lg',
        default: 'bg-gradient-to-br from-gray-50 to-slate-50 hover:from-gray-100 hover:to-slate-200 text-gray-700 hover:text-gray-800 hover:shadow-lg border border-gray-200 hover:border-gray-300'
      }
    };

    return colorClasses[color] || colorClasses.gray;
  };

  const colors = getButtonColors();

  return (
    <div 
      onClick={handleClick}
      className={`w-full h-20 flex items-center justify-center cursor-pointer shadow-md rounded-lg transition-all duration-200 ${
        isSelected ? colors.selected : colors.default
      }`}
    >
      {/* نام انبار */}
      <div className="text-center">
        <div className="font-semibold text-sm">
          {warehouse.persian_name || warehouse.name}
        </div>
      </div>
    </div>
  );
};

export default WarehouseButton; 