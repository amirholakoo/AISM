import React from "react";
import { Button } from "@/components/ui/button";
import { Truck, Database, Upload, Home } from "lucide-react";
import Spinner from "@/components/Spinner";

const ShipmentSelectionHeader = ({ 
  onBackToHome, 
  onCopyDatabase, 
  loading, 
  startingLoading,
  operationType = 'unloading'
}) => {
  const isLoading = operationType === 'loading';
  
  const getTitle = () => {
    return isLoading ? 'انتخاب محموله بارگیری' : 'انتخاب محموله تخلیه';
  };
  
  const getSubtitle = () => {
    return isLoading 
      ? 'محموله مورد نظر برای عملیات بارگیری را انتخاب کنید'
      : 'محموله مورد نظر برای عملیات تخلیه را انتخاب کنید';
  };
  
  const getOperationText = () => {
    return isLoading ? 'عملیات بارگیری' : 'عملیات تخلیه';
  };
  
  const getOperationIcon = () => {
    return isLoading ? Upload : Truck;
  };
  
  const getOperationColors = () => {
    return isLoading 
      ? 'bg-blue-100 text-blue-700 border-blue-200' 
      : 'bg-red-100 text-red-700 border-red-200';
  };
  
  const OperationIcon = getOperationIcon();

  return (
    <header className="sticky-header">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
          {/* دکمه خانه در سطر اول وسط در موبایل */}
          <div className="flex justify-center sm:justify-end sm:hidden mb-2">
            <div 
              className="p-2 bg-blue-100 rounded-lg cursor-pointer hover:bg-blue-200 transition-colors duration-200"
              onClick={onBackToHome}
            >
              <Home className="w-6 h-6 text-blue-600" />
            </div>
          </div>
          
          {/* عنوان و زیرعنوان با دکمه خانه در دسکتاپ */}
          <div className="text-center sm:text-right sm:flex sm:items-center sm:gap-3">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">
                {getTitle()}
              </h1>
              <p className="text-slate-600 text-sm">
                {getSubtitle()}
              </p>
            </div>
            {/* دکمه خانه در کنار عنوان در دسکتاپ */}
            <div className="hidden sm:block order-first">
              <div 
                className="p-2 bg-blue-100 rounded-lg cursor-pointer hover:bg-blue-200 transition-colors duration-200"
                onClick={onBackToHome}
              >
                <Home className="w-6 h-6 text-blue-600" />
              </div>
            </div>
          </div>
          
          {/* دکمه‌های عملیات در دسکتاپ - سمت راست */}
          <div className="hidden sm:flex items-center gap-3">
            <Button
              onClick={onCopyDatabase}
              disabled={startingLoading || loading}
              variant="ghost"
              size="sm"
              className="px-4 py-4 bg-blue-50 hover:bg-blue-100 border border-blue-300 text-blue-700 hover:text-blue-800 text-sm font-medium rounded-lg shadow-sm flex items-center gap-2"
            >
              {(startingLoading || loading) ? (
                <Spinner className="w-4 h-4" />
              ) : (
                <Database className="w-4 h-4" />
              )}
              <span>دریافت دیتابیس خارجی</span>
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              className={`px-4 py-4 text-sm font-medium rounded-lg border shadow-sm flex items-center gap-2 opacity-100 pointer-events-none ${getOperationColors()}`}
            >
              <OperationIcon className="w-4 h-4" />
              <span>{getOperationText()}</span>
            </Button>
          </div>
        </div>
        
        {/* دکمه‌های عملیات در موبایل - در همان سطر عنوان */}
        <div className="flex justify-center gap-2 sm:hidden mt-4">
          <Button
            onClick={onCopyDatabase}
            disabled={startingLoading || loading}
            variant="ghost"
            size="sm"
            className="px-4 py-4 bg-blue-50 hover:bg-blue-100 border border-blue-300 text-blue-700 hover:text-blue-800 text-sm font-medium rounded-lg shadow-sm flex items-center gap-2"
          >
            {(startingLoading || loading) ? (
              <Spinner className="w-4 h-4" />
            ) : (
              <Database className="w-4 h-4" />
            )}
            <span>دریافت دیتابیس خارجی</span>
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            className={`px-4 py-4 text-sm font-medium rounded-lg border shadow-sm flex items-center gap-2 opacity-100 pointer-events-none ${getOperationColors()}`}
          >
            <OperationIcon className="w-4 h-4" />
            <span>{getOperationText()}</span>
          </Button>
        </div>
      </div>
    </header>
  );
};

export default ShipmentSelectionHeader; 