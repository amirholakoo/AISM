import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Truck, Package, Warehouse, Calendar, Clock } from "lucide-react";

export default function ShipmentCard({ 
  shipment, 
  isSelected = false, 
  onClick, 
  onWarehouseSelect,
  showDetails = true,
  showStatus = true,
  variant = "compact", // فقط "compact" variant باقی مانده
  operationType = "unloading"
}) {
  const formatDate = (dateString) => {
    if (!dateString) return "نامشخص";
    return new Date(dateString).toLocaleDateString('fa-IR');
  };

  const formatTime = (dateString) => {
    if (!dateString) return "نامشخص";
    return new Date(dateString).toLocaleTimeString('fa-IR', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const formatNumber = (number) => {
    if (number === null || number === undefined) return "نامشخص";
    return new Intl.NumberFormat('fa-IR').format(Number(number));
  };

  const getStatusColor = (status) => {
    return status === 'Loaded' 
      ? 'bg-green-100 text-green-800 border border-green-200' 
      : 'bg-yellow-100 text-yellow-800 border border-yellow-200';
  };

  const getStatusText = (status) => {
    return status === 'Loaded' ? 'بارگیری' : 'تخلیه';
  };

  const getOperationColors = () => {
    return operationType === 'loading' 
      ? {
          selected: 'border-blue-400 bg-blue-50',
          icon: 'bg-blue-100',
          iconColor: 'text-blue-600',
          button: 'bg-blue-600 hover:bg-blue-700'
        }
      : {
          selected: 'border-red-400 bg-red-50',
          icon: 'bg-red-100',
          iconColor: 'text-red-600',
          button: 'bg-red-600 hover:bg-red-700'
        };
  };

  const getCardClasses = () => {
    const colors = getOperationColors();
    const baseClasses = "transition-all duration-200";
    return `${baseClasses} cursor-pointer hover:shadow-md border py-3 ${
      isSelected 
        ? `${colors.selected} shadow-md` 
        : 'border-slate-200 hover:border-slate-300 hover:shadow-sm'
    }`;
  };

  if (variant === "compact") {
    return (
      <Card className={getCardClasses()} onClick={onClick}>
        <CardContent className="p-3 py-0">
          <div className="space-y-2">
            {/* Header with icon and license */}
            <div className="flex items-center justify-between">
              <div className={`p-2 rounded-lg ${getOperationColors().icon}`}>
                <Truck className={`w-5 h-5 ${getOperationColors().iconColor}`} />
              </div>
              <div className="text-left">
                <div className="text-sm font-semibold text-slate-900">
                  {shipment.license_number || 'نامشخص'}
                </div>
                <div className="flex items-center gap-1 text-xs text-slate-500">
                  <Clock className="w-3 h-3" />
                  <span>{formatTime(shipment.date)}</span>
                  <Calendar className="w-3 h-3" />
                  <span>{formatDate(shipment.date)}</span>
                </div>
              </div>
            </div>
            
            {/* Divider */}
            <div className="border-t border-slate-100"></div>
            
            {/* Details */}
            {showDetails && (
              <div className="space-y-1">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-medium text-slate-600">تامین‌کننده:</span>
                  <span className="text-xs text-slate-800">{shipment.supplier_name || 'نامشخص'}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs font-medium text-slate-600">نوع ماده:</span>
                  <span className="text-xs text-slate-800">{shipment.material_type || 'نامشخص'}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs font-medium text-slate-600">نام ماده:</span>
                  <span className="text-xs text-slate-800">{shipment.material_name || 'نامشخص'}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs font-medium text-slate-600">واحد:</span>
                  <span className="text-xs text-slate-800">{shipment.unit || 'نامشخص'}</span>
                </div>
              </div>
            )}
            
            {/* دکمه انتخاب انبار - فقط وقتی انتخاب شده نمایش داده می‌شود */}
            <div className={`mt-4 text-center w-full ${isSelected ? 'visible' : 'invisible'}`}>
              <div className="w-full px-0">
                <Button
                  onClick={(e) => {
                    e.stopPropagation(); // جلوگیری از کلیک روی کارت
                    if (onWarehouseSelect) {
                      onWarehouseSelect(shipment.id.toString());
                    }
                  }}
                  disabled={false}
                  className={`w-full h-7 text-white text-xs font-medium rounded-sm disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer ${getOperationColors().button}`}
                >
                  انتخاب انبار
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (variant === "detailed") {
    return (
      <Card className={getCardClasses()} onClick={onClick}>
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Truck className={`w-5 h-5 ${getOperationColors().iconColor}`} />
              <span className="font-semibold text-gray-900">{shipment.license_number}</span>
            </div>
            {showStatus && (
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(shipment.status)}`}>
                {getStatusText(shipment.status)}
              </span>
            )}
          </div>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-1">
                <Clock className="w-4 h-4 text-gray-500" />
                <span className="text-gray-600">ساعت:</span>
              </div>
              <span className="text-gray-900">{formatTime(shipment.date)}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-1">
                <Calendar className="w-4 h-4 text-gray-500" />
                <span className="text-gray-600">تاریخ:</span>
              </div>
              <span className="text-gray-900">{formatDate(shipment.date)}</span>
            </div>
            
            {shipment.supplier_name && (
              <div className="flex justify-between">
                <span className="text-gray-600">تامین‌کننده:</span>
                <span className="text-gray-900 truncate">{shipment.supplier_name}</span>
              </div>
            )}
            
            {shipment.customer_name && (
              <div className="flex justify-between">
                <span className="text-gray-600">مشتری:</span>
                <span className="text-gray-900 truncate">{shipment.customer_name}</span>
              </div>
            )}
            
            {shipment.material_name && (
              <div className="flex justify-between">
                <span className="text-gray-600">ماده:</span>
                <span className="text-gray-900 truncate">{shipment.material_name}</span>
              </div>
            )}
            
            {shipment.net_weight && (
              <div className="flex justify-between">
                <span className="text-gray-600">وزن خالص:</span>
                <span className="text-gray-900 font-medium">{formatNumber(shipment.net_weight)} کیلوگرم</span>
              </div>
            )}
          </div>
          

        </CardContent>
      </Card>
    );
  }

  // Default variant (for EditPage)
  return (
    <Card className="bg-white shadow-md max-w-md w-full">
      <CardContent className="p-6">
        <div className="space-y-4">
          {/* Header with icon and license */}
          <div className="flex items-center justify-between">
            <div className={`p-2 rounded-lg ${getOperationColors().icon}`}>
              <Truck className={`w-5 h-5 ${getOperationColors().iconColor}`} />
            </div>
            <div className="text-left">
              <div className="text-sm font-semibold text-slate-900">
                {shipment.license_number || 'نامشخص'}
              </div>
              <div className="flex items-center gap-1 text-xs text-slate-500">
                <Clock className="w-3 h-3" />
                <span>{formatTime(shipment.date)}</span>
                <Calendar className="w-3 h-3" />
                <span>{formatDate(shipment.date)}</span>
              </div>
            </div>
          </div>
          
          {/* Divider */}
          <div className="border-t border-slate-100"></div>
          
          {/* Details */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-slate-600">تامین‌کننده:</span>
              <span className="text-sm text-slate-800">{shipment.supplier_name || 'نامشخص'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-slate-600">نوع ماده:</span>
              <span className="text-sm text-slate-800">{shipment.material_type || 'نامشخص'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-slate-600">نام ماده:</span>
              <span className="text-sm text-slate-800">{shipment.material_name || 'نامشخص'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-slate-600">واحد:</span>
              <span className="text-sm text-slate-800">{shipment.unit || 'نامشخص'}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}