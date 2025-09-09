import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Truck, ArrowLeft, ArrowRight, Database, Calendar, Clock, Home, Upload, ShoppingCart, ArrowRightLeft, RotateCcw, Package, Warehouse, List, Download } from "lucide-react";
import AlertManager from "@/components/AlertManager";
import Spinner from "@/components/Spinner";
import { API_ENDPOINTS } from "@/config";
import { useUnloadingContext } from "@/contexts/LoadingContext";

export default function ShipmentSelectionUnloadingPage() {
  const navigate = useNavigate();
  const { 
    selectedShipment: contextShipment, 
    setSelectedShipment: setContextShipment,
    operationType,
    setOperationType
  } = useUnloadingContext();
  
  const [shipments, setShipments] = useState([]);
  const [selectedShipment, setSelectedShipment] = useState(contextShipment);
  const [selectedShipmentId, setSelectedShipmentId] = useState(contextShipment?.id || "");
  const [clickedShipmentId, setClickedShipmentId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pageTransitionLoading, setPageTransitionLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState(false);
  const [operationTypes, setOperationTypes] = useState([]);

  // Get icon component by name
  const getIconComponent = (iconName) => {
    const iconMap = {
      'Truck': Truck,
      'Upload': Upload,
      'ShoppingCart': ShoppingCart,
      'ArrowRightLeft': ArrowRightLeft,
      'RotateCcw': RotateCcw,
      'Package': Package,
      'Warehouse': Warehouse,
      'List': List,
      'Download': Download
    };
    return iconMap[iconName] || Truck;
  };

  // Get unloading operation icon
  const getUnloadingIcon = () => {
    const unloadingOperation = operationTypes.find(op => op.name === 'unloading');
    if (unloadingOperation && unloadingOperation.icon) {
      const IconComponent = getIconComponent(unloadingOperation.icon);
      return <IconComponent className="w-4 h-4" />;
    }
    return <Truck className="w-4 h-4" />; // fallback to Truck icon
  };

  // بارگذاری محموله‌ها
  const loadShipments = async () => {
    try {
      const res = await fetch(API_ENDPOINTS.SHIPMENTS_FOR_UNLOADING);
      const data = await res.json();
      
      if (data.success) {
        setShipments(data.data || []);
      } else {
        setError(true);
        setMessage(data.message || "خطا در بارگذاری محموله‌ها");
      }
    } catch (error) {
      setError(true);
      setMessage("خطا در اتصال به سرور");
    }
  };

  useEffect(() => {
    // اتصال خودکار به دیتابیس خارجی هنگام بارگذاری صفحه
    const autoConnectToExternalDatabase = async () => {
      try {
        setLoading(true);
        setMessage("در حال دریافت دیتابیس خارجی...");
        setError(false);
        
        // اتصال SSH و دریافت دیتابیس خارجی
        const response = await fetch(API_ENDPOINTS.SSH_COPY_DATABASE, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            operation: 'unloading',
            action: 'copy_database'
          })
        });

        const data = await response.json();
        
        if (data.success) {
          setMessage("دیتابیس خارجی با موفقیت دریافت شد");
          setError(false);
          
          // بارگذاری محموله‌ها از دیتابیس جدید
          await loadShipments();
          
          // نمایش پیام موفقیت
          setTimeout(() => {
            setMessage("");
          }, 500);
        } else {
          setMessage("دریافت دیتابیس خارجی ناموفق بود. از دیتابیس قبلی استفاده می‌شود.");
          setError(true);
          
          // بارگذاری محموله‌ها از دیتابیس موجود
          await loadShipments();
          
          // نمایش پیام
          setTimeout(() => {
            setMessage("");
          }, 500);
        }
      } catch (error) {
        console.error("❌ Error connecting to external database:", error);
        setMessage("خطا در اتصال به سرور SSH. از دیتابیس موجود استفاده می‌شود.");
        setError(true);
        
        // بارگذاری محموله‌ها از دیتابیس موجود
        await loadShipments();
        
        // نمایش پیام
        setTimeout(() => {
          setMessage("");
        }, 500);
      } finally {
        setLoading(false);
      }
    };

    autoConnectToExternalDatabase();
    loadOperationTypes();
  }, []);

  // کلیک روی کارت محموله
  const handleCardClick = (shipmentId) => {
    setClickedShipmentId(clickedShipmentId === shipmentId ? null : shipmentId);
  };

  // بارگذاری انواع عملیات
  const loadOperationTypes = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.OPERATION_TYPES);
      const data = await response.json();
      
      if (data.success) {
        setOperationTypes(data.data || []);
      } else {
        console.error('Error fetching operation types:', data.error);
      }
    } catch (error) {
      console.error('Error fetching operation types:', error);
    }
  };

  // انتخاب محموله و رفتن به انتخاب انبار
  const handleSelectShipment = async (shipment) => {
    if (!shipment) {
      setError(true);
      setMessage("لطفاً یک محموله انتخاب کنید");
      return;
    }
    
    setSelectedShipmentId(shipment.id);
    setSelectedShipment(shipment);
    
    try {
      // ابتدا اتصال دیتابیس را ببند
      console.log("🔒 Closing database connections...");
      const closeRes = await fetch(API_ENDPOINTS.DATABASE_CLOSE, {
        method: "POST",
        headers: { "Content-Type": "application/json" }
      });
      
      if (closeRes.ok) {
        console.log("✅ Database connections closed");
      }
      
      // کمی صبر کن تا اتصالات بسته شوند
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // ذخیره اطلاعات در Context
      setContextShipment(shipment);
      setOperationType('unloading');
      
      // ذخیره در localStorage برای استفاده در صفحات بعدی
      localStorage.setItem('selectedShipment', JSON.stringify(shipment));
      localStorage.setItem('operationType', 'unloading');
      
      setPageTransitionLoading(true);
      // انتقال به صفحه انتخاب انبار
      navigate(`/warehouse-select`);
    } catch (error) {
      console.error("❌ Error processing shipment selection:", error);
      setError(true);
      setMessage("خطا در پردازش انتخاب محموله");
    }
  };

  // بازگشت به صفحه اصلی
  const handleBackToHome = () => {
    navigate('/');
  };

  // دریافت دیتابیس خارجی
  const handleReceiveExternalDatabase = async () => {
    try {
      setLoading(true);
      setMessage("در حال دریافت دیتابیس خارجی...");
      setError(false);
      
      // اتصال SSH و دریافت دیتابیس خارجی
      const response = await fetch(API_ENDPOINTS.SSH_COPY_DATABASE, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          operation: 'unloading',
          action: 'copy_database'
        })
      });

      const data = await response.json();
      
      if (data.success) {
        setMessage("دیتابیس خارجی با موفقیت دریافت شد");
        setError(false);
        
        // بارگذاری مجدد محموله‌ها از دیتابیس جدید
        await loadShipments();
        
        // نمایش پیام موفقیت
        setTimeout(() => {
          setMessage("");
        }, 500);
      } else {
        setMessage("دریافت دیتابیس خارجی ناموفق بود. از دیتابیس قبلی استفاده می‌شود.");
        setError(true);
        
        // بارگذاری محموله‌ها از دیتابیس موجود
        await loadShipments();
        
        // نمایش پیام
        setTimeout(() => {
          setMessage("");
        }, 500);
      }
    } catch (error) {
      console.error("❌ Error connecting to external database:", error);
      setMessage("دریافت دیتابیس خارجی ناموفق بود. از دیتابیس قبلی استفاده می‌شود.");
      setError(true);
      
      // بارگذاری محموله‌ها از دیتابیس موجود
      await loadShipments();
      
      // نمایش پیام
      setTimeout(() => {
        setMessage("");
      }, 500);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="sticky-header">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
            {/* دکمه خانه در سطر اول وسط در موبایل */}
            <div className="flex justify-center sm:justify-end sm:hidden mb-2">
              <div 
                className="p-2 bg-red-100 rounded-lg cursor-pointer hover:bg-red-200 transition-colors duration-200"
                onClick={handleBackToHome}
              >
                <Home className="w-6 h-6 text-red-600" />
              </div>
            </div>
            
            {/* عنوان و زیرعنوان با دکمه خانه در دسکتاپ */}
            <div className="text-center sm:text-right sm:flex sm:items-center sm:gap-3">
              <div>
                <h1 className="text-2xl font-bold text-slate-900">
                  انتخاب محموله تخلیه
                </h1>
                <p className="text-slate-600 text-sm">
                  محموله مورد نظر برای عملیات تخلیه را انتخاب کنید
                </p>
              </div>
               {/* دکمه خانه در کنار عنوان در دسکتاپ */}
               <div className="hidden sm:block order-first">
                 <div 
                   className="p-2 bg-red-100 rounded-lg cursor-pointer hover:bg-red-200 transition-colors duration-200"
                   onClick={handleBackToHome}
                 >
                   <Home className="w-6 h-6 text-red-600" />
                 </div>
               </div>
            </div>

            {/* Header Buttons */}
            <div className="hidden sm:flex items-center gap-3">
              <Button
                onClick={handleReceiveExternalDatabase}
                disabled={loading}
                variant="ghost"
                size="sm"
                className="px-4 py-4 bg-blue-50 hover:bg-blue-100 border border-blue-300 text-blue-700 hover:text-blue-800 text-sm font-medium rounded-lg shadow-sm flex items-center gap-2"
              >
                {loading ? (
                  <Spinner className="w-4 h-4" />
                ) : (
                  <Database className="w-4 h-4" />
                )}
                <span>دریافت دیتابیس خارجی</span>
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                className="px-4 py-4 text-sm font-medium rounded-lg border shadow-sm flex items-center gap-2 opacity-100 pointer-events-none bg-red-100 text-red-700 border-red-200"
              >
                {getUnloadingIcon()}
                <span>عملیات تخلیه</span>
              </Button>
            </div>
          </div>
        </div>
        
        {/* دکمه‌های عملیات در موبایل - در همان سطر عنوان */}
        <div className="flex justify-center gap-2 sm:hidden mt-4">
          <Button
            onClick={handleReceiveExternalDatabase}
            disabled={loading}
            variant="ghost"
            size="sm"
            className="px-4 py-4 bg-blue-50 hover:bg-blue-100 border border-blue-300 text-blue-700 hover:text-blue-800 text-sm font-medium rounded-lg shadow-sm flex items-center gap-2"
          >
            {loading ? (
              <Spinner className="w-4 h-4" />
            ) : (
              <Database className="w-4 h-4" />
            )}
            <span>دریافت دیتابیس خارجی</span>
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            className="px-4 py-4 text-sm font-medium rounded-lg border shadow-sm flex items-center gap-2 opacity-100 pointer-events-none bg-red-100 text-red-700 border-red-200"
          >
            {getUnloadingIcon()}
            <span>عملیات تخلیه</span>
          </Button>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <AlertManager
          showEditExpiredAlert={false}
          setShowEditExpiredAlert={() => {}}
          editingLoading={null}
          canEdit={false}
          remainingMinutes={0}
          showRemainingTimeAlert={false}
          setShowRemainingTimeAlert={() => {}}
          connectedToExisting={false}
          setConnectedToExisting={() => {}}
          started={false}
          message={message}
          error={error}
          setMessage={setMessage}
          setError={setError}
        />

        <div className="space-y-6">
          {/* کارت‌های محموله‌ها */}
          {!loading && shipments.length > 0 && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {shipments.map((shipment) => (
                  <Card 
                    key={shipment.id} 
                    className={`transition-all duration-200 cursor-pointer hover:shadow-md border py-3 ${
                      clickedShipmentId === shipment.id 
                        ? 'border-red-400 bg-red-50 shadow-md' 
                        : 'border-slate-200 hover:border-slate-300 hover:shadow-sm'
                    }`}
                    onClick={() => handleCardClick(shipment.id)}
                  >
                    <CardContent className="p-3 py-0">
                      <div className="space-y-2">
                        {/* Header with icon and license */}
                        <div className="flex items-center justify-between">
                          <div className="p-2 bg-red-100 rounded-lg">
                            <Truck className="w-5 h-5 text-red-600" />
                          </div>
                          <div className="text-left">
                            <div className="text-sm font-semibold text-slate-900">
                              {shipment.license_number || 'نامشخص'}
                            </div>
                            <div className="flex items-center gap-1 text-xs text-slate-500">
                              <Clock className="w-3 h-3 text-slate-400" />
                              <span className="font-medium">{shipment.created_at ? new Date(shipment.created_at).toLocaleTimeString('fa-IR', { hour: '2-digit', minute: '2-digit' }) : '۰۸:۵۸'}</span>
                              <Calendar className="w-3 h-3 text-slate-400" />
                              <span className="font-medium">{shipment.created_at ? new Date(shipment.created_at).toLocaleDateString('fa-IR') : '۱۴۰۴/۱/۷'}</span>
                            </div>
                          </div>
                        </div>
                        
                        {/* Divider */}
                        <div className="border-t border-slate-100"></div>
                        
                        {/* Details */}
                        <div className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-xs font-medium text-slate-600">تامین‌کننده:</span>
                            <span className="text-xs text-slate-800">{shipment.supplier_name || 'نامشخص'}</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-xs font-medium text-slate-600">نوع ماده:</span>
                            <span className="text-xs text-slate-800">{shipment.material_type || shipment.material_name || 'نامشخص'}</span>
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
                        
                        {/* دکمه انتخاب انبار - فقط وقتی انتخاب شده نمایش داده می‌شود */}
                        <div className={`mt-4 text-center w-full ${clickedShipmentId === shipment.id ? 'visible' : 'invisible'}`}>
                          <div className="w-full px-0">
                            <Button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleSelectShipment(shipment);
                              }}
                              disabled={loading || pageTransitionLoading}
                              className="w-full h-7 bg-red-600 hover:bg-red-700 text-white text-xs font-medium rounded-sm disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                            >
                              {pageTransitionLoading && selectedShipmentId === shipment.id ? (
                                <>
                                  <Spinner className="w-4 h-4" />
                                  در حال انتقال...
                                </>
                              ) : (
                                'انتخاب انبار'
                              )}
                            </Button>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* Loading state */}
          {loading && (
            <div className="space-y-6">
              {/* Skeleton Loading Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-5xl mx-auto">
                {Array.from({ length: 6 }).map((_, index) => (
                  <Card key={index} className="bg-white shadow-md animate-pulse border py-3">
                    <CardContent className="p-3 py-0">
                      <div className="space-y-2">
                        {/* Header with icon and license skeleton */}
                        <div className="flex items-center justify-between">
                          <div className="p-2 bg-gray-200 rounded-lg w-9 h-9"></div>
                          <div className="text-left">
                            <div className="w-20 h-5 bg-gray-200 rounded mb-1"></div>
                            <div className="flex items-center gap-1">
                              <div className="w-3 h-4 bg-gray-200 rounded"></div>
                              <div className="w-12 h-4 bg-gray-200 rounded"></div>
                              <div className="w-3 h-4 bg-gray-200 rounded"></div>
                              <div className="w-16 h-4 bg-gray-200 rounded"></div>
                            </div>
                          </div>
                        </div>
                        
                        {/* Divider skeleton */}
                        <div className="border-t border-gray-200"></div>
                        
                        {/* Details skeleton */}
                        <div className="space-y-1">
                          <div className="flex justify-between items-center">
                            <div className="w-16 h-4 bg-gray-200 rounded"></div>
                            <div className="w-20 h-4 bg-gray-200 rounded"></div>
                          </div>
                          <div className="flex justify-between items-center">
                            <div className="w-12 h-4 bg-gray-200 rounded"></div>
                            <div className="w-16 h-4 bg-gray-200 rounded"></div>
                          </div>
                          <div className="flex justify-between items-center">
                            <div className="w-14 h-4 bg-gray-200 rounded"></div>
                            <div className="w-18 h-4 bg-gray-200 rounded"></div>
                          </div>
                          <div className="flex justify-between items-center">
                            <div className="w-8 h-4 bg-gray-200 rounded"></div>
                            <div className="w-12 h-4 bg-gray-200 rounded"></div>
                          </div>
                        </div>
                        
                        {/* Button skeleton */}
                        <div className="mt-4 text-center w-full">
                          <div className="w-full h-7 bg-gray-200 rounded-sm"></div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* Empty state */}
          {!loading && shipments.length === 0 && (
            <div className="text-center py-12">
              <Truck className="w-16 h-16 text-slate-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-slate-800 mb-2">
                محموله‌ای برای تخلیه یافت نشد
              </h3>
              <p className="text-slate-600">
                هیچ محموله‌ای در وضعیت مناسب برای تخلیه وجود ندارد.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
