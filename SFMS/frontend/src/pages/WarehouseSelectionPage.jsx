import React, { useEffect, useState, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger 
} from "@/components/ui/dialog";
import AlertManager from "@/components/AlertManager";
import WarehouseButtons from "@/components/WarehouseButtons";
import Spinner from "@/components/Spinner";
import { API_ENDPOINTS } from "@/config";
import { Settings, Warehouse, Package, List, Truck, Home } from "lucide-react";
import HomeButton from "@/components/ui/home-button";
import { useUnloadingContext } from "@/contexts/LoadingContext";

export default function WarehouseSelectionPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { selectedShipment, setSelectedWarehouse, operationType } = useUnloadingContext();
  const [warehouses, setWarehouses] = useState([]);
  const [selectedWarehouseId, setSelectedWarehouseId] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState(false);
  const [pageTransitionLoading, setPageTransitionLoading] = useState(false);
  const warehousesRef = useRef([]);
  const [operationTypeData, setOperationTypeData] = useState(null);

  const formatDate = (dateString) => {
    if (!dateString) return "نامشخص";
    return new Date(dateString).toLocaleDateString('fa-IR');
  };

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

  // دریافت لیست انبارها
  useEffect(() => {
    const loadData = async () => {
      try {
        const warehousesRes = await fetch(API_ENDPOINTS.WAREHOUSES);
        const warehousesData = await warehousesRes.json();
        const warehouses = warehousesData.warehouses || [];
        setWarehouses(warehouses);
        warehousesRef.current = warehouses;
      } catch (error) {
        console.error('خطا در بارگذاری انبارها:', error);
      }
    };
    
    loadData();
  }, []);

  // انتخاب انبار و انتقال به صفحه دوربین
  const handleStart = async (warehouseId = selectedWarehouseId) => {
    // ذخیره اطلاعات انبار در Context
    const selectedWarehouse = warehouses.find(w => w.id === warehouseId);
    if (selectedWarehouse) {
      setSelectedWarehouse(selectedWarehouse);
    }
    
    // انتقال به صفحه انتخاب دوربین
    navigate(`/camera-select/${warehouseId}`);
  };

  // بارگذاری آخرین بارگیری
  const handleLoadLastCompleted = async () => {
    setLoading(true);
    setPageTransitionLoading(true);
    setMessage("");
    setError(false);
    
    try {
      const res = await fetch(API_ENDPOINTS.OPERATIONS_LAST_COMPLETED);
      const data = await res.json();
      
      if (data.success) {
        if (data.status === 'completed' || data.status === 'vision' || data.status === 'edited') {
          setMessage("آخرین عملیات بارگذاری شد.");
          // انتقال به صفحه ویرایش مناسب
          if (data.type === 'loading') {
            navigate(`/loading-edit/${data.token}`);
          } else {
            navigate(`/unloading-edit/${data.token}`);
          }
        } else {
          setError(true);
          setMessage(`عملیات با وضعیت '${data.status}' قابل ویرایش نیست.`);
        }
      } else {
        setError(true);
        setMessage(data.message || "عملیات تکمیل شده‌ای یافت نشد.");
      }
    } catch (error) {
      setError(true);
      setMessage("خطا در بارگذاری آخرین عملیات");
    }
    
    setLoading(false);
    setTimeout(() => {
      setPageTransitionLoading(false);
    }, 1000);
  };

  // تعیین رنگ‌های نشانگر عملیات بر اساس رنگ ذخیره شده در دیتابیس
  const getOperationIndicatorColors = () => {
    if (!operationTypeData || !operationTypeData.color) {
      // رنگ پیش‌فرض اگر اطلاعات نوع عملیات موجود نباشد
      return 'bg-gray-100 text-gray-700 border-gray-200';
    }

    const color = operationTypeData.color;
    
    // تعریف کلاس‌های رنگ بر اساس نام رنگ
    const colorClasses = {
      blue: 'bg-blue-100 text-blue-700 border-blue-200',
      red: 'bg-red-100 text-red-700 border-red-200',
      green: 'bg-green-100 text-green-700 border-green-200',
      yellow: 'bg-yellow-100 text-yellow-700 border-yellow-200',
      purple: 'bg-purple-100 text-purple-700 border-purple-200',
      pink: 'bg-pink-100 text-pink-700 border-pink-200',
      indigo: 'bg-indigo-100 text-indigo-700 border-indigo-200',
      gray: 'bg-gray-100 text-gray-700 border-gray-200'
    };

    return colorClasses[color] || colorClasses.gray;
  };

  // تعیین متن فارسی نوع عملیات
  const getOperationTypeText = () => {
    if (!operationTypeData) {
      return 'عملیات';
    }
    return operationTypeData.persian_name || operationTypeData.name || 'عملیات';
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
                className="p-2 bg-blue-100 rounded-lg cursor-pointer hover:bg-blue-200 transition-colors duration-200"
                onClick={() => navigate('/')}
              >
                <Home className="w-6 h-6 text-blue-600" />
              </div>
            </div>
            
            {/* عنوان و زیرعنوان با دکمه خانه در دسکتاپ */}
            <div className="text-center sm:text-right sm:flex sm:items-center sm:gap-3">
              <div>
                <h1 className="text-2xl font-bold text-slate-900">
                  انتخاب انبار
                </h1>
                <p className="text-slate-600 text-sm">
                  انبار مورد نظر برای شروع عملیات را انتخاب کنید
                </p>
              </div>
              {/* دکمه خانه در کنار عنوان در دسکتاپ */}
              <div className="hidden sm:block order-first">
                <div 
                  className="p-2 bg-blue-100 rounded-lg cursor-pointer hover:bg-blue-200 transition-colors duration-200"
                  onClick={() => navigate('/')}
                >
                  <Home className="w-6 h-6 text-blue-600" />
                </div>
              </div>
            </div>
            
            {/* نشانگر عملیات در دسکتاپ - سمت راست */}
            {selectedShipment && (
              <div className="hidden sm:flex">
                <Button
                  variant="ghost"
                  size="sm"
                  className={`px-4 py-4 text-sm font-medium rounded-lg border shadow-sm flex items-center gap-2 opacity-100 pointer-events-none ${
                    getOperationIndicatorColors()
                  }`}
                >
                  <Truck className="w-4 h-4" />
                  <span>
                    <span className="font-semibold">
                      {getOperationTypeText()}
                    </span> {selectedShipment.license_number || 'نامشخص'}
                  </span>
                </Button>
              </div>
            )}
          </div>
          
          {/* نشانگر عملیات در موبایل - در همان سطر عنوان */}
          {selectedShipment && (
            <div className="flex justify-center sm:hidden mt-4">
              <Button
                variant="ghost"
                size="sm"
                className={`px-4 py-4 text-sm font-medium rounded-lg border shadow-sm flex items-center gap-2 opacity-100 pointer-events-none ${
                  getOperationIndicatorColors()
                }`}
              >
                <Truck className="w-4 h-4" />
                <span>
                  <span className="font-semibold">
                    {getOperationTypeText()}
                  </span> {selectedShipment.license_number || 'نامشخص'}
                </span>
              </Button>
            </div>
          )}
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

        {pageTransitionLoading ? (
          <div className="space-y-6">
            <Card className="bg-white shadow-md">
              <CardContent className="p-6">
          <div className="flex flex-col items-center justify-center py-12 space-y-4">
              <Spinner className="w-12 h-12 text-blue-600" />
                  <p className="text-slate-600 text-lg font-medium">در حال انتقال...</p>
                  <p className="text-slate-500 text-sm">لطفا صبر نمایید</p>
            </div>
              </CardContent>
            </Card>
          </div>
        ) : (
          <div className="space-y-6">

            
            <WarehouseButtons
              warehouses={warehouses}
              selectedWarehouseId={selectedWarehouseId}
              setSelectedWarehouseId={setSelectedWarehouseId}
              started={false}
              loading={loading}
              onStart={handleStart}
              onLoadLastCompleted={handleLoadLastCompleted}
            />
          </div>
        )}
      </main>
    </div>
  );
} 