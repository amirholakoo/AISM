import React, { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Camera, Home, ArrowLeft } from "lucide-react";

import AlertManager from "@/components/AlertManager";
import CameraSelection from "@/components/CameraSelection";
import Spinner from "@/components/Spinner";
import { API_ENDPOINTS } from "@/config";
import { useUnloadingContext } from "@/contexts/LoadingContext";

export default function CameraSelectionPage() {
  const { warehouseId } = useParams();
  const navigate = useNavigate();
  const { selectedShipment, selectedWarehouse, setSelectedWarehouse, operationType, setSelectedCameraId } = useUnloadingContext();
  const [warehouses, setWarehouses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [started, setStarted] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState(false);
  const [connectedToExisting, setConnectedToExisting] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);

  // دریافت لیست انبارها
  React.useEffect(() => {
    const loadWarehouses = async () => {
      try {
        const warehousesRes = await fetch(API_ENDPOINTS.WAREHOUSES);
        const warehousesData = await warehousesRes.json();
        setWarehouses(warehousesData.warehouses || []);
      } catch (error) {
        console.error('خطا در بارگذاری انبارها:', error);
      }
    };
    
    loadWarehouses();
  }, []);

  // نمایش تایید پایان عملیات
  const handleEndClick = () => {
    setShowConfirmation(true);
  };

  // لغو تایید
  const handleCancelConfirmation = () => {
    setShowConfirmation(false);
  };

  // پایان عملیات
  const handleEnd = async () => {
    setLoading(true);
    setMessage("");
    setError(false);
    setConnectedToExisting(false);
    setShowConfirmation(false);
    
    try {
      const currentWarehouseId = selectedWarehouse?.id || warehouseId;
      console.log("🛑 Stopping operation for warehouse:", currentWarehouseId);
      
      const requestBody = { 
        warehouse_id: currentWarehouseId,
        operation_type: operationType  // اضافه کردن operation_type
      };
      
      console.log("📤 Request body:", requestBody);
      
      const res = await fetch(API_ENDPOINTS.VISION_STOP, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      
      if (data.success) {
        setMessage(data.message);
        setStarted(false);
        
        // پاک کردن داده‌ها از localStorage
        localStorage.removeItem('selectedShipment');
        localStorage.removeItem('selectedWarehouse');
        
        // پاک کردن Context
        // setSelectedShipment(null);
        // setSelectedWarehouse(null);
        
        // انتقال به صفحه خانه بعد از 3 ثانیه
        setTimeout(() => {
          navigate('/');
        }, 3000);
      } else {
        setError(true);
        setMessage(data.message || "خطا در پایان عملیات");
      }
    } catch (error) {
      console.error("❌ Error stopping operation:", error);
      setError(true);
      setMessage("خطا در اتصال به سرور");
    } finally {
      setLoading(false);
    }
  };

  // شروع عملیات
  const handleStart = async (warehouseId, cameraId, cameraData) => {
    setLoading(true);
    setMessage("");
    setError(false);
    setConnectedToExisting(false);
    
    try {
      console.log("🚀 Starting operation for warehouse:", warehouseId, "camera:", cameraId);
      
      const requestBody = { 
        warehouse_id: warehouseId,
        shipment_id: selectedShipment?.id,
        camera_id: cameraId,
        operation_type: operationType,
        video_source: cameraData?.video_source || 'picamera'  // استفاده از video_source دوربین یا پیش‌فرض picamera
      };
      
      const res = await fetch(API_ENDPOINTS.VISION_START, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      
      if (data.success) {
        setMessage(data.message);
        setStarted(true);
        setConnectedToExisting(data.connected_to_existing || false);
        
        // ذخیره camera_id در context برای استفاده در stop
        setSelectedCameraId(cameraId);
        
        // انتقال به صفحه عملیات
        if (operationType === 'loading') {
          navigate(`/loading/${warehouseId}`);
        } else {
          navigate(`/unloading/${warehouseId}`);
        }
      } else {
        setError(true);
        setMessage(data.message || "خطا در شروع عملیات");
      }
    } catch (error) {
      console.error("❌ Error starting operation:", error);
      setError(true);
      setMessage("خطا در اتصال به سرور");
    } finally {
      setLoading(false);
    }
  };

  // بازگشت به انتخاب انبار
  const handleBackToWarehouseSelection = () => {
    navigate('/warehouse-select');
  };

  // بازگشت به خانه
  const handleBackToHome = () => {
    // پاک کردن داده‌ها از localStorage
    localStorage.removeItem('selectedShipment');
    localStorage.removeItem('selectedWarehouse');
    
    // پاک کردن Context
    // setSelectedShipment(null);
    // setSelectedWarehouse(null);
    
    navigate('/');
  };

  // پیدا کردن نام انبار
  const getWarehouseName = () => {
    // اول از Context استفاده می‌کنیم
    if (selectedWarehouse) {
      return selectedWarehouse.persian_name || selectedWarehouse.name;
    }
    
    // اگر در Context نبود، از localStorage می‌خوانیم
    const savedWarehouse = localStorage.getItem('selectedWarehouse');
    if (savedWarehouse) {
      try {
        const warehouse = JSON.parse(savedWarehouse);
        return warehouse.persian_name || warehouse.name;
      } catch (error) {
        console.error('خطا در خواندن انبار از localStorage:', error);
      }
    }
    
    // اگر هیچ‌کدام نبود، از لیست انبارها پیدا می‌کنیم
    const warehouse = warehouses.find(w => w.id.toString() === warehouseId);
    return warehouse ? (warehouse.persian_name || warehouse.name) : 'نامشخص';
  };

  if (!selectedShipment) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="p-6 text-center">
            <Camera className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h2 className="text-lg font-semibold text-gray-900 mb-2">محموله انتخاب نشده</h2>
            <p className="text-gray-600 mb-4">لطفاً ابتدا یک محموله انتخاب کنید</p>
            <Button onClick={handleBackToHome} className="w-full">
              بازگشت به خانه
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
             {/* Header */}
       <header className="sticky-header">
         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
           <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
                           {/* عنوان و زیرعنوان با آیکون قابل کلیک */}
              <div className="text-center sm:text-right sm:flex sm:items-center sm:gap-3">
                <div className="flex items-center gap-3">
                  <button 
                    onClick={handleBackToWarehouseSelection}
                    className="p-2 bg-blue-100 rounded-lg hover:bg-blue-200 transition-colors duration-200 cursor-pointer"
                  >
                    <Camera className="w-6 h-6 text-blue-600" />
                  </button>
                  <div>
                    <h1 className="text-2xl font-bold text-slate-900">
                      انتخاب دوربین
                    </h1>
                    <p className="text-slate-600 text-sm">
                      دوربین مورد نظر برای عملیات {operationType === 'loading' ? 'بارگیری' : 'تخلیه'} را انتخاب کنید
                    </p>
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
                     operationType === 'loading' 
                       ? 'bg-blue-100 text-blue-700 border-blue-200' 
                       : 'bg-orange-100 text-orange-700 border-orange-200'
                   }`}
                 >
                   <Camera className="w-4 h-4" />
                   <span>
                     <span className="font-semibold">
                       {operationType === 'loading' ? 'بارگیری' : 'تخلیه'}
                     </span> {selectedShipment.license_number || 'نامشخص'} در انبار {getWarehouseName()}
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
                   operationType === 'loading' 
                     ? 'bg-blue-100 text-blue-700 border-blue-200' 
                     : 'bg-orange-100 text-orange-700 border-orange-200'
                 }`}
               >
                 <Camera className="w-4 h-4" />
                 <span>
                   <span className="font-semibold">
                     {operationType === 'loading' ? 'بارگیری' : 'تخلیه'}
                   </span> {selectedShipment.license_number || 'نامشخص'} در انبار {getWarehouseName()}
                 </span>
               </Button>
             </div>
           )}
         </div>
       </header>

             {/* Main Content */}
       <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
         <div className="space-y-6">
           {/* Alert Manager */}
          <AlertManager
            showEditExpiredAlert={false}
            setShowEditExpiredAlert={() => {}}
            editingLoading={null}
            canEdit={false}
            remainingMinutes={0}
            showRemainingTimeAlert={false}
            setShowRemainingTimeAlert={() => {}}
            connectedToExisting={connectedToExisting}
            setConnectedToExisting={setConnectedToExisting}
            started={started}
            message={message}
            error={error}
            setMessage={setMessage}
            setError={setError}
          />

          {/* Camera Selection Component */}
          <CameraSelection
            warehouseId={selectedWarehouse?.id || warehouseId}
            operationType={operationType}
            onStart={handleStart}
            onBack={handleBackToWarehouseSelection}
            loading={loading}
            started={started}
            showConfirmation={showConfirmation}
            onCancelConfirmation={handleCancelConfirmation}
            onConfirmEnd={handleEnd}
          />
        </div>
      </main>
    </div>
  );
}
