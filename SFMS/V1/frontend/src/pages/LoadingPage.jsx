import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Truck } from "lucide-react";

import AlertManager from "@/components/AlertManager";
import Spinner from "@/components/Spinner";
import JsonDisplayBox from "@/components/JsonDisplayBox";
import ShipmentCard from "@/components/shipment-selection/ShipmentCard";
import { API_ENDPOINTS } from "@/config";
import { useUnloadingContext } from "@/contexts/LoadingContext";

export default function LoadingPage() {
  const { warehouseId } = useParams();
  const navigate = useNavigate();
  const { selectedShipment, selectedWarehouse, setSelectedShipment, setSelectedWarehouse } = useUnloadingContext();
  const [warehouses, setWarehouses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [started, setStarted] = useState(true);
  const [message, setMessage] = useState("");
  const [error, setError] = useState(false);
  const [connectedToExisting, setConnectedToExisting] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [serverResponse, setServerResponse] = useState(null);
  const [debugInfo, setDebugInfo] = useState(null);

  // بازیابی داده‌ها از localStorage اگر در Context موجود نباشند
  useEffect(() => {
    if (!selectedShipment) {
      const savedShipment = localStorage.getItem('selectedShipment');
      if (savedShipment) {
        setSelectedShipment(JSON.parse(savedShipment));
      }
    }
    
    if (!selectedWarehouse) {
      const savedWarehouse = localStorage.getItem('selectedWarehouse');
      if (savedWarehouse) {
        setSelectedWarehouse(JSON.parse(savedWarehouse));
      }
    }
  }, [selectedShipment, selectedWarehouse, setSelectedShipment, setSelectedWarehouse]);

  // دریافت لیست انبارها
  useEffect(() => {
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

  // نمایش تایید پایان بارگیری
  const handleEndClick = () => {
    setShowConfirmation(true);
  };

  // لغو تایید
  const handleCancelConfirmation = () => {
    setShowConfirmation(false);
  };

  // پایان بارگیری
  const handleEnd = async () => {
    setLoading(true);
    setMessage("");
    setError(false);
    setConnectedToExisting(false);
    setShowConfirmation(false);
    
    try {
      console.log("🛑 Stopping loading for warehouse:", warehouseId);
      console.log("🔗 API endpoint:", API_ENDPOINTS.VISION_STOP);
      
      const requestBody = { 
        warehouse_id: warehouseId,
        operation_type: 'loading'
      };
      console.log("📤 Request body:", requestBody);
      
      const res = await fetch(API_ENDPOINTS.VISION_STOP, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      
      console.log("📊 Response status:", res.status);
      console.log("📊 Response ok:", res.ok);
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      console.log("📊 Response data:", data);
      
      // ذخیره پاسخ سرور برای نمایش
      setServerResponse(data);
      
      if (data.success) {
        setMessage(data.message);
        setStarted(false);
        
        // ذخیره loading_token برای انتقال به صفحه ویرایش
        if (data.loading_token) {
          localStorage.setItem('loadingToken', data.loading_token);
          // انتقال خودکار به صفحه ویرایش آیتم‌های بارگیری
          navigate(`/loading-edit/${data.loading_token}`);
        }
        
        // پاک کردن داده‌ها از localStorage
        localStorage.removeItem('selectedShipment');
        localStorage.removeItem('selectedWarehouse');
        
        // پاک کردن Context
        setSelectedShipment(null);
        setSelectedWarehouse(null);
      } else {
        setError(true);
        setMessage(data.message || "خطا در پایان بارگیری");
      }
    } catch (error) {
      console.error("❌ Error stopping loading:", error);
      setError(true);
      setMessage("خطا در اتصال به سرور");
    } finally {
      setLoading(false);
    }
  };

  // شروع بارگیری
  const handleStart = async () => {
    setLoading(true);
    setMessage("");
    setError(false);
    setConnectedToExisting(false);
    
    try {
      console.log("🚀 Starting loading for warehouse:", warehouseId);
      console.log("🔗 API endpoint:", API_ENDPOINTS.VISION_START);
      
      const requestBody = { 
        warehouse_id: warehouseId,
        shipment_id: selectedShipment?.id,
        operation_type: 'loading',
        video_source: 'picamera'  // پیش‌فرض دوربین 0
      };
      console.log("📤 Request body:", requestBody);
      
      const res = await fetch(API_ENDPOINTS.VISION_START, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      
      console.log("📊 Response status:", res.status);
      console.log("📊 Response ok:", res.ok);
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      console.log("📊 Response data:", data);
      
      if (data.success) {
        setMessage(data.message);
        setStarted(true);
        setConnectedToExisting(data.connected_to_existing || false);
      } else {
        setError(true);
        setMessage(data.message || "خطا در شروع بارگیری");
      }
    } catch (error) {
      console.error("❌ Error starting loading:", error);
      setError(true);
      setMessage("خطا در اتصال به سرور");
    } finally {
      setLoading(false);
    }
  };

  // بازگشت به خانه
  const handleBackToHome = () => {
    // پاک کردن داده‌ها از localStorage
    localStorage.removeItem('selectedShipment');
    localStorage.removeItem('selectedWarehouse');
    
    // پاک کردن Context
    setSelectedShipment(null);
    setSelectedWarehouse(null);
    
    navigate('/');
  };

  // پیدا کردن نام انبار
  const getWarehouseName = () => {
    if (selectedWarehouse) {
      return selectedWarehouse.name;
    }
    
    const warehouse = warehouses.find(w => w.id.toString() === warehouseId);
    return warehouse ? warehouse.name : 'نامشخص';
  };

  // بررسی وضعیت دیتابیس
  const handleDebug = async () => {
    try {
      const res = await fetch(`${API_ENDPOINTS.DEBUG_OPERATIONS}?warehouse_id=${warehouseId}`);
      const data = await res.json();
      setDebugInfo(data);
      console.log("🔍 Debug info:", data);
    } catch (error) {
      console.error("❌ Error getting debug info:", error);
    }
  };

  // بررسی وضعیت سرور بینایی
  const handleDebugVisionServer = async () => {
    try {
      const res = await fetch(`${API_ENDPOINTS.DEBUG_VISION_SERVERS}?operation_type=loading`);
      const data = await res.json();
      setDebugInfo(data);
      console.log("🔍 Vision server debug info:", data);
    } catch (error) {
      console.error("❌ Error getting vision server debug info:", error);
    }
  };

  // اگر بارگیری تمام شده و serverResponse موجوده، نمایش log content
  if (!selectedShipment && serverResponse) {
    return (
      <div className="min-h-screen bg-slate-50">
        {/* Header */}
        <div className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
              {/* عنوان و زیرعنوان */}
              <div className="text-center sm:text-right">
                <h1 className="text-2xl font-bold text-slate-900">
                  بارگیری به پایان رسید
                </h1>
                <p className="text-slate-600 text-sm">
                  انبار: {getWarehouseName()}
                </p>
              </div>
              
              {/* دکمه بازگشت */}
              <div className="flex justify-center sm:justify-end">
                <Button 
                  onClick={handleBackToHome}
                  variant="outline"
                  className="border-gray-300 text-gray-700 hover:bg-gray-50"
                >
                  بازگشت به خانه
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="space-y-6">
            {/* نمایش log content */}
            {serverResponse.summary?.log_content && (
              <JsonDisplayBox 
                data={serverResponse.summary.log_content} 
                title="محتوای فایل Log" 
              />
            )}
          </div>
        </main>
      </div>
    );
  }

  // اگر محموله انتخاب نشده و serverResponse هم نیست
  if (!selectedShipment) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="p-6 text-center">
            <Truck className="w-12 h-12 text-gray-400 mx-auto mb-4" />
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
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
            {/* عنوان و زیرعنوان */}
            <div className="text-center sm:text-right">
              <h1 className="text-2xl font-bold text-slate-900">
                عملیات بارگیری
              </h1>
              <p className="text-slate-600 text-sm">
                انبار: {getWarehouseName()}
              </p>
            </div>
            
            {/* دکمه بازگشت */}
            <div className="flex justify-center sm:justify-end">
              <Button 
                onClick={handleBackToHome}
                variant="outline"
                className="border-gray-300 text-gray-700 hover:bg-gray-50"
              >
                بازگشت به خانه
              </Button>
            </div>
          </div>
        </div>
      </div>

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

          {/* Shipment Info Card */}
          <div className="flex justify-center">
            <ShipmentCard
              shipment={selectedShipment}
              variant="default"
            />
          </div>

          {/* Status Card */}
          <Card className="bg-white shadow-md">
            <CardContent className="p-6">
              {started ? (
                // Loading in progress
                <div className="flex flex-col items-center justify-center py-12 space-y-6">
                  <Spinner className="w-12 h-12 text-blue-600" />
                  <div className="text-center">
                    <h3 className="text-lg font-semibold text-slate-800">
                      بارگیری در حال اجرا است
                    </h3>
                    <p className="text-slate-600 mt-1">
                      منتظر پایان بارگیری باشید
                    </p>
                  </div>
                  
                  <div className="flex w-full max-w-md flex-col gap-3">
                    {/* دکمه Debug دیتابیس */}
                    <Button
                      onClick={handleDebug}
                      variant="outline"
                      className="w-full bg-blue-50 hover:bg-blue-100 text-blue-700 border-blue-200"
                    >
                      🔍 بررسی وضعیت دیتابیس
                    </Button>
                    
                    {/* دکمه Debug سرور بینایی */}
                    <Button
                      onClick={handleDebugVisionServer}
                      variant="outline"
                      className="w-full bg-green-50 hover:bg-green-100 text-green-700 border-green-200"
                    >
                      🔍 بررسی سرور بینایی
                    </Button>
                    
                    {!showConfirmation ? (
                      <Button
                        onClick={handleEndClick}
                        disabled={loading}
                        className="w-full bg-red-600 hover:bg-red-700 text-white transition-none"
                      >
                        {loading ? (
                          <>
                            <Spinner className="w-4 h-4 ml-2" />
                            پایان بارگیری
                          </>
                        ) : (
                          "پایان بارگیری"
                        )}
                      </Button>
                    ) : (
                     <div className="flex gap-3 w-full">
                       <Button
                         onClick={handleEnd}
                         disabled={loading}
                         className="flex-1 bg-red-600 hover:bg-red-700 text-white transition-none"
                       >
                         {loading ? (
                           <>
                             <Spinner className="w-4 h-4 ml-2" />
                             در حال پایان...
                           </>
                         ) : (
                           "تایید پایان بارگیری"
                         )}
                       </Button>
                       <Button
                         onClick={handleCancelConfirmation}
                         disabled={loading}
                         variant="outline"
                         className="flex-1 bg-slate-50 hover:bg-slate-100 border-slate-300 text-slate-700 hover:text-slate-800 hover:border-slate-400 transition-none"
                       >
                         لغو
                       </Button>
                     </div>
                   )}
                  </div>
                </div>
              ) : (
                // Loading finished
                <div className="flex flex-col items-center justify-center py-12 space-y-6">
                  <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                    <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <div className="text-center">
                    <h3 className="text-lg font-semibold text-green-800">
                      بارگیری با موفقیت به پایان رسید
                    </h3>
                    <p className="text-slate-600 mt-1">
                      می‌توانید آیتم‌های بارگیری را بررسی و ویرایش کنید
                    </p>
                  </div>
                  
                  <div className="flex w-full max-w-md flex-col gap-3">
                    {/* دکمه ویرایش آیتم‌ها */}
                    {localStorage.getItem('loadingToken') && (
                      <Button
                        onClick={() => navigate(`/loading-edit/${localStorage.getItem('loadingToken')}`)}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                      >
                        ✏️ ویرایش آیتم‌های بارگیری
                      </Button>
                    )}
                    
                    {/* دکمه بازگشت به خانه */}
                    <Button
                      onClick={handleBackToHome}
                      variant="outline"
                      className="w-full border-slate-300 text-slate-700 hover:bg-slate-50"
                    >
                      بازگشت به خانه
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* نمایش پاسخ سرور */}
          {serverResponse && (
            <JsonDisplayBox 
              data={serverResponse} 
              title="پاسخ سرور بینایی" 
            />
          )}

          {/* نمایش اطلاعات Debug */}
          {debugInfo && (
            <JsonDisplayBox 
              data={debugInfo} 
              title="وضعیت دیتابیس" 
            />
          )}
        </div>
      </main>
    </div>
  );
}


