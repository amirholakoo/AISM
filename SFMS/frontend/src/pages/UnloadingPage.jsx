import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Truck } from "lucide-react";

import AlertManager from "@/components/AlertManager";
import ControlPanel from "@/components/ControlPanel";
import Spinner from "@/components/Spinner";
import { API_ENDPOINTS } from "@/config";
import { useUnloadingContext } from "@/contexts/LoadingContext";

export default function UnloadingPage() {
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
      console.log("🛑 Stopping unloading for warehouse:", warehouseId);
      console.log("🔗 API endpoint:", API_ENDPOINTS.VISION_STOP);
      
      const requestBody = { warehouse_id: warehouseId, operation_type: 'unloading' };
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
      
      if (data.success) {
        setMessage(data.message);
        setStarted(false);
        
        // اگر loading_token یا unloading_token داریم، به صفحه ویرایش برو
        if (data.loading_token) {
          navigate(`/loading-edit/${data.loading_token}`);
        } else if (data.unloading_token) {
          navigate(`/unloading-edit/${data.unloading_token}`);
        } else {
          // اگر token نداریم، به خانه برگرد
          navigate('/');
        }
      } else {
        setError(true);
        setMessage(data.message || "خطا در پایان تخلیه");
      }
    } catch (error) {
      console.error("❌ Error stopping loading:", error);
      setError(true);
      setMessage("خطا در پایان تخلیه");
    }
    
    setLoading(false);
  };



  const currentWarehouse = warehouses.find(w => w.id === warehouseId);

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="sticky-header">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
            {/* عنوان و زیرعنوان وسط‌چین در موبایل */}
            <div className="text-center sm:text-right">
              <h1 className="text-2xl font-bold text-slate-900">
                تخلیه در حال اجرا
              </h1>
              <p className="text-slate-600 text-sm">
                {selectedWarehouse && (
                  <span className="text-slate-600">
                    انبار: {selectedWarehouse.persian_name || selectedWarehouse.name}
                  </span>
                )}
              </p>
            </div>
            
            {/* نشانگر تخلیه در موبایل - زیر عنوان */}
            {selectedShipment && (
              <div className="flex justify-center sm:hidden mt-2">
                <Button
                  variant="ghost"
                  size="sm"
                  className="px-4 py-4 bg-orange-100 text-orange-700 text-sm font-medium rounded-lg border border-orange-200 shadow-sm flex items-center gap-2 opacity-100 pointer-events-none"
                >
                  <Truck className="w-4 h-4" />
                  <span><span className="font-semibold">تخلیه</span> {selectedShipment.license_number || 'نامشخص'}</span>
                </Button>
              </div>
            )}
            
            {/* نشانگر تخلیه در دسکتاپ - سمت راست */}
            {selectedShipment && (
              <div className="hidden sm:flex">
                <Button
                  variant="ghost"
                  size="sm"
                  className="px-4 py-4 bg-orange-100 text-orange-700 text-sm font-medium rounded-lg border border-orange-200 shadow-sm flex items-center gap-2 opacity-100 pointer-events-none"
                >
                  <Truck className="w-4 h-4" />
                  <span><span className="font-semibold">تخلیه</span> {selectedShipment.license_number || 'نامشخص'}</span>
                </Button>
              </div>
            )}
          </div>
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
            connectedToExisting={connectedToExisting}
            setConnectedToExisting={setConnectedToExisting}
            started={started}
            message={message}
            error={error}
            setMessage={setMessage}
            setError={setError}
          />

        <div className="space-y-6">
          {/* Status Card */}
          <Card className="bg-white shadow-md">
            <CardContent className="p-6">
          <div className="flex flex-col items-center justify-center py-12 space-y-6">
            <Spinner className="w-12 h-12 text-blue-600" />
              <div className="text-center">
                  <h3 className="text-lg font-semibold text-slate-800">
                  تخلیه در حال اجرا است
                </h3>
                  <p className="text-slate-600 mt-1">
                  منتظر پایان تخلیه باشید
                </p>
            </div>
            
                <div className="flex w-full max-w-md">
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
                      "پایان تخلیه"
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
                      "تایید پایان تخلیه"
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
        </CardContent>
      </Card>
        </div>
      </main>
    </div>
  );
} 