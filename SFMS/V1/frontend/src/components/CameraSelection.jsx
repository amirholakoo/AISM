import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircleIcon, XCircleIcon, RotateCcwIcon, Camera, Play, Square } from "lucide-react";
import Spinner from "@/components/Spinner";
import { API_ENDPOINTS } from "@/config";
import { useUnloadingContext } from "@/contexts/LoadingContext";

const CameraSelection = ({
  warehouseId,
  operationType,
  onStart,
  onBack,
  loading,
  started,
  showConfirmation,
  onCancelConfirmation,
  onConfirmEnd
}) => {
  const { selectedWarehouse } = useUnloadingContext();
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // دریافت انبار انتخاب شده از context یا localStorage
  const getSelectedWarehouse = () => {
    if (selectedWarehouse && selectedWarehouse.id) {
      return selectedWarehouse.id;
    }
    
    // اگر در context نبود، از localStorage می‌خوانیم
    const savedWarehouse = localStorage.getItem('selectedWarehouse');
    if (savedWarehouse) {
      try {
        const warehouse = JSON.parse(savedWarehouse);
        return warehouse.id;
      } catch (error) {
        console.error('خطا در خواندن انبار از localStorage:', error);
      }
    }
    
    // اگر هیچ‌کدام نبود، از props استفاده می‌کنیم
    return warehouseId;
  };

  // دریافت دوربین‌های مربوط به انبار
  useEffect(() => {
    loadCameras();
  }, [operationType, selectedWarehouse]);

  const loadCameras = async () => {
    setIsLoading(true);
    try {
      const currentWarehouseId = getSelectedWarehouse();
      console.log('🔍 Loading cameras for warehouse:', currentWarehouseId);
      
      // دریافت ویژن سرورهای اختصاص داده شده به انبار انتخاب شده
      const response = await fetch(API_ENDPOINTS.VISION_SERVERS_BY_WAREHOUSE(currentWarehouseId));
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const visionServers = await response.json();
      console.log('📡 Raw vision servers response:', visionServers);
      
      // تبدیل ویژن سرورها به فرمت دوربین و فیلتر کردن بر اساس نوع عملیات
      const cameras = visionServers
        .filter(server => {
          console.log(`🔍 Filtering server ${server.name}:`, {
            is_enabled: server.is_enabled,
            is_available: server.is_available,
            operation_type: server.operation_type,
            expected_type: operationType
          });
          
          return server.is_enabled && 
                 server.is_available &&
                 (server.operation_type === "both" || server.operation_type === operationType);
        })
        .map(server => ({
          id: server.id,
          name: server.name,
          persian_name: server.persian_name || server.name,
          status: server.is_enabled && server.is_available ? "active" : "inactive",
          location: server.location || "محل نامشخص",
          operation_type: server.operation_type,
          warehouse_id: currentWarehouseId,
          video_source: server.video_source
        }));

      console.log('📹 Final cameras list:', cameras);
      setCameras(cameras);
    } catch (error) {
      console.error("خطا در بارگذاری دوربین‌ها:", error);
      setCameras([]);
    } finally {
      setIsLoading(false);
    }
  };

  const refreshCameras = async () => {
    setIsRefreshing(true);
    await loadCameras();
    setTimeout(() => setIsRefreshing(false), 1000);
  };

  const handleCameraSelect = (cameraId) => {
    setSelectedCameraId(cameraId);
  };

  const handleStart = () => {
    if (selectedCameraId) {
      const currentWarehouseId = getSelectedWarehouse();
      const selectedCamera = cameras.find(camera => camera.id === selectedCameraId);
      onStart(currentWarehouseId, selectedCameraId, selectedCamera);
    }
  };

  const getStatusConfig = (status) => {
    switch (status) {
      case 'active':
        return {
          text: 'فعال',
          color: 'bg-green-100 text-green-700',
          icon: <CheckCircleIcon className="w-4 h-4" />
        };
      case 'inactive':
        return {
          text: 'غیرفعال',
          color: 'bg-red-100 text-red-700',
          icon: <XCircleIcon className="w-4 h-4" />
        };
      default:
        return {
          text: 'نامشخص',
          color: 'bg-gray-100 text-gray-700',
          icon: <Square className="w-4 h-4" />
        };
    }
  };

  const getOperationTypeText = (operationType) => {
    switch (operationType) {
      case 'unloading':
        return 'تخلیه';
      case 'loading':
        return 'بارگیری';
      case 'both':
        return 'هر دو';
      default:
        return 'نامشخص';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <Spinner className={`w-8 h-8 mx-auto mb-4 ${operationType === 'loading' ? 'text-blue-600' : 'text-red-600'}`} />
          <p className="text-slate-600">در حال بارگذاری دوربین‌ها...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Camera className={`w-5 h-5 ${operationType === 'loading' ? 'text-blue-600' : 'text-red-600'}`} />
            <h2 className="text-xl font-semibold text-slate-900">
              انتخاب دوربین
            </h2>
          </div>
          <Button
            onClick={refreshCameras}
            disabled={isRefreshing}
            variant="outline"
            size="sm"
            className={`bg-white hover:bg-slate-50 border-slate-300 hover:border-slate-400 ${
              operationType === 'loading' 
                ? 'text-blue-700 hover:text-blue-800 hover:border-blue-400' 
                : 'text-red-700 hover:text-red-800 hover:border-red-400'
            }`}
          >
            {isRefreshing ? (
              <>
                <Spinner className="w-4 h-4 ml-1" />
                به‌روزرسانی...
              </>
            ) : (
              <>
                <RotateCcwIcon className="w-4 h-4 ml-1" />
                به‌روزرسانی
              </>
            )}
          </Button>
        </div>

        {/* Camera Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {cameras.map(camera => {
            const statusConfig = getStatusConfig(camera.status);
            const isSelected = selectedCameraId === camera.id;
            const isDisabled = camera.status !== 'active';

            return (
              <Card 
                key={camera.id}
                className={`cursor-pointer transition-all duration-200 ${
                  isSelected 
                    ? `border-1 border-${operationType === 'loading' ? 'blue' : 'red'}-300 bg-${operationType === 'loading' ? 'blue' : 'red'}-50` 
                    : 'hover:shadow-md border border-slate-200'
                } ${isDisabled ? 'opacity-50' : ''}`}
                onClick={() => !isDisabled && handleCameraSelect(camera.id)}
              >
                <CardContent className="px-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Camera className={`w-5 h-5 ${operationType === 'loading' ? 'text-blue-600' : 'text-red-600'}`} />
                      <h3 className="font-semibold text-slate-900">
                        {camera.persian_name || camera.name}
                      </h3>
                    </div>
                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${statusConfig.color}`}>
                      <div className="flex items-center gap-1">
                        {statusConfig.icon}
                        {statusConfig.text}
                      </div>
                    </div>
                  </div>
                  


                  {isSelected && (
                    <div className="mt-3 pt-3 border-slate-200">
                      <Button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleStart();
                        }}
                        disabled={loading}
                        className={`w-full text-white ${
                          operationType === 'loading' 
                            ? 'bg-blue-600 hover:bg-blue-700' 
                            : 'bg-red-600 hover:bg-red-700'
                        }`}
                      >
                        {loading ? (
                          <>
                            <Spinner className="w-4 h-4 ml-2" />
                            در حال شروع...
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4 ml-2" />
                            شروع عملیات {getOperationTypeText(operationType)}
                          </>
                        )}
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>

        {cameras.length === 0 && (
          <div className="text-center py-8">
            <Camera className={`w-12 h-12 mx-auto mb-4 ${operationType === 'loading' ? 'text-blue-400' : 'text-red-400'}`} />
            <p className="text-gray-600">هیچ دوربینی برای این عملیات یافت نشد.</p>
          </div>
        )}
      </div>


    </div>
  );
};

export default CameraSelection;
