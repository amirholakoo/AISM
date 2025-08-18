import React, { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { API_ENDPOINTS } from "@/config";
import { useUnloadingContext } from "@/contexts/LoadingContext";
import AlertManager from "@/components/AlertManager";
import { 
  ShipmentSelectionHeader, 
  ShipmentSelectionContent 
} from "@/components/shipment-selection";
import {
  manageDatabaseConnection,
  copyDatabaseViaSSH,
  testSSHConnection,
  checkDatabaseStatus,
  checkDatabaseStatusExtended,
  checkRemoteServerHealth,
  quickDatabaseCheck
} from "@/utils/databaseUtils";

export default function ShipmentSelectionLoadingPage() {
  const navigate = useNavigate();
  const { setSelectedShipment: setContextShipment, setOperationType } = useUnloadingContext();
  const [shipments, setShipments] = useState([]);
  const [selectedShipment, setSelectedShipment] = useState(null);
  const [selectedShipmentId, setSelectedShipmentId] = useState("");
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [error, setError] = useState(false);
  const [pageTransitionLoading, setPageTransitionLoading] = useState(false);
  const [startingLoading, setStartingLoading] = useState(false);

  // دریافت لیست محموله‌ها
  useEffect(() => {
    const loadShipments = async () => {
      try {
        setLoading(true);
        // ابتدا تلاش کن از سرور خارجی دیتابیس را دریافت کنی
        await handleCopyDatabase(true); // نمایش alert در هنگام بارگذاری اولیه
      } catch (error) {
        console.error("❌ Error loading shipments:", error);
        setError(true);
        setMessage("خطا در بارگذاری محموله‌ها.");
      } finally {
        setLoading(false);
      }
    };

    loadShipments();
  }, []);

  // تغییر محموله انتخاب شده
  const handleShipmentChange = async (shipmentId) => {
    if (!shipmentId) {
      setSelectedShipment(null);
      setSelectedShipmentId("");
      return;
    }

    setSelectedShipmentId(shipmentId);
    
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
      
      // حالا جزئیات محموله را دریافت کن
      console.log("📡 Fetching shipment details for ID:", shipmentId);
      const res = await fetch(API_ENDPOINTS.SHIPMENT_DETAIL(shipmentId));
      const data = await res.json();
      
      if (data.success) {
        setSelectedShipment(data.data);
        console.log("✅ Shipment details loaded successfully");
      } else {
        setError(true);
        setMessage(data.message || "خطا در دریافت جزئیات محموله");
      }
    } catch (error) {
      console.error("❌ Error loading shipment details:", error);
      setError(true);
      setMessage("خطا در اتصال به سرور");
    }
  };

  // ادامه به انتخاب انبار
  const handleContinueToWarehouse = () => {
    if (!selectedShipment) {
      setError(true);
      setMessage("لطفاً یک محموله انتخاب کنید");
      return;
    }
    
    // ذخیره اطلاعات در Context
    setContextShipment(selectedShipment);
    setOperationType('loading');
    
    setPageTransitionLoading(true);
    // انتقال به صفحه انتخاب انبار
    navigate(`/warehouse-select`);
  };

  // انتخاب انبار از طریق دکمه
  const handleWarehouseSelect = (shipmentId) => {
    // اگر محموله انتخاب شده همان محموله فعلی است، به مرحله بعد برو
    if (selectedShipmentId === shipmentId && selectedShipment) {
      handleContinueToWarehouse();
    }
  };

  // بازگشت به خانه
  const handleBackToHome = () => {
    navigate('/');
  };

  // بارگذاری مستقیم محموله‌ها بدون بررسی‌های اضافی
  const loadShipmentsDirectly = async () => {
    try {
      setMessage("در حال بارگذاری محموله‌ها...");
      setError(false);
      setStartingLoading(true);
      
      console.log("📡 Fetching shipments data directly...");
      const res = await fetch(API_ENDPOINTS.SHIPMENTS_FOR_LOADING, {
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      const data = await res.json();
      
      if (data.success) {
        const shipmentsData = data.data || [];
        setShipments(shipmentsData);
        console.log(`✅ Successfully loaded ${shipmentsData.length} shipments directly`);
        setMessage(`محموله‌ها بارگذاری شدند (${shipmentsData.length} محموله)`);
        setError(false);
      } else {
        setError(true);
        setMessage(data.message || "خطا در دریافت محموله‌ها");
      }
    } catch (error) {
      console.error("❌ Error loading shipments directly:", error);
      setError(true);
      setMessage("خطا در اتصال به سرور");
    } finally {
      setStartingLoading(false);
    }
  };

  // دریافت دیتابیس از سرور خارجی و بارگذاری محموله‌ها
  const handleCopyDatabase = async (showAlerts = true) => {
    try {
      if (showAlerts) {
        setMessage("در حال دریافت دیتابیس خارجی...");
        setError(false);
        setStartingLoading(true);
        setLoading(true); // فعال کردن اسکلتون لودینگ
      }
      
      // بررسی وضعیت سرور خارجی
      console.log("🔍 Checking remote server status...");
      const remoteServerHealthy = await checkRemoteServerHealth();
      let hasError = false;
      let errorMessage = "";
      
      if (!remoteServerHealthy) {
        console.warn("⚠️ Remote server is offline");
        hasError = true;
        errorMessage = "سرور خارجی در دسترس نیست. از دیتابیس قبلی استفاده می‌شود.";
      }
      
      // ابتدا اتصال دیتابیس را ببند
      await manageDatabaseConnection();
      
      // کمی صبر کن تا اتصالات بسته شوند
      await new Promise(resolve => setTimeout(resolve, 200));
      
      // کپی دیتابیس از سرور remote
      console.log("🔄 Copying database from remote server...");
      const sshTest = await testSSHConnection();
      let databaseUpdated = false;
      
      if (sshTest) {
        const copySuccess = await copyDatabaseViaSSH();
        if (copySuccess) {
          console.log("✅ Database copied successfully from remote server");
          databaseUpdated = true;
        } else {
          console.warn("⚠️ Failed to copy database, continuing with existing file");
          hasError = true;
          errorMessage = "خطا در کپی دیتابیس از سرور خارجی. از دیتابیس قبلی استفاده می‌شود.";
        }
      } else {
        console.warn("⚠️ SSH connection failed, continuing with existing database");
        hasError = true;
        errorMessage = "اتصال به سرور خارجی برقرار نشد. از دیتابیس قبلی استفاده می‌شود.";
      }
      
      // کمی صبر کن تا فایل کپی شود
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // بررسی وضعیت دیتابیس با تلاش چندباره
      console.log("🔍 Checking database accessibility...");
      let dbStatus = await quickDatabaseCheck();
      
      // اگر تلاش سریع موفق نبود، تلاش extended انجام بده
      if (!dbStatus) {
        console.log("🔄 Quick check failed, trying extended database check...");
        dbStatus = await checkDatabaseStatusExtended(3, 150, 500);
      }
      
      if (!dbStatus) {
        console.warn("⚠️ Database accessibility check failed after all attempts, but continuing with data loading...");
        hasError = true;
        errorMessage = "خطا در دسترسی به دیتابیس. لطفاً دوباره تلاش کنید.";
      } else {
        console.log("✅ Database accessibility confirmed");
      }
      
      // حالا اطلاعات محموله‌ها را دریافت کن
      console.log("📡 Fetching shipments data...");
      const res = await fetch(API_ENDPOINTS.SHIPMENTS_FOR_LOADING);
      const data = await res.json();
      console.log("📊 Shipments data:", data.data);
      
      if (data.success) {
        const shipmentsData = data.data || [];
        setShipments(shipmentsData);
        console.log(`✅ Successfully loaded ${shipmentsData.length} shipments`);
        
        if (showAlerts) {
          if (hasError) {
            // اگر خطایی رخ داده، پیام خطا را نمایش بده
            setError(true);
            setMessage(errorMessage);
            
            // نمایش پیام خطا برای 3 ثانیه
            setTimeout(() => {
              setMessage("");
            }, 3000);
          } else if (databaseUpdated) {
            setMessage("دیتابیس خارجی با موفقیت دریافت شد و محموله‌ها به‌روزرسانی شدند!");
            setError(false);
            
            // نمایش پیام موفقیت برای 3 ثانیه
            setTimeout(() => {
              setMessage("");
            }, 3000);
          } else {
            setMessage(`محموله‌ها از دیتابیس قبلی بارگذاری شدند (${shipmentsData.length} محموله)`);
            setError(false);
            
            // نمایش پیام برای 3 ثانیه
            setTimeout(() => {
              setMessage("");
            }, 3000);
          }
        }
      } else {
        if (showAlerts) {
          setError(true);
          setMessage(data.message || "خطا در دریافت محموله‌ها");
          
          // نمایش پیام خطا برای 3 ثانیه
          setTimeout(() => {
            setMessage("");
          }, 3000);
        }
      }
    } catch (error) {
      console.error("❌ Error copying database:", error);
      if (showAlerts) {
        setError(true);
        setMessage("دریافت دیتابیس خارجی ناموفق بود. از دیتابیس قبلی استفاده می‌شود.");
        
        // نمایش پیام خطا برای 3 ثانیه
        setTimeout(() => {
          setMessage("");
        }, 3000);
      }
    } finally {
      if (showAlerts) {
        setStartingLoading(false);
        setLoading(false); // غیرفعال کردن اسکلتون لودینگ
      }
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <ShipmentSelectionHeader
        onBackToHome={handleBackToHome}
        onCopyDatabase={handleCopyDatabase}
        loading={loading}
        startingLoading={startingLoading}
        operationType="loading"
      />
      
      <div>
        <ShipmentSelectionContent
          pageTransitionLoading={pageTransitionLoading}
          message={message}
          error={error}
          setMessage={setMessage}
          setError={setError}
          shipments={shipments}
          loading={loading}
          selectedShipmentId={selectedShipmentId}
          onShipmentSelect={handleShipmentChange}
          onWarehouseSelect={handleWarehouseSelect}
          operationType="loading"
        />
      </div>
    </div>
  );
} 