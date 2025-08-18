import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Truck, Calendar, Scale, PackageIcon } from "lucide-react";
import HomeButton from "@/components/ui/home-button";
import AlertManager from "@/components/AlertManager";
import LoadingItemForm from "@/components/loadings/LoadingItemForm";
import ShipmentCard from "@/components/shipment-selection/ShipmentCard";

import Spinner from "@/components/Spinner";
import { API_ENDPOINTS } from "@/config";
import { useUnloadingContext } from "@/contexts/LoadingContext";

export default function LoadingEditPage() {
  const { loadingToken } = useParams();
  const navigate = useNavigate();
  const { selectedShipment: contextShipment, setSelectedShipment } = useUnloadingContext();
  const [loading, setLoading] = useState(true);
  const [editingLoading, setEditingLoading] = useState(null);
  const [items, setItems] = useState([]);
  const [products, setProducts] = useState([]);
  const [canEdit, setCanEdit] = useState(false);
  const [remainingMinutes, setRemainingMinutes] = useState(0);
  const [message, setMessage] = useState("");
  const [error, setError] = useState(false);
  const [showEditExpiredAlert, setShowEditExpiredAlert] = useState(false);
  const [showRemainingTimeAlert, setShowRemainingTimeAlert] = useState(true);
  const [shipmentDetails, setShipmentDetails] = useState(null);


  // تبدیل نام بینایی به نام فارسی
  const getPersianName = (visionName) => {
    const product = products.find(p => p.vision_name === visionName);
    return product ? product.persian_name : visionName;
  };

  // بارگذاری جزئیات محموله از API
  const loadShipmentDetails = async (shipmentId) => {
    if (!shipmentId) return;
    
    try {
      const res = await fetch(API_ENDPOINTS.SHIPMENT_DETAIL(shipmentId));
      const data = await res.json();
      
      if (data.success) {
        setShipmentDetails(data.data);
      } else {
        console.error('خطا در بارگذاری جزئیات محموله:', data.message);
      }
    } catch (error) {
      console.error('خطا در بارگذاری جزئیات محموله:', error);
    }
  };

  // بارگذاری جزئیات محموله بر اساس توکن
  const loadShipmentByToken = async () => {
    try {
      const res = await fetch(API_ENDPOINTS.LOADING_SHIPMENT_BY_TOKEN(loadingToken));
      const data = await res.json();
      
      if (data.success) {
        setShipmentDetails(data.data);
      } else {
        console.log('اطلاعات محموله در دسترس نیست:', data.message);
      }
    } catch (error) {
      console.error('خطا در بارگذاری جزئیات محموله:', error);
    }
  };

  // بارگذاری داده‌های بارگیری
  const loadLoadingData = async () => {
    try {
      console.log('🔍 Loading loading data for token:', loadingToken);
      console.log('🔗 API endpoint:', API_ENDPOINTS.LOADING_BY_TOKEN(loadingToken));
      
      const res = await fetch(API_ENDPOINTS.LOADING_BY_TOKEN(loadingToken));
      console.log('📡 API response status:', res.status);
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      console.log('📊 API response data:', data);
      
      if (data.success) {
        console.log('✅ Loading data received successfully:', data);
        setEditingLoading(data);
        setCanEdit(data.can_edit);
        setRemainingMinutes(data.remaining_minutes || 0);
        
        // اگر اطلاعات shipment در پاسخ API وجود دارد، از آن استفاده کن
        if (data.shipment_info && !contextShipment) {
          setShipmentDetails(data.shipment_info);
        } else if (data.shipment_id && !contextShipment) {
          // اگر shipment_info وجود ندارد اما shipment_id وجود دارد، از API دریافت کن
          await loadShipmentDetails(data.shipment_id);
        } else if (!contextShipment) {
          // اگر هیچ‌کدام وجود ندارد، سعی کن از توکن دریافت کن
          await loadShipmentByToken();
        }
        
        // فیلتر کردن آیتم‌ها برای نمایش آخرین نسخه
        const allItems = data.items || [];
        const latestVersion = data.version || 1;
        
        console.log('📦 All items from API:', allItems);
        console.log('🔢 Latest version:', latestVersion);
        
        // فقط آیتم‌هایی که در آخرین نسخه وجود دارند
        const latestItems = allItems
          .filter(item => item.version === latestVersion && Number(item.count) > 0)
          .map(item => ({
            name: item.name,
            source: item.source,
            version: item.version,
            reel_number: item.reel_number,
            width: item.width,
            gsm: item.gsm,
            length: item.length,
            breaks: item.breaks,
            grade: item.grade
          }));
        
        console.log('✅ Filtered latest items:', latestItems);
        setItems(latestItems);
        
        // نمایش پیام مناسب بر اساس تعداد آیتم‌ها
        if (latestItems.length === 0) {
          setMessage("بارگیری بارگذاری شد. می‌توانید آیتم‌ها را به صورت دستی اضافه کنید.");
        } else {
          setMessage(`بارگیری بارگذاری شد. ${latestItems.length} آیتم از سیستم بینایی دریافت شد.`);
        }
      } else {
        setError(true);
        setMessage(data.message || "بارگیری یافت نشد.");
      }
    } catch (error) {
      console.error('❌ Error in loadLoadingData:', error);
      setError(true);
      setMessage(`خطا در بارگذاری بارگیری: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const refreshLoadingData = async () => {
    try {
      const res = await fetch(API_ENDPOINTS.LOADING_BY_TOKEN(loadingToken));
      const data = await res.json();
      
      if (data.success) {
        console.log('🔄 Refreshed loading data:', data);
        setEditingLoading(data);
        setCanEdit(data.can_edit);
        setRemainingMinutes(data.remaining_minutes || 0);
        
        // فیلتر کردن آیتم‌ها برای نمایش آخرین نسخه
        const allItems = data.items || [];
        const latestVersion = data.version || 1;
        
        console.log('📦 Refreshed all items:', allItems);
        console.log('🔢 Refreshed latest version:', latestVersion);
        
        // فقط آیتم‌هایی که در آخرین نسخه وجود دارند
        const latestItems = allItems
          .filter(item => item.version === latestVersion && Number(item.count) > 0)
          .map(item => ({
            name: item.name,
            source: item.source,
            version: item.version,
            reel_number: item.reel_number,
            width: item.width,
            gsm: item.gsm,
            length: item.length,
            breaks: item.breaks,
            grade: item.grade
          }));
        
        console.log('✅ Refreshed filtered latest items:', latestItems);
        setItems(latestItems);
        
        // نمایش پیام مناسب بر اساس تعداد آیتم‌ها
        if (latestItems.length === 0) {
          setMessage("می‌توانید آیتم‌ها را به صورت دستی اضافه کنید.");
        } else {
          setMessage(`${latestItems.length} آیتم از سیستم بینایی دریافت شد.`);
        }
      }
    } catch (error) {
      console.error('خطا در به‌روزرسانی داده‌ها:', error);
    }
  };

  const loadProducts = async () => {
    try {
      const productsRes = await fetch(API_ENDPOINTS.PRODUCTS);
      const productsData = await productsRes.json();
      if (productsData.success) {
        setProducts(productsData.data || []);
      } else {
        console.error('خطا در بارگذاری محصولات:', productsData.error);
      }
    } catch (error) {
      console.error('خطا در بارگذاری محصولات:', error);
    }
  };

  // بازیابی داده‌ها از localStorage اگر در Context موجود نباشند
  useEffect(() => {
    if (!contextShipment) {
      const savedShipment = localStorage.getItem('selectedShipment');
      if (savedShipment) {
        setSelectedShipment(JSON.parse(savedShipment));
      }
    }
  }, [contextShipment, setSelectedShipment]);

  useEffect(() => {
    loadLoadingData();
    loadProducts();
  }, [loadingToken]);

  // نمایش alert وقتی زمان ویرایش به پایان می‌رسد - فقط وقتی که واقعاً زمان تمام شده باشد
  useEffect(() => {
    if (editingLoading && !canEdit && remainingMinutes === 0) {
      // فقط وقتی که زمان باقی‌مانده صفر باشه، alert نمایش بده
      setShowEditExpiredAlert(true);
      setShowRemainingTimeAlert(false);
    } else if (editingLoading && canEdit) {
      setShowEditExpiredAlert(false);
      setShowRemainingTimeAlert(true);
    }
  }, [editingLoading, canEdit, remainingMinutes]);

  const handleItemsChange = (newItems) => {
    setItems(newItems);
  };

  const handleAddItem = (itemData) => {
    // اضافه کردن فیلد version به آیتم جدید
    const currentVersion = editingLoading?.version || 1;
    const newVersion = currentVersion + 1;
    
    const newItem = {
      ...itemData,
      version: newVersion,
      // اضافه کردن فیلدهای ضروری
      count: 1,
      type: 'loaded',
      source: 'user'
    };
    
    console.log('Adding new item with version:', newItem);
    setItems([...items, newItem]);
  };

  const handleDeleteItem = (index) => {
    const newItems = items.filter((_, idx) => idx !== index);
    setItems(newItems);
  };

  const handleEditItem = (index, updatedItem) => {
    const newItems = [...items];
    newItems[index] = updatedItem;
    setItems(newItems);
  };

  const handleEdit = async () => {
    setLoading(true);
    setMessage("");
    setError(false);
    
    try {
      console.log('Sending edit request with:', {
        loading_token: loadingToken,
        items: items
      });
      
      const res = await fetch(API_ENDPOINTS.LOADINGS_EDIT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          loading_token: loadingToken,
          items: items 
        }),
      });
      
      console.log('Response status:', res.status);
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      console.log('Response data:', data);
      
      if (data.success) {
        setMessage(data.message);
        // آپدیت کردن وضعیت ویرایش بر اساس پاسخ سرور
        setCanEdit(data.can_edit);
        setRemainingMinutes(data.remaining_minutes || 0);
        // به‌روزرسانی داده‌ها برای اطمینان از صحت
        await refreshLoadingData();
        // انتقال به صفحه اصلی
        navigate('/');
      } else {
        setError(true);
        setMessage(data.message);
      }
    } catch (error) {
      setError(true);
      console.error('Error in handleEdit:', error);
      setMessage(`خطا در ویرایش بارگیری: ${error.message}`);
    }
    
    setLoading(false);
  };

  const handleSave = async () => {
    setLoading(true);
    setMessage("");
    setError(false);
    
    try {
      // ایجاد نسخه جدید برای تمام آیتم‌ها
      const currentVersion = editingLoading?.version || 1;
      const newVersion = currentVersion + 1;
      
      // اگر هیچ آیتمی وجود ندارد، مستقیماً به صفحه خانه منتقل شو
      if (items.length === 0) {
        console.log('📝 No items to save, redirecting to home');
        setMessage("بارگیری بدون آیتم ذخیره شد.");
        navigate('/');
        return;
      }
      
      // کپی کردن آیتم‌های موجود با نسخه جدید
      const itemsWithNewVersion = items.map(item => ({
        ...item,
        version: newVersion,
        // اضافه کردن فیلدهای ضروری که ممکن است وجود نداشته باشند
        count: 1,
        type: 'loaded'
      }));
      
      console.log('Saving items with new version:', {
        loading_token: loadingToken,
        items: itemsWithNewVersion,
        new_version: newVersion,
        current_version: currentVersion
      });
      
      const res = await fetch(API_ENDPOINTS.LOADINGS_SAVE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          items: itemsWithNewVersion,
          loading_token: loadingToken,
          version: newVersion
        }),
      });
      const data = await res.json();
      
      if (data.success) {
        setMessage(data.message);
        navigate('/');
      } else {
        setError(true);
        setMessage(data.message);
      }
    } catch (error) {
      setError(true);
      setMessage("خطا در ذخیره بارگیری");
    }
    
    setLoading(false);
  };

  const handleBackToHome = () => {
    navigate('/');
  };

  // Parse vision output JSON
  const parseVisionOutput = () => {
    if (!editingLoading?.vision_output) return null;
    
    try {
      return JSON.parse(editingLoading.vision_output);
    } catch (error) {
      console.error("Error parsing vision output:", error);
      return null;
    }
  };

  const visionData = parseVisionOutput();

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50">
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Card className="bg-white shadow-md">
            <CardContent className="p-6">
              <div className="flex flex-col items-center justify-center py-12 space-y-4">
                <Spinner className="w-12 h-12 text-blue-600" />
                <p className="text-slate-600 text-lg">در حال بارگذاری...</p>
              </div>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="sticky-header">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
            {/* دکمه خانه در سطر اول وسط در موبایل */}
            {editingLoading && editingLoading.version >= 2 && (
              <div className="flex justify-center sm:justify-end sm:hidden mb-2">
                <HomeButton 
                  onClick={handleBackToHome}
                  disabled={loading}
                />
              </div>
            )}
            
            {/* عنوان و زیرعنوان با دکمه خانه در دسکتاپ */}
            <div className="text-center sm:text-right sm:flex sm:items-center sm:gap-3">
              <div>
                <h1 className="text-2xl font-bold text-slate-900">
                  ویرایش بارگیری
                </h1>
                <p className="text-slate-600 text-sm">
                  {editingLoading && (
                    <span className="text-slate-600">
                      انبار: {editingLoading.warehouse_name || 'نامشخص'} | 
                      نسخه فعلی: {editingLoading.version || 1} | 
                      نسخه جدید: {(editingLoading.version || 1) + 1}
                    </span>
                  )}
                </p>
              </div>
              {/* دکمه خانه در کنار عنوان در دسکتاپ */}
              {editingLoading && editingLoading.version >= 2 && (
                <div className="hidden sm:block order-first">
                  <HomeButton 
                    onClick={handleBackToHome}
                    disabled={loading}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <AlertManager
          showEditExpiredAlert={showEditExpiredAlert}
          setShowEditExpiredAlert={setShowEditExpiredAlert}
          editingLoading={editingLoading}
          canEdit={canEdit}
          remainingMinutes={remainingMinutes}
          showRemainingTimeAlert={showRemainingTimeAlert}
          setShowRemainingTimeAlert={setShowRemainingTimeAlert}
          connectedToExisting={false}
          setConnectedToExisting={() => {}}
          started={false}
          message={message}
          error={error}
          setMessage={setMessage}
          setError={setError}
        />

        <div className="space-y-6">
          {/* کارت اطلاعات محموله */}
          {(contextShipment || shipmentDetails) && (
            <div className="flex justify-center">
              <ShipmentCard
                shipment={contextShipment || shipmentDetails}
                variant="default"
              />
            </div>
          )}

          {/* Loading Items Form - بدون تب */}
          <div className="space-y-6 mt-6">
            <LoadingItemForm
              items={items}
              onItemsChange={handleItemsChange}
              products={products}
              canEdit={canEdit}
              onAddItem={handleAddItem}
              onDeleteItem={handleDeleteItem}
              onEditItem={handleEditItem}
            />
          </div>

          {/* دکمه تایید و ذخیره */}
          <div className="flex justify-center pt-6">
            <Button
              onClick={handleSave}
              disabled={loading || !canEdit}
              className="bg-green-600 hover:bg-green-700 text-white px-8 py-3"
            >
              {loading ? (
                <>
                  <Spinner className="w-4 h-4 ml-2" />
                  در حال ذخیره...
                </>
              ) : (
                "تایید و ذخیره"
              )}
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}
