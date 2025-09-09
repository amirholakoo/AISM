import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Truck, Calendar, Scale } from "lucide-react";
import HomeButton from "@/components/ui/home-button";
import AlertManager from "@/components/AlertManager";
import ItemsTable from "@/components/ItemsTable";
import ShipmentCard from "@/components/shipment-selection/ShipmentCard";

import Spinner from "@/components/Spinner";
import { API_ENDPOINTS } from "@/config";
import { useUnloadingContext } from "@/contexts/LoadingContext";

export default function EditPage() {
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
  const [operationType, setOperationType] = useState(null); // 'loading' or 'unloading'

  // تبدیل نام بینایی به نام فارسی
  const getPersianName = (visionName) => {
    const product = products.find(p => p.vision_name === visionName);
    return product ? product.persian_name : visionName;
  };

  // تشخیص نوع عملیات بر اساس پاسخ API
  const detectOperationType = (data) => {
    // اگر در پاسخ API فیلد type وجود دارد، از آن استفاده کن
    if (data.type) {
      return data.type;
    }
    
    // اگر warehouse_id وجود دارد، احتمالاً تخلیه است
    if (data.warehouse_id) {
      return 'unloading';
    }
    
    // پیش‌فرض تخلیه
    return 'unloading';
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
      let res;
      if (operationType === 'loading') {
        res = await fetch(API_ENDPOINTS.LOADING_SHIPMENT_BY_TOKEN(loadingToken));
      } else {
        res = await fetch(API_ENDPOINTS.UNLOADING_SHIPMENT_BY_TOKEN(loadingToken));
      }
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

  // بارگذاری داده‌های بارگیری/تخلیه
  const loadLoadingData = async () => {
    try {
      let res;
      // ابتدا سعی کن با endpoint تخلیه
      try {
        res = await fetch(API_ENDPOINTS.UNLOADING_BY_TOKEN(loadingToken));
        const data = await res.json();
        
        if (data.success) {
          setOperationType('unloading');
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
          
          // فقط آیتم‌هایی که در آخرین نسخه وجود دارند
          const latestItems = allItems
            .filter(item => item.version === latestVersion && Number(item.count) > 0)
            .map(item => ({
              name: item.name,
              type: item.type,
              count: item.count,
              source: item.source,
              version: item.version
            }));
          
          setItems(latestItems);
          setMessage("تخلیه بارگذاری شد.");
          return;
        }
      } catch (error) {
        console.log('خطا در بارگذاری تخلیه، سعی در بارگذاری بارگیری...');
      }
      
      // اگر تخلیه موفق نبود، سعی کن بارگیری
      try {
        res = await fetch(API_ENDPOINTS.LOADING_BY_TOKEN(loadingToken));
        const data = await res.json();
        
        if (data.success) {
          setOperationType('loading');
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
          
          // فقط آیتم‌هایی که در آخرین نسخه وجود دارند
          const latestItems = allItems
            .filter(item => item.version === latestVersion && Number(item.count) > 0)
            .map(item => ({
              name: item.name,
              type: item.type,
              count: item.count,
              source: item.source,
              version: item.version
            }));
          
          setItems(latestItems);
          setMessage("بارگیری بارگذاری شد.");
          return;
        }
      } catch (error) {
        console.log('خطا در بارگذاری بارگیری');
      }
      
      // اگر هیچ‌کدام موفق نبود
      setError(true);
      setMessage("بارگیری/تخلیه یافت نشد.");
    } catch (error) {
      setError(true);
      setMessage("خطا در بارگذاری بارگیری/تخلیه");
    } finally {
      setLoading(false);
    }
  };

  const refreshLoadingData = async () => {
    try {
      let res;
      if (operationType === 'loading') {
        res = await fetch(API_ENDPOINTS.LOADING_BY_TOKEN(loadingToken));
      } else {
        res = await fetch(API_ENDPOINTS.UNLOADING_BY_TOKEN(loadingToken));
      }
      const data = await res.json();
      
      if (data.success) {
        setEditingLoading(data);
        setCanEdit(data.can_edit);
        setRemainingMinutes(data.remaining_minutes || 0);
        
        // فیلتر کردن آیتم‌ها برای نمایش آخرین نسخه (همان منطق loadLoadingData)
        const allItems = data.items || [];
        const latestVersion = data.version || 1;
        
        // فقط آیتم‌هایی که در آخرین نسخه وجود دارند
        const latestItems = allItems
          .filter(item => item.version === latestVersion && Number(item.count) > 0)
          .map(item => ({
            name: item.name,
            type: item.type,
            count: item.count,
            source: item.source,
            version: item.version
          }));
        
        setItems(latestItems);
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

  const handleItemChange = (idx, value) => {
    const newItems = [...items];
    // اگر مقدار خالی باشد، آن را 0 قرار نده، بلکه همان خالی بگذار
    newItems[idx].count = value === '' ? '' : value;
    setItems(newItems);
  };

  const handleDeleteItem = (name, type) => {
    const newItems = items.filter(item => 
      !(item.name === name && item.type === type)
    );
    setItems(newItems);
  };

  const handleRestoreItem = (name, type) => {
    const newItems = items.map(item =>
      item.name === name && item.type === type ? { ...item, count: 1 } : item
    );
    setItems(newItems);
  };

  const handleAddItem = (type, productName) => {
    const existingItem = items.find(item => item.name === productName && item.type === type);
    if (!existingItem) {
      setItems([...items, { name: productName, type, count: 1 }]);
    } else {
      const newItems = items.map(item =>
        item.name === productName && item.type === type ? { ...item, count: Number(item.count) + 1 } : item
      );
      setItems(newItems);
    }
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
      
      let res;
      if (operationType === 'loading') {
        res = await fetch(API_ENDPOINTS.LOADINGS_EDIT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            loading_token: loadingToken,
            items: items 
          }),
        });
      } else {
        res = await fetch(API_ENDPOINTS.UNLOADINGS_EDIT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            unloading_token: loadingToken,
            items: items 
          }),
        });
      }
      
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
      setMessage(`خطا در ویرایش ${operationType === 'loading' ? 'بارگیری' : 'تخلیه'}: ${error.message}`);
    }
    
    setLoading(false);
  };

  const handleSave = async () => {
    setLoading(true);
    setMessage("");
    setError(false);
    
    try {
      let res;
      if (operationType === 'loading') {
        res = await fetch(API_ENDPOINTS.LOADINGS_SAVE, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            items: items,
            loading_token: loadingToken,
          }),
        });
      } else {
        res = await fetch(API_ENDPOINTS.UNLOADINGS_SAVE, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            items: items,
            unloading_token: loadingToken,
            warehouse_id: editingLoading?.warehouse_id,
          }),
        });
      }
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
      setMessage(`خطا در ذخیره ${operationType === 'loading' ? 'بارگیری' : 'تخلیه'}`);
    }
    
    setLoading(false);
  };

  const handleBackToHome = () => {
    navigate('/');
  };

  // تابع‌های کمکی برای فرمت کردن

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
                  ویرایش {operationType === 'loading' ? 'بارگیری' : 'تخلیه'}
                </h1>
                <p className="text-slate-600 text-sm">
                  {editingLoading && (
                    <span className="text-slate-600">
                      انبار: {editingLoading.warehouse_name || 'نامشخص'}
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

          <ItemsTable
            items={items}
            products={products}
            editingLoading={editingLoading}
            loading={loading}
            canEdit={canEdit}
            onItemChange={handleItemChange}
            onDeleteItem={handleDeleteItem}
            onRestoreItem={handleRestoreItem}
            onAddItem={handleAddItem}
            onEdit={handleEdit}
            onSave={handleSave}
            showEditExpiredAlert={showEditExpiredAlert}
            getPersianName={getPersianName}
          />
        </div>
      </main>
    </div>
  );
} 