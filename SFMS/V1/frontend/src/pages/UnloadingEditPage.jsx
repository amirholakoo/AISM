import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Truck, Calendar, Scale } from "lucide-react";
import HomeButton from "@/components/ui/home-button";
import AlertManager from "@/components/AlertManager";
import ItemsTable from "@/components/ItemsTable";
import ShipmentCard from "@/components/shipment-selection/ShipmentCard";
import JsonDisplayBox from "@/components/JsonDisplayBox";

import Spinner from "@/components/Spinner";
import { API_ENDPOINTS } from "@/config";
import { useUnloadingContext } from "@/contexts/LoadingContext";

export default function UnloadingEditPage() {
  const { unloadingToken } = useParams();
  const navigate = useNavigate();
  const { selectedShipment: contextShipment, setSelectedShipment } = useUnloadingContext();
  const [loading, setLoading] = useState(true);
  const [editingUnloading, setEditingUnloading] = useState(null);
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
      const res = await fetch(API_ENDPOINTS.UNLOADING_SHIPMENT_BY_TOKEN(unloadingToken));
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

  // بارگذاری داده‌های تخلیه
  const loadUnloadingData = async () => {
    try {
      const res = await fetch(API_ENDPOINTS.UNLOADING_BY_TOKEN(unloadingToken));
      const data = await res.json();
      
      console.log('LoadUnloadingData response:', data);
      console.log('LoadUnloadingData data.data:', data.data);
      console.log('LoadUnloadingData data.data.items:', data.data?.items);
      
      if (data.success) {
        console.log('Setting editingUnloading to:', data.data);
        setEditingUnloading(data.data);
        setCanEdit(data.data.can_edit);
        setRemainingMinutes(data.data.remaining_minutes || 0);
        
        // اگر اطلاعات shipment در پاسخ API وجود دارد، از آن استفاده کن
        if (data.data.shipment_info && !contextShipment) {
          setShipmentDetails(data.data.shipment_info);
        } else if (data.data.shipment_id && !contextShipment) {
          // اگر shipment_info وجود ندارد اما shipment_id وجود دارد، از API دریافت کن
          await loadShipmentDetails(data.data.shipment_id);
        } else if (!contextShipment) {
          // اگر هیچ‌کدام وجود ندارد، سعی کن از توکن دریافت کن
          await loadShipmentByToken();
        }
        
        // فیلتر کردن آیتم‌ها برای نمایش آخرین نسخه
        const allItems = data.data.items || [];
        const latestVersion = data.data.version || 1;
        
        console.log('DEBUG: allItems:', allItems);
        console.log('DEBUG: latestVersion:', latestVersion);
        
        // نمایش همه آیتم‌ها در آخرین نسخه (شامل آیتم‌های با count 0)
        const latestItems = allItems
          .filter(item => item.version === latestVersion)
          .map(item => ({
            name: item.name,
            type: item.type,
            count: item.count,
            source: item.source,
            version: item.version
          }));
        
        console.log('DEBUG: latestItems:', latestItems);
        setItems(latestItems);
        setMessage("تخلیه بارگذاری شد.");
      } else {
        setError(true);
        setMessage(data.message || "تخلیه یافت نشد.");
      }
    } catch (error) {
      setError(true);
      setMessage("خطا در بارگذاری تخلیه");
    } finally {
      setLoading(false);
    }
  };

  const refreshUnloadingData = async () => {
    try {
      const res = await fetch(API_ENDPOINTS.UNLOADING_BY_TOKEN(unloadingToken));
      const data = await res.json();
      
      if (data.success) {
        setEditingUnloading(data.data);
        setCanEdit(data.data.can_edit);
        setRemainingMinutes(data.data.remaining_minutes || 0);
        
        // فیلتر کردن آیتم‌ها برای نمایش آخرین نسخه
        const allItems = data.data.items || [];
        const latestVersion = data.data.version || 1;
        
        console.log('DEBUG refreshUnloadingData: allItems:', allItems);
        console.log('DEBUG refreshUnloadingData: latestVersion:', latestVersion);
        
        // نمایش همه آیتم‌ها در آخرین نسخه (شامل آیتم‌های با count 0)
        const latestItems = allItems
          .filter(item => item.version === latestVersion)
          .map(item => ({
            name: item.name,
            type: item.type,
            count: item.count,
            source: item.source,
            version: item.version
          }));
        
        console.log('DEBUG refreshUnloadingData: latestItems:', latestItems);
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
    loadUnloadingData();
    loadProducts();
  }, [unloadingToken]);

  // نمایش alert وقتی زمان ویرایش به پایان می‌رسد
  useEffect(() => {
    if (editingUnloading && !canEdit && remainingMinutes === 0) {
      // فقط وقتی که زمان باقی‌مانده صفر باشه، alert نمایش بده
      setShowEditExpiredAlert(true);
      setShowRemainingTimeAlert(false);
    } else if (editingUnloading && canEdit) {
      setShowEditExpiredAlert(false);
    }
  }, [editingUnloading, canEdit, remainingMinutes]);

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
      console.log('DEBUG: Sending edit request with:', {
        unloading_token: unloadingToken,
        items: items,
        items_count: items.length
      });
      
      // Validate items before sending
      const validItems = items.filter(item => {
        const count = item.count;
        const isValid = count !== '' && count !== null && count !== undefined;
        if (!isValid) {
          console.log('DEBUG: Skipping invalid item:', item);
        }
        return isValid;
      });
      
      console.log('DEBUG: Valid items to send:', validItems);
      
      const res = await fetch(API_ENDPOINTS.UNLOADINGS_EDIT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          unloading_token: unloadingToken,
          items: validItems 
        }),
      });
      
      console.log('DEBUG: Response status:', res.status);
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      console.log('DEBUG: Response data:', data);
      
      if (data.success) {
        setMessage(data.message);
        // آپدیت کردن وضعیت ویرایش بر اساس پاسخ سرور
        setCanEdit(data.can_edit);
        setRemainingMinutes(data.remaining_minutes || 0);
        // به‌روزرسانی داده‌ها برای اطمینان از صحت
        await refreshUnloadingData();
        // انتقال به صفحه اصلی
        navigate('/');
      } else {
        setError(true);
        setMessage(data.message);
      }
    } catch (error) {
      setError(true);
      console.error('DEBUG: Error in handleEdit:', error);
      setMessage(`خطا در ویرایش تخلیه: ${error.message}`);
    }
    
    setLoading(false);
  };

  const handleSave = async () => {
    setLoading(true);
    setMessage("");
    setError(false);
    
    try {
      console.log('DEBUG: Sending save request with:', {
        items: items,
        items_count: items.length,
        unloading_token: unloadingToken,
        warehouse_id: editingUnloading?.warehouse_id,
        editingUnloading: editingUnloading
      });
      
      // Validate items before sending
      const validItems = items.filter(item => {
        const count = item.count;
        const isValid = count !== '' && count !== null && count !== undefined && Number(count) > 0;
        if (!isValid) {
          console.log('DEBUG: Skipping invalid item for save:', item);
        }
        return isValid;
      });
      
      console.log('DEBUG: Valid items to save:', validItems);
      
      const res = await fetch(API_ENDPOINTS.UNLOADINGS_SAVE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          items: validItems,
          unloading_token: unloadingToken,
          warehouse_id: editingUnloading?.warehouse_id,
        }),
      });
      
      console.log('DEBUG: Save response status:', res.status);
      const data = await res.json();
      console.log('DEBUG: Save response data:', data);
      
      if (data.success) {
        setMessage(data.message);
        navigate('/');
      } else {
        setError(true);
        setMessage(data.message);
      }
    } catch (error) {
      setError(true);
      console.error('DEBUG: Error in handleSave:', error);
      setMessage("خطا در ذخیره تخلیه");
    }
    
    setLoading(false);
  };

  const handleBackToHome = () => {
    navigate('/');
  };



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
            {editingUnloading && editingUnloading.version >= 2 && (
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
                  ویرایش تخلیه
                </h1>
                <p className="text-slate-600 text-sm">
                  {editingUnloading && (
                    <span className="text-slate-600">
                      انبار: {editingUnloading.warehouse_name || 'نامشخص'}
                    </span>
                  )}
                </p>
              </div>
              {/* دکمه خانه در کنار عنوان در دسکتاپ */}
              {editingUnloading && editingUnloading.version >= 2 && (
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
          editingLoading={editingUnloading}
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
            editingLoading={editingUnloading}
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
