import React, { useState, useEffect } from "react";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger 
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Package, Truck, Calendar, Scale, User, Building, FileText } from "lucide-react";
import { API_ENDPOINTS } from "@/config";

export default function ShipmentDetailsModal({ shipmentId, children }) {
  const [shipment, setShipment] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  const loadShipmentDetails = async () => {
    if (!shipmentId) return;
    
    setLoading(true);
    setError(false);
    
    try {
      const res = await fetch(API_ENDPOINTS.SHIPMENT_DETAIL(shipmentId));
      const data = await res.json();
      
      if (data.success) {
        setShipment(data.data);
      } else {
        setError(true);
      }
    } catch (error) {
      setError(true);
    } finally {
      setLoading(false);
    }
  };

  // بارگذاری جزئیات محموله وقتی مودال باز می‌شود
  useEffect(() => {
    if (isOpen && shipmentId && !shipment) {
      loadShipmentDetails();
    }
  }, [isOpen, shipmentId, shipment]);

  const formatDate = (dateString) => {
    if (!dateString) return "نامشخص";
    return new Date(dateString).toLocaleDateString('fa-IR');
  };

  const formatTime = (dateString) => {
    if (!dateString) return "نامشخص";
    return new Date(dateString).toLocaleTimeString('fa-IR', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const formatNumber = (number) => {
    if (number === null || number === undefined) return "نامشخص";
    // تبدیل به عدد صحیح برای حذف اعشار اضافی (مثل .00) که ممکن است از API بیاید
    return new Intl.NumberFormat('fa-IR').format(Number(number));
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        {children}
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto [&>button]:right-auto [&>button]:left-4">
        <DialogHeader>
          <DialogTitle className="text-xl font-bold text-gray-800 flex items-center gap-2">
            <Package className="w-5 h-5" />
            جزئیات محموله
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-6">
          {loading && (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
              <p className="text-gray-600 mt-2">در حال بارگذاری...</p>
            </div>
          )}
          
          {error && (
            <div className="text-center py-8">
              <p className="text-red-600">خطا در بارگذاری جزئیات محموله</p>
            </div>
          )}
          
          {shipment && !loading && (
            <div className="space-y-4">
              {/* اطلاعات اصلی */}
              <div className="space-y-2">
                <h4 className="text-md font-semibold text-blue-700 border-b border-blue-200 pb-1">
                  اطلاعات اصلی
                </h4>
                <div className="space-y-1">
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <div className="flex items-center gap-2">
                      <Truck className="w-4 h-4 text-blue-600" />
                      <span className="font-medium text-gray-700">شماره پلاک:</span>
                    </div>
                    <span className="text-gray-900 text-sm">{shipment.license_number}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <div className="flex items-center gap-2">
                      <Package className="w-4 h-4 text-blue-600" />
                      <span className="font-medium text-gray-700">وضعیت:</span>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      shipment.status === 'Loaded' 
                        ? 'bg-green-100 text-green-800 border border-green-200' 
                        : shipment.status === 'Unloaded'
                        ? 'bg-yellow-100 text-yellow-800 border border-yellow-200'
                        : 'bg-gray-100 text-gray-800 border border-gray-200'
                    }`}>
                      {shipment.status === 'Loaded' ? 'بارگیری شده' : 
                       shipment.status === 'Unloaded' ? 'بارگیری نشده' : 
                       shipment.status || 'نامشخص'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-blue-600" />
                      <span className="font-medium text-gray-700">تاریخ:</span>
                    </div>
                    <span className="text-gray-900 text-sm">{formatDate(shipment.date)}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-blue-600" />
                      <span className="font-medium text-gray-700">تاریخ دریافت:</span>
                    </div>
                    <span className="text-gray-900 text-sm">{formatDate(shipment.receive_date)}</span>
                  </div>
                </div>
              </div>

              {/* اطلاعات طرفین */}
              <div className="space-y-2">
                <h4 className="text-md font-semibold text-blue-700 border-b border-blue-200 pb-1">
                  اطلاعات طرفین
                </h4>
                <div className="space-y-1">
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">تامین‌کننده:</span>
                    <span className="text-gray-900 text-sm">{shipment.supplier_name || "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">مشتری:</span>
                    <span className="text-gray-900 text-sm">{shipment.customer_name || "نامشخص"}</span>
                  </div>
                </div>
              </div>

              {/* اطلاعات ماده */}
              <div className="space-y-2">
                <h4 className="text-md font-semibold text-blue-700 border-b border-blue-200 pb-1">
                  اطلاعات ماده
                </h4>
                <div className="space-y-1">
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">نوع ماده:</span>
                    <span className="text-gray-900 text-sm">{shipment.material_type || "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">نام ماده:</span>
                    <span className="text-gray-900 text-sm">{shipment.material_name || "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">مقدار:</span>
                    <span className="text-gray-900 text-sm">
                      {formatNumber(shipment.quantity)} {shipment.unit || ''}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">کیفیت:</span>
                    <span className="text-gray-900 text-sm">{shipment.quality || "نامشخص"}</span>
                  </div>
                </div>
              </div>
              
                              {/* اطلاعات وزن */}
                <div className="space-y-2">
                  <h4 className="text-md font-semibold text-blue-700 border-b border-blue-200 pb-1">
                    اطلاعات وزن
                  </h4>
                  <div className="space-y-1">
                    <div className="flex justify-between items-center py-1 border-b border-gray-100">
                      <div className="flex items-center gap-2">
                        <Scale className="w-4 h-4 text-blue-600" />
                        <span className="font-medium text-gray-700">وزن اول:</span>
                      </div>
                      <span className="text-gray-900 text-sm">{formatNumber(shipment.weight1)} کیلوگرم</span>
                    </div>
                    
                    <div className="flex justify-between items-center py-1 border-b border-gray-100">
                      <div className="flex items-center gap-2">
                        <Scale className="w-4 h-4 text-blue-600" />
                        <span className="font-medium text-gray-700">وزن دوم:</span>
                      </div>
                      <span className="text-gray-900 text-sm">{formatNumber(shipment.weight2)} کیلوگرم</span>
                    </div>
                    
                    <div className="flex justify-between items-center py-1 border-b border-gray-100">
                      <div className="flex items-center gap-2">
                        <Scale className="w-4 h-4 text-blue-600" />
                        <span className="font-medium text-gray-700">وزن خالص:</span>
                      </div>
                      <span className="text-gray-900 text-sm font-semibold">{formatNumber(shipment.net_weight)} کیلوگرم</span>
                    </div>
                  </div>
                </div>
              
              {/* اطلاعات زمانی */}
              <div className="space-y-2">
                <h4 className="text-md font-semibold text-blue-700 border-b border-blue-200 pb-1">
                  اطلاعات زمانی
                </h4>
                <div className="space-y-1">
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-blue-600" />
                      <span className="font-medium text-gray-700">زمان ورود:</span>
                    </div>
                    <span className="text-gray-900 text-sm">{shipment.entry_time ? formatTime(shipment.entry_time) : "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <div className="flex items-center gap-2">
                      <Scale className="w-4 h-4 text-blue-600" />
                      <span className="font-medium text-gray-700">زمان وزن اول:</span>
                    </div>
                    <span className="text-gray-900 text-sm">{shipment.weight1_time ? formatTime(shipment.weight1_time) : "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <div className="flex items-center gap-2">
                      <Scale className="w-4 h-4 text-blue-600" />
                      <span className="font-medium text-gray-700">زمان وزن دوم:</span>
                    </div>
                    <span className="text-gray-900 text-sm">{shipment.weight2_time ? formatTime(shipment.weight2_time) : "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-blue-600" />
                      <span className="font-medium text-gray-700">زمان خروج:</span>
                    </div>
                    <span className="text-gray-900 text-sm">{shipment.exit_time ? formatTime(shipment.exit_time) : "نامشخص"}</span>
                  </div>
                </div>
              </div>
              
              {/* اطلاعات مالی */}
              <div className="space-y-2">
                <h4 className="text-md font-semibold text-blue-700 border-b border-blue-200 pb-1">
                  اطلاعات مالی
                </h4>
                <div className="space-y-1">
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">قیمت هر کیلو:</span>
                    <span className="text-gray-900 text-sm">{formatNumber(shipment.price_per_kg)} تومان</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">قیمت کل:</span>
                    <span className="text-gray-900 text-sm font-semibold">{formatNumber(shipment.total_price)} تومان</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">هزینه اضافی:</span>
                    <span className="text-gray-900 text-sm">{formatNumber(shipment.extra_cost)} تومان</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">مالیات:</span>
                    <span className="text-gray-900 text-sm">{formatNumber(shipment.vat)} تومان</span>
                  </div>
                </div>
              </div>
              
              {/* اطلاعات فنی */}
              <div className="space-y-2">
                <h4 className="text-md font-semibold text-blue-700 border-b border-blue-200 pb-1">
                  اطلاعات فنی
                </h4>
                <div className="space-y-1">
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">نوع محموله:</span>
                    <span className="text-gray-900 text-sm">{shipment.shipment_type || "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">کیفیت:</span>
                    <span className="text-gray-900 text-sm">{shipment.quality || "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">جریمه:</span>
                    <span className="text-gray-900 text-sm">{shipment.penalty || "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">محل تخلیه:</span>
                    <span className="text-gray-900 text-sm">{shipment.unload_location || "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">نام پروفیل:</span>
                    <span className="text-gray-900 text-sm">{shipment.profile_name || "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">عرض:</span>
                    <span className="text-gray-900 text-sm">{formatNumber(shipment.width)} میلی‌متر</span>
                  </div>
                </div>
              </div>

              {/* وضعیت‌ها */}
              <div className="space-y-2">
                <h4 className="text-md font-semibold text-blue-700 border-b border-blue-200 pb-1">
                  وضعیت‌ها
                </h4>
                <div className="space-y-1">
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">وضعیت فاکتور:</span>
                    <span className="text-gray-900 text-sm">{shipment.invoice_status || "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">وضعیت پرداخت:</span>
                    <span className="text-gray-900 text-sm">{shipment.payment_status || "نامشخص"}</span>
                  </div>
                </div>
              </div>

              {/* اطلاعات تکمیلی */}
              <div className="space-y-2">
                <h4 className="text-md font-semibold text-blue-700 border-b border-blue-200 pb-1">
                  اطلاعات تکمیلی
                </h4>
                <div className="space-y-1">
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">لیست رول‌ها:</span>
                    <span className="text-gray-900 text-sm">{shipment.list_of_reels || "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">اطلاعات اسناد:</span>
                    <span className="text-gray-900 text-sm">{shipment.document_info || "نامشخص"}</span>
                  </div>
                  
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">توضیحات:</span>
                    <span className="text-gray-900 text-sm">{shipment.comments || "نامشخص"}</span>
                  </div>
                </div>
              </div>

              {/* اطلاعات متفرقه */}
              <div className="space-y-2">
                <h4 className="text-md font-semibold text-blue-700 border-b border-blue-200 pb-1">
                  اطلاعات متفرقه
                </h4>
                <div className="space-y-1">
                  
                  {/* شناسه محموله */}
                  <div className="flex justify-between items-center py-1 border-b border-gray-100">
                    <span className="font-medium text-gray-700">شناسه محموله:</span>
                    <span className="text-gray-900 text-sm">{shipment.id}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
} 