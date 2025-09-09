import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

import { Package, Warehouse, List, Settings, Truck, Upload, ShoppingCart, ArrowRightLeft, RotateCcw, Download } from "lucide-react";
import EditLastLoadingButton from "@/components/EditLastLoadingButton";
import { API_ENDPOINTS } from "@/config";

export default function HomePage() {
  const navigate = useNavigate();
  const [operationTypes, setOperationTypes] = useState([]);
  const [loading, setLoading] = useState(true);

  // Fetch operation types from database
  useEffect(() => {
    const fetchOperationTypes = async () => {
      try {
        const response = await fetch(API_ENDPOINTS.OPERATION_TYPES);
        const data = await response.json();
        
        if (data.success) {
          setOperationTypes(data.data);
        } else {
          console.error('Error fetching operation types:', data.error);
        }
      } catch (error) {
        console.error('Error fetching operation types:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchOperationTypes();
  }, []);

  // Get icon component by name
  const getIconComponent = (iconName) => {
    const iconMap = {
      'Truck': Truck,
      'Upload': Upload,
      'ShoppingCart': ShoppingCart,
      'ArrowRightLeft': ArrowRightLeft,
      'RotateCcw': RotateCcw,
      'Package': Package,
      'Warehouse': Warehouse,
      'List': List,
      'Download': Download
    };
    return iconMap[iconName] || Package;
  };

  // Get color classes by color name
  const getColorClasses = (colorName, isDisabled = false) => {
    if (isDisabled) {
      return {
        bg: 'bg-gray-100',
        text: 'text-gray-400'
      };
    }
    
    const colorMap = {
      'red': {
        bg: 'bg-red-100',
        text: 'text-red-600'
      },
      'blue': {
        bg: 'bg-blue-100',
        text: 'text-blue-600'
      },
      'green': {
        bg: 'bg-green-100',
        text: 'text-green-600'
      },
      'yellow': {
        bg: 'bg-yellow-100',
        text: 'text-yellow-600'
      },
      'purple': {
        bg: 'bg-purple-100',
        text: 'text-purple-600'
      },
      'pink': {
        bg: 'bg-pink-100',
        text: 'text-pink-600'
      },
      'indigo': {
        bg: 'bg-indigo-100',
        text: 'text-indigo-600'
      },
      'gray': {
        bg: 'bg-gray-100',
        text: 'text-gray-600'
      }
    };
    
    return colorMap[colorName] || colorMap['gray'];
  };

  // Handle operation button click
  const handleOperationClick = (operationType) => {
    if (!operationType.is_available) {
      return; // Don't navigate if operation is not available
    }

    switch (operationType.name) {
      case 'unloading':
        navigate('/shipment-select-unloading');
        break;
      case 'loading':
        navigate('/shipment-select-loading');
        break;
      case 'consumption':
      case 'transfer':
      case 'return':
        navigate('/warehouse-select');
        break;
      default:
        console.warn('Unknown operation type:', operationType.name);
    }
  };

  // بازگشت به خانه
  const handleBackToHome = () => {
    navigate('/');
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
            {/* آیکون در سطر اول وسط در موبایل */}
            <div className="flex justify-center sm:justify-end sm:hidden mb-2">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Package className="w-6 h-6 text-blue-600" />
              </div>
            </div>
            
            {/* عنوان و زیرعنوان با آیکون در دسکتاپ */}
            <div className="text-center sm:text-right sm:flex sm:items-center sm:gap-3">
              <div>
                <h1 className="text-2xl font-bold">
                  سیستم مدیریت بارگیری
                </h1>
                <p className="text-slate-600 text-sm">
                  مدیریت هوشمند انبارها و بارگیری
                </p>
              </div>
              {/* آیکون در کنار عنوان در دسکتاپ */}
              <div className="hidden sm:block order-first">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <Package className="w-6 h-6 text-blue-600" />
                </div>
              </div>
            </div>
            
            {/* دکمه‌ها در دسکتاپ - سمت راست */}
            <div className="hidden sm:flex items-center gap-3">
              <Button 
                onClick={() => navigate('/admin')}
                className="bg-green-600 hover:bg-green-700 border border-green-600 hover:border-green-700 transition-all duration-200"
              >
                <Settings className="h-4 w-4 ml-2" />
                پنل مدیریت
              </Button>
            </div>
          </div>
          
          {/* دکمه‌ها در موبایل - در همان سطر عنوان */}
          <div className="flex justify-center gap-2 sm:hidden mt-4">
            <Button 
              onClick={() => navigate('/admin')}
              className="bg-green-600 hover:bg-green-700 border border-green-600 hover:border-green-700 transition-all duration-200"
            >
              <Settings className="h-4 w-4 ml-2" />
              پنل مدیریت
            </Button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="space-y-6">
            {/* Dashboard Header */}
            <div className="bg-white rounded-xl shadow-md p-6">
              <div className="text-center">
                <h2 className="text-xl font-semibold text-slate-900 mb-1">
                  انتخاب عملیات
                </h2>
                <p className="text-slate-600">
                  نوع عملیات مورد نظر را انتخاب کنید
                </p>
              </div>
          </div>

          {/* Operation Buttons */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-w-3xl mx-auto">
            {loading ? (
              // Loading state
              Array.from({ length: 5 }).map((_, index) => (
                <Card key={index} className="bg-white shadow-md animate-pulse">
                  <CardContent className="px-3 py-0 text-center">
                    <div className="flex flex-col items-center space-y-0.5">
                      <div className="p-3 bg-gray-200 rounded-full w-16 h-16"></div>
                      <div className="w-20 h-4 bg-gray-200 rounded"></div>
                      <div className="w-32 h-3 bg-gray-200 rounded"></div>
                    </div>
                  </CardContent>
                </Card>
              ))
            ) : (
              // Dynamic operation buttons
              operationTypes
                .filter(op => op.is_enabled)
                .sort((a, b) => a.order - b.order)
                .map((operationType) => {
                  const IconComponent = getIconComponent(operationType.icon);
                  const isDisabled = !operationType.is_available;
                  const { bg, text } = getColorClasses(operationType.color, isDisabled);
                  
                  return (
                    <Card 
                      key={operationType.id}
                      className={`bg-white shadow-md hover:shadow-lg transition-all duration-200 cursor-pointer ${
                        isDisabled ? 'opacity-50 cursor-not-allowed' : ''
                      }`} 
                      onClick={() => handleOperationClick(operationType)}
                    >
                      <CardContent className="px-3 py-0 text-center">
                        <div className="flex flex-col items-center space-y-0.5">
                          <div className={`p-3 rounded-full ${bg}`}>
                            <IconComponent 
                              className={`w-12 h-12 ${text}`} 
                            />
                          </div>
                          {/* Separator line between icon and title */}
                          <div className="w-16 h-px bg-gray-300 my-2"></div>
                          <div>
                            <h3 className={`text-base font-semibold mb-0 ${
                              isDisabled ? 'text-gray-500' : 'text-slate-900'
                            }`}>
                              {operationType.persian_name}
                            </h3>
                            <p className={`text-xs ${
                              isDisabled ? 'text-gray-400' : 'text-slate-600'
                            }`}>
                              {operationType.description}
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })
            )}
          </div>

          {/* Edit Last Loading Button */}
          <div className="mt-8 max-w-3xl mx-auto">
            <EditLastLoadingButton
              onClick={async () => {
                try {
                  const res = await fetch(API_ENDPOINTS.OPERATIONS_LAST_COMPLETED);
                  const data = await res.json();
                  
                  if (data.success) {
                    if (data.status === 'completed' || data.status === 'vision' || data.status === 'edited') {
                      // اگر اطلاعات shipment در پاسخ وجود دارد، آن را در context ذخیره کن
                      if (data.shipment_info) {
                        // اینجا نمی‌توانیم مستقیماً context را به‌روزرسانی کنیم
                        // چون در HomePage هستیم و context در EditPage استفاده می‌شود
                        // اطلاعات shipment در EditPage از API دریافت خواهد شد
                      }
                      // تشخیص نوع عملیات و هدایت به صفحه ویرایش مناسب
                      if (data.type === 'loading') {
                        navigate(`/loading-edit/${data.token}`);
                      } else {
                        navigate(`/unloading-edit/${data.token}`);
                      }
                    } else {
                      // اگر عملیات قابل ویرایش نیست، به لیست عملیات برو
                      if (data.type === 'loading') {
                        navigate('/loadings');
                      } else {
                        navigate('/unloadings');
                      }
                    }
                  } else {
                    // اگر عملیاتی یافت نشد، به صفحه انتخاب انبار برو
                    navigate('/warehouse-selection');
                  }
                } catch (error) {
                  console.error('خطا در بارگذاری آخرین عملیات:', error);
                  navigate('/warehouse-selection');
                }
              }}
            />
          </div>
        </div>
      </main>
    </div>
  );
} 