import React, { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger 
} from "@/components/ui/dialog";
import { 
  List, 
  Truck, 
  Upload, 
  Download,
  ShoppingCart, 
  ArrowRightLeft,
  Package,
  RotateCcw
} from "lucide-react";
import AlertManager from "@/components/AlertManager";
import Spinner from "@/components/Spinner";
import { API_ENDPOINTS, API_BASE_URL } from "@/config";
import {
  UnloadingList,
  UnloadingDetailsModal,
  UnloadingPagination,
  getStatusColor,
  getStatusText,
  getStatusIcon,
  getTypeIcon,
  getVersionText,
  getSourceBadgeColor,
  getGroupTitle,
  groupItemsByVersionAndSource,
  formatDate,
  formatTime,
  getOperationTypeText,
  extractIdFromToken
} from "@/components/unloadings";

export default function OperationsListPage() {
  const navigate = useNavigate();
  const [unloadings, setUnloadings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [error, setError] = useState(false);
  const [selectedUnloading, setSelectedUnloading] = useState(null);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [warehouses, setWarehouses] = useState([]);
  const [products, setProducts] = useState([]);
  const [operationTypes, setOperationTypes] = useState([]);
  const [selectedOperationType, setSelectedOperationType] = useState('unloading');
  const [pagination, setPagination] = useState({
    page: 1,
    per_page: 10, // نمایش 10 مورد در هر صفحه (مطابق با backend)
    total: 0,
    pages: 0,
    has_next: false,
    has_prev: false
  });
  const currentPageRef = useRef(1);

  // دریافت لیست تخلیه‌ها
  useEffect(() => {
    loadData(currentPageRef.current);
  }, []);

  // ریست کردن صفحه‌بندی وقتی نوع عملیات تغییر می‌کند
  useEffect(() => {
    currentPageRef.current = 1;
    // Reset pagination state when operation type changes
    setPagination(prev => ({
      ...prev,
      page: 1,
      total: 0,
      pages: 0,
      has_next: false,
      has_prev: false
    }));
    loadData(1);
  }, [selectedOperationType]);

  const loadData = async (page = 1) => {
    try {
      setLoading(true);
      currentPageRef.current = page;
      
      // دریافت لیست انبارها (فقط یک بار)
      if (warehouses.length === 0) {
        const warehousesRes = await fetch(API_ENDPOINTS.WAREHOUSES);
        const warehousesData = await warehousesRes.json();
        if (warehousesData.success) {
          setWarehouses(warehousesData.warehouses || []);
        }
      }
      
      // دریافت لیست محصولات (فقط یک بار)
      if (products.length === 0) {
        const productsRes = await fetch(API_ENDPOINTS.PRODUCTS);
        const productsData = await productsRes.json();
        if (productsData.success) {
          setProducts(productsData.data || []);
        }
      }
      
      // دریافت لیست انواع عملیات (فقط یک بار)
      if (operationTypes.length === 0) {
        const operationTypesRes = await fetch(API_ENDPOINTS.OPERATION_TYPES);
        const operationTypesData = await operationTypesRes.json();
        if (operationTypesData.success) {
          setOperationTypes(operationTypesData.data || []);
        }
      }
      
      // دریافت لیست عملیات با صفحه‌بندی
      const perPage = pagination.per_page;
      const apiUrl = `${API_ENDPOINTS.OPERATIONS_ALL}?page=${page}&per_page=${perPage}&operation_type=${selectedOperationType}`;
      
      const operationsRes = await fetch(apiUrl);
      const operationsData = await operationsRes.json();
      
      if (operationsData.success) {
        setUnloadings(operationsData.operations || []);
        // Update pagination state with the response data
        const newPagination = {
          page: page,
          per_page: perPage,
          total: operationsData.pagination?.total || 0,
          pages: operationsData.pagination?.pages || 0,
          has_next: operationsData.pagination?.has_next || false,
          has_prev: operationsData.pagination?.has_prev || false,
          next_num: operationsData.pagination?.next_num || null,
          prev_num: operationsData.pagination?.prev_num || null
        };
        
        setPagination(newPagination);
      } else {
        setError(true);
        setMessage(operationsData.message || "خطا در دریافت لیست عملیات");
        // Reset pagination on error
        setPagination(prev => ({
          ...prev,
          page: 1,
          total: 0,
          pages: 0,
          has_next: false,
          has_prev: false
        }));
      }
    } catch (error) {
      console.error("❌ Error loading data:", error);
      setError(true);
      setMessage("خطا در اتصال به سرور");
      // Reset pagination on error
      setPagination(prev => ({
        ...prev,
        page: 1,
        total: 0,
        pages: 0,
        has_next: false,
        has_prev: false
      }));
    } finally {
      setLoading(false);
    }
  };

  // بازگشت به خانه
  const handleBackToAdmin = () => {
    navigate('/admin');
  };

  // نمایش جزئیات عملیات
  const handleShowDetails = async (operation) => {
    try {
      // تعیین endpoint بر اساس نوع عملیات
      let endpoint;
      if (operation.type === 'loading') {
        endpoint = `${API_BASE_URL}/api/loadings/${operation.id}/items?all_versions=true`;
      } else {
        // برای unloading از endpoint جدید استفاده کن
        endpoint = `${API_BASE_URL}/api/unloadings/${operation.id}/items?all_versions=true`;
      }
      
      const operationRes = await fetch(endpoint);
      const operationData = await operationRes.json();
      
      if (operationData.success) {
        setSelectedUnloading({
          ...operation,
          items: operationData.items || [],
          start_time: operationData.start_time,
          end_time: operationData.end_time,
          user_confirm_time: operationData.user_confirm_time,
          edit_time: operationData.edit_time,
          vision_output: operationData.vision_output
        });
      } else {
        setSelectedUnloading(operation);
      }
      setShowDetailsModal(true);
    } catch (error) {
      console.error("❌ Error loading items:", error);
      setSelectedUnloading(operation);
      setShowDetailsModal(true);
    }
  };

  // بستن مودال
  const handleCloseModal = () => {
    setShowDetailsModal(false);
    setSelectedUnloading(null);
  };

  // تغییر صفحه
  const handlePageChange = (newPage) => {
    loadData(newPage);
  };

  // صفحه بعدی
  const handleNextPage = () => {
    if (pagination.has_next) {
      const nextPage = pagination.page + 1;
      handlePageChange(nextPage);
    }
  };

  // صفحه قبلی
  const handlePrevPage = () => {
    if (pagination.has_prev) {
      const prevPage = pagination.page - 1;
      handlePageChange(prevPage);
    }
  };

  // دریافت نام انبار
  const getWarehouseName = (warehouseId) => {
    const warehouse = warehouses.find(w => w.id === warehouseId);
    return warehouse ? (warehouse.persian_name || warehouse.name) : `انبار ${warehouseId}`;
  };

  // تبدیل نام بینایی به نام فارسی
  const getPersianName = (visionName) => {
    const product = products.find(p => p.vision_name === visionName);
    return product ? product.persian_name : visionName;
  };

  // تبدیل نام آیکون به React component
  const getIconComponent = (iconName) => {
    const iconMap = {
      'Truck': Truck,
      'Upload': Upload,
      'Download': Download,
      'ShoppingCart': ShoppingCart,
      'ArrowRightLeft': ArrowRightLeft,
      'Package': Package,
      'RotateCcw': RotateCcw
    };
    return iconMap[iconName] || null;
  };

  // استفاده مستقیم از داده‌های دریافتی از backend (backend خودش فیلتر می‌کند)
  const filteredOperations = unloadings;

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
            {/* آیکون در سطر اول وسط در موبایل */}
            <div className="flex justify-center sm:justify-end sm:hidden mb-2">
              <div 
                className="p-2 bg-red-100 rounded-lg cursor-pointer hover:bg-red-200 transition-colors duration-200"
                onClick={handleBackToAdmin}
              >
                <List className="w-6 h-6 text-red-600" />
              </div>
            </div>
            
            {/* عنوان و زیرعنوان با آیکون در دسکتاپ */}
            <div className="text-center sm:text-right sm:flex sm:items-center sm:gap-3">
              <div>
                <h1 className="text-2xl font-bold">
                  لیست عملیات انجام شده
                </h1>
                <p className="text-slate-600 text-sm">
                  {loading ? "در حال دریافت و بارگذاری..." : `مشاهده و مدیریت عملیات انجام شده (${pagination.total})`}
                </p>
              </div>
              {/* آیکون در کنار عنوان در دسکتاپ */}
              <div className="hidden sm:block order-first">
                <div 
                  className="p-2 bg-red-100 rounded-lg cursor-pointer hover:bg-red-200 transition-colors duration-200"
                  onClick={handleBackToAdmin}
                >
                  <List className="w-6 h-6 text-red-600" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-6">
          <AlertManager
            showEditExpiredAlert={false}
            setShowEditExpiredAlert={() => {}}
            editingLoading={null}
            canEdit={false}
            remainingMinutes={0}
            showRemainingTimeAlert={false}
            setShowRemainingTimeAlert={() => {}}
            connectedToExisting={false}
            setConnectedToExisting={() => {}}
            started={false}
            message={message}
            error={error}
            setMessage={setMessage}
            setError={setError}
          />

          {/* Filter Card */}
          <Card className="bg-white shadow-sm border border-slate-200">
            <CardContent className="px-4 py-0">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div className="flex items-center gap-3">
                  <span className="text-sm font-medium text-gray-700">نوع عملیات:</span>
                  <Select value={selectedOperationType} onValueChange={setSelectedOperationType}>
                    <SelectTrigger className="w-48">
                      <SelectValue placeholder="انتخاب نوع عملیات" />
                    </SelectTrigger>
                    <SelectContent>
                      {operationTypes.map((type) => (
                        <SelectItem key={type.id} value={type.name}>
                          <div className="flex items-center gap-2">
                            {(() => {
                              const IconComponent = getIconComponent(type.icon);
                              return IconComponent ? <IconComponent className="w-4 h-4" /> : null;
                            })()}
                            <span>{type.persian_name}</span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-sm text-gray-500">
                    {(() => {
                      const selectedType = operationTypes.find(type => type.name === selectedOperationType);
                      if (selectedType) {
                        return `${selectedType.persian_name} ${filteredOperations.length} مورد`;
                      }
                      
                      return `عملیات ${filteredOperations.length} مورد`;
                    })()}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Loadings List Card */}
          <Card className="bg-white shadow-sm border border-slate-200">
            <CardContent>
              {loading ? (
                <div className="flex flex-col items-center justify-center py-12 space-y-4">
                  <Spinner className="w-12 h-12 text-blue-600" />
                  <p className="text-slate-600 text-lg">در حال بارگذاری لیست عملیات...</p>
                </div>
              ) : (
                <UnloadingList
                  unloadings={filteredOperations}
                  onShowDetails={handleShowDetails}
                  getWarehouseName={getWarehouseName}
                  getStatusColor={getStatusColor}
                  getStatusText={getStatusText}
                  getStatusIcon={getStatusIcon}
                  getTypeIcon={getTypeIcon}
                  getVersionText={getVersionText}
                  formatDate={formatDate}
                  formatTime={formatTime}
                  selectedOperationType={selectedOperationType}
                  operationTypes={operationTypes}
                />
              )}

              {!loading && (
                <>
                  {/* Always render pagination component when we have pagination data */}
                  <UnloadingPagination
                    pagination={pagination}
                    onPageChange={handlePageChange}
                    onNextPage={handleNextPage}
                    onPrevPage={handlePrevPage}
                  />
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </main>

      <UnloadingDetailsModal
        showDetailsModal={showDetailsModal}
        setShowDetailsModal={setShowDetailsModal}
        selectedUnloading={selectedUnloading}
        getWarehouseName={getWarehouseName}
        getPersianName={getPersianName}
        getStatusColor={getStatusColor}
        getStatusText={getStatusText}
        getTypeIcon={getTypeIcon}
        getVersionText={getVersionText}
        getSourceBadgeColor={getSourceBadgeColor}
        getGroupTitle={getGroupTitle}
        groupItemsByVersionAndSource={groupItemsByVersionAndSource}
        formatDate={formatDate}
        formatTime={formatTime}
      />
    </div>
  );
} 