import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';
import { API_ENDPOINTS } from '@/config';
import DeleteConfirmDialog from '@/components/DeleteConfirmDialog';
import { 
  Eye, 
  Warehouse, 
  CheckCircle, 
  XCircle,
  Truck,
  Settings,
  X,
  RefreshCw
} from 'lucide-react';
import { ServerAssignmentHeader } from '@/components/vision-servers';

export default function ServerAssignmentPage() {
  const [servers, setServers] = useState([]);
  const [warehouses, setWarehouses] = useState([]);
  const [assignments, setAssignments] = useState({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // بارگذاری سرورها و انبارها
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      
      // بارگذاری سرورها
      const serversResponse = await fetch(API_ENDPOINTS.VISION_SERVERS);
      const serversData = await serversResponse.json();
      
      // بارگذاری انبارها
      const warehousesResponse = await fetch(API_ENDPOINTS.WAREHOUSES);
      const warehousesData = await warehousesResponse.json();
      
      // بارگذاری تخصیص‌های فعلی
      const assignmentsResponse = await fetch(API_ENDPOINTS.VISION_SERVERS_GET_ASSIGNMENTS);
      const assignmentsData = await assignmentsResponse.json();
      
      if (serversData.success && warehousesData.warehouses) {
        const serversList = serversData.data || [];
        const warehousesList = warehousesData.warehouses || [];
        // assignmentsData حالا مستقیماً assignments است
        const currentAssignments = assignmentsData || {};
        
        console.log('📊 Loaded data:', {
          servers: serversList,
          warehouses: warehousesList,
          assignments: currentAssignments
        });
        
        setServers(serversList);
        setWarehouses(warehousesList);
        setAssignments(currentAssignments);
      } else {
        toast.error('خطا در بارگذاری اطلاعات');
      }
    } catch (error) {
      console.error('خطا در بارگذاری اطلاعات:', error);
      toast.error('خطا در بارگذاری اطلاعات');
    } finally {
      setLoading(false);
    }
  };

  // اضافه کردن انبار به سرور
  const handleAddWarehouse = (serverId, warehouseId) => {
    if (!warehouseId) return;
    
    setAssignments(prev => {
      const currentAssignments = prev[serverId] || [];
      if (!currentAssignments.includes(warehouseId)) {
        return {
          ...prev,
          [serverId]: [...currentAssignments, warehouseId]
        };
      }
      return prev;
    });
  };

  // حذف انبار از سرور
  const handleRemoveWarehouse = (serverId, warehouseId) => {
    setAssignments(prev => ({
      ...prev,
      [serverId]: prev[serverId]?.filter(id => id !== warehouseId) || []
    }));
  };

  // ذخیره assignments
  const handleSaveAssignments = async () => {
    try {
      setSaving(true);
      
      console.log('Saving assignments:', assignments);
      
      const response = await fetch(API_ENDPOINTS.VISION_SERVERS_ASSIGNMENTS, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ assignments })
      });

      console.log('Response status:', response.status);
      const data = await response.json();
      console.log('Response data:', data);

      if (data.success) {
        toast.success('تخصیص سرورها با موفقیت ذخیره شد');
        await loadData(); // بارگذاری مجدد برای اطمینان از sync
      } else {
        toast.error(data.error || 'خطا در ذخیره تخصیص سرورها');
      }
    } catch (error) {
      console.error('خطا در ذخیره تخصیص سرورها:', error);
      toast.error('خطا در ذخیره تخصیص سرورها');
    } finally {
      setSaving(false);
    }
  };

  // تعداد انبارهای اختصاص داده شده به سرور
  const getAssignmentCount = (serverId) => {
    return assignments[serverId]?.length || 0;
  };

  // بررسی اینکه آیا سرور متحرک است (به چند انبار اختصاص داده شده)
  const isMobileServer = (serverId) => {
    return getAssignmentCount(serverId) > 1;
  };

  // دریافت نام انبار بر اساس ID
  const getWarehouseName = (warehouseId) => {
    const warehouse = warehouses.find(w => w.id === warehouseId);
    return warehouse ? (warehouse.persian_name || warehouse.name) : 'نامشخص';
  };

  // دریافت انبارهای اختصاص داده نشده برای یک سرور
  const getAvailableWarehouses = (serverId) => {
    const assignedWarehouseIds = assignments[serverId] || [];
    return warehouses.filter(warehouse => 
      warehouse && 
      warehouse.id && 
      !assignedWarehouseIds.includes(warehouse.id)
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="flex items-center gap-2">
          <RefreshCw className="animate-spin h-6 w-6 text-blue-600" />
          <span className="text-gray-600">در حال بارگذاری...</span>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <ServerAssignmentHeader 
            onSaveClick={handleSaveAssignments}
            onRefreshClick={loadData}
            loading={loading}
            saving={saving}
            serversCount={servers.length}
          />
        </div>
      </div>

      {/* Main content */}
      <div className="min-h-screen bg-slate-50">
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {/* Instructions */}
          <Card className="mb-6">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <Settings className="h-5 w-5 text-blue-600 mt-0.5" />
                <div>
                  <h3 className="font-medium text-gray-900 mb-1">راهنمای تخصیص سرورها</h3>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• هر سرور می‌تواند به چندین انبار اختصاص داده شود</li>
                    <li>• سرورهایی که به بیش از یک انبار اختصاص داده شده‌اند، دوربین‌های متحرک محسوب می‌شوند</li>
                    <li>• برای اضافه کردن انبار، از Select استفاده کنید</li>
                    <li>• برای حذف انبار، روی دکمه X کنار نام انبار کلیک کنید</li>
                    <li>• پس از اعمال تغییرات، روی دکمه "ذخیره تخصیص‌ها" کلیک کنید</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Servers Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {servers.map((server) => (
              <Card key={server.id} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Eye className="h-5 w-5 text-blue-600" />
                      <div>
                        <CardTitle className="text-lg">{server.name}</CardTitle>
                        <p className="text-sm text-gray-600">{server.persian_name}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {isMobileServer(server.id) && (
                        <Badge variant="secondary" className="bg-orange-100 text-orange-800">
                          <Truck className="h-3 w-3 mr-1" />
                          متحرک
                        </Badge>
                      )}
                      <Badge variant="outline">
                        {getAssignmentCount(server.id)} انبار
                      </Badge>
                    </div>
                  </div>
                </CardHeader>
                
                <CardContent className="pt-0">
                  <div className="space-y-4">
                    {/* انبارهای اختصاص داده شده */}
                    <div>
                      <div className="text-sm font-medium text-gray-700 mb-2">
                        انبارهای اختصاص داده شده:
                      </div>
                      
                      {assignments[server.id]?.length > 0 ? (
                        <div className="space-y-2">
                          {assignments[server.id].map((warehouseId) => (
                            <div key={warehouseId} className="flex items-center justify-between bg-gray-50 p-2 rounded-lg">
                              <div className="flex items-center gap-2">
                                <Warehouse className="h-4 w-4 text-gray-400" />
                                <span className="text-sm">{getWarehouseName(warehouseId)}</span>
                              </div>
                                                             <DeleteConfirmDialog
                                 itemName={getWarehouseName(warehouseId)}
                                 itemType="warehouse-assignment"
                                 onConfirm={() => handleRemoveWarehouse(server.id, warehouseId)}
                               >
                                 <Button
                                   variant="ghost"
                                   size="sm"
                                   className="h-6 w-6 p-0 text-red-500 hover:text-red-700 hover:bg-red-50"
                                 >
                                   <X className="h-3 w-3" />
                                 </Button>
                               </DeleteConfirmDialog>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-sm text-gray-500 bg-gray-50 p-3 rounded-lg text-center">
                          هیچ انباری اختصاص داده نشده
                        </div>
                      )}
                    </div>

                    {/* Select برای اضافه کردن انبار */}
                    <div>
                      <div className="text-sm font-medium text-gray-700 mb-2">
                        اضافه کردن انبار:
                      </div>
                      <Select onValueChange={(value) => handleAddWarehouse(server.id, value)}>
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="انتخاب انبار برای اضافه کردن..." />
                        </SelectTrigger>
                        <SelectContent>
                          {getAvailableWarehouses(server.id)
                            .filter(warehouse => warehouse && warehouse.id)
                            .map((warehouse) => (
                              <SelectItem key={warehouse.id} value={warehouse.id.toString()}>
                                {warehouse.persian_name || warehouse.name || 'نامشخص'}
                              </SelectItem>
                            ))}
                          {getAvailableWarehouses(server.id).filter(warehouse => warehouse && warehouse.id).length === 0 && (
                            <div className="px-2 py-1.5 text-sm text-gray-500">
                              همه انبارها اختصاص داده شده‌اند
                            </div>
                          )}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Empty State */}
          {servers.length === 0 && (
            <Card>
              <CardContent className="p-8 text-center">
                <Eye className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">هیچ سروری یافت نشد</h3>
                <p className="text-gray-600">ابتدا سرورهای بینایی را در صفحه مدیریت سرورها ایجاد کنید.</p>
              </CardContent>
            </Card>
          )}
        </main>
      </div>
    </>
  );
}
