import React, { useState, useEffect } from "react";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle,
  DialogClose
} from "@/components/ui/dialog";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { 
  PackageIcon,
  UserIcon,
  BotIcon,
  HistoryIcon,
  X,
  HashIcon,
  Clock,
  Calendar,
  CheckCircle,
  Edit,
  Building2,
  Activity,
  Tag,
  EyeIcon,
  CameraIcon,
  FileText
} from "lucide-react";
import { API_ENDPOINTS } from "@/config";
import JsonDisplayBox from "@/components/JsonDisplayBox";
import { getOperationTypeText, extractIdFromToken } from "./UnloadingUtils";

// Map icon names to components
const iconMap = {
  BotIcon,
  UserIcon,
  HistoryIcon,
  PackageIcon,
  EyeIcon
};

const UnloadingDetailsModal = ({ 
  showDetailsModal, 
  setShowDetailsModal, 
  selectedUnloading, 
  getWarehouseName, 
  getPersianName, 
  getStatusColor, 
  getStatusText, 
  getTypeIcon, 
  getVersionText, 
  getSourceBadgeColor, 
  getGroupTitle, 
  groupItemsByVersionAndSource, 
  formatDate, 
  formatTime 
}) => {
  const [warehouseInfo, setWarehouseInfo] = useState(null);

  // Helper function to get operation type from different possible field names
  const getOperationType = (operation) => {
    return operation?.type;
  };

  // دریافت اطلاعات انبار برای آدرس سرور بینایی
  useEffect(() => {
    const fetchWarehouseInfo = async () => {
      if (selectedUnloading?.warehouse_id) {
        try {
          const response = await fetch(API_ENDPOINTS.WAREHOUSES);
          const data = await response.json();
          if (data.success && data.warehouses) {
            const warehouse = data.warehouses.find(w => w.id === selectedUnloading.warehouse_id);
            setWarehouseInfo(warehouse);
          }
        } catch (error) {
          console.error('خطا در دریافت اطلاعات انبار:', error);
        }
      }
    };

    fetchWarehouseInfo();
  }, [selectedUnloading?.warehouse_id]);
  // Helper function to render icons
  const renderIcon = (iconName) => {
    const IconComponent = iconMap[iconName];
    return IconComponent ? <IconComponent className="w-4 h-4" /> : <PackageIcon className="w-4 h-4" />;
  };

  // Helper function to get all groups including empty vision groups
  const getAllGroups = (items) => {
    console.log('getAllGroups called with items:', items);
    console.log('selectedUnloading:', selectedUnloading);
    
    const groupedItems = groupItemsByVersionAndSource(items);
    console.log('groupedItems:', groupedItems);
    
    // Get the maximum version from the unloading
    const maxVersion = selectedUnloading.version || 1;
    
    // Create a map of existing groups for easy lookup
    const existingGroups = {};
    groupedItems.forEach(group => {
      const key = `${group.version}-${group.source}`;
      existingGroups[key] = group;
    });
    
    // Ensure we have all expected groups
    const allGroups = [];
    
    // Always add vision group for version 1
    const visionKey = '1-vision';
    if (existingGroups[visionKey]) {
      allGroups.push(existingGroups[visionKey]);
    } else {
      allGroups.push({
        version: 1,
        source: 'vision',
        items: []
      });
    }
    
    // Add user groups for all versions (2 and above)
    for (let version = 2; version <= maxVersion; version++) {
      const userKey = `${version}-user`;
      if (existingGroups[userKey]) {
        allGroups.push(existingGroups[userKey]);
      } else {
        allGroups.push({
          version: version,
          source: 'user',
          items: []
        });
      }
    }
    
    // Add any other groups that might exist (like vision groups for other versions)
    groupedItems.forEach(group => {
      const key = `${group.version}-${group.source}`;
      if (!existingGroups[key] || !allGroups.some(g => `${g.version}-${g.source}` === key)) {
        allGroups.push(group);
      }
    });
    
    console.log('Final allGroups:', allGroups);
    
    // Sort groups by version (descending) and source (user first)
    return allGroups.sort((a, b) => {
      if (a.version !== b.version) {
        return b.version - a.version; // نسخه جدیدتر اول
      }
      // اگر نسخه یکسان است، منبع user اول
      if (a.source === 'user' && b.source !== 'user') return -1;
      if (b.source === 'user' && a.source !== 'user') return 1;
      return a.source.localeCompare(b.source);
    });
  };

  // Helper function to parse vision output
  const parseVisionOutput = () => {
    if (!selectedUnloading?.vision_output) return null;
    
    try {
      return JSON.parse(selectedUnloading.vision_output);
    } catch (error) {
      console.error("Error parsing vision output:", error);
      return null;
    }
  };

  // Helper function to format vision event timestamp
  const formatVisionTimestamp = (timestamp) => {
    if (!timestamp) return "نامشخص";
    try {
      const date = new Date(timestamp);
      const dateStr = date.toLocaleDateString('fa-IR');
      const timeStr = date.toLocaleTimeString('fa-IR');
      return `${dateStr} ${timeStr}`;
    } catch (error) {
      return timestamp;
    }
  };

  const visionData = parseVisionOutput();

  return (
    <Dialog open={showDetailsModal} onOpenChange={setShowDetailsModal}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto sm:max-w-2xl" showCloseButton={false}>
        <DialogHeader className="flex flex-row items-center justify-between">
                      <DialogTitle className="flex items-center gap-2 text-base sm:text-lg">
              <PackageIcon className="w-4 h-4 sm:w-5 sm:h-5" />
              <span className="truncate">
                جزئیات {getOperationTypeText(getOperationType(selectedUnloading))} {extractIdFromToken(selectedUnloading?.token)}
              </span>
            </DialogTitle>
          <DialogClose asChild>
            <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
              <X className="h-4 w-4" />
            </Button>
          </DialogClose>
        </DialogHeader>
        
        {selectedUnloading && (
          <Tabs defaultValue="general" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="general" className="flex items-center gap-2">
                <PackageIcon className="w-4 h-4" />
                جزئیات عمومی
              </TabsTrigger>
              <TabsTrigger value="vision" className="flex items-center gap-2">
                <CameraIcon className="w-4 h-4" />
                جزئیات بینایی
              </TabsTrigger>
              <TabsTrigger value="json" className="flex items-center gap-2">
                <FileText className="w-4 h-4" />
                خروجی بینایی (JSON)
              </TabsTrigger>
            </TabsList>

            {/* Tab 1: General Details */}
            <TabsContent value="general" className="space-y-6 mt-6">
              {/* اطلاعات اصلی */}
              <div className="space-y-4">
                {/* اطلاعات اصلی و زمانی کنار هم در دسکتاپ */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {/* اطلاعات اصلی */}
                  <div className="space-y-3">
                    <h4 className="font-semibold text-slate-800 text-sm sm:text-base">اطلاعات اصلی</h4>
                    <div className="bg-slate-50 rounded-lg p-3 space-y-2 border border-slate-200">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          <HashIcon className="w-4 h-4 text-slate-500" />
                          <span className="text-slate-600 text-sm">شناسه:</span>
                        </div>
                        <span className="font-medium text-sm">{selectedUnloading.token}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          <Building2 className="w-4 h-4 text-slate-500" />
                          <span className="text-slate-600 text-sm">انبار:</span>
                        </div>
                        <span className="font-medium text-sm">{getWarehouseName(selectedUnloading.warehouse_id)}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          <Activity className="w-4 h-4 text-slate-500" />
                          <span className="text-slate-600 text-sm">وضعیت:</span>
                        </div>
                        <Badge className={`${getStatusColor(selectedUnloading.status)} text-xs`}>
                          {getStatusText(selectedUnloading.status)}
                        </Badge>
                      </div>
                      {selectedUnloading.version && (
                        <div className="flex justify-between items-center">
                          <div className="flex items-center gap-2">
                            <Tag className="w-4 h-4 text-slate-500" />
                            <span className="text-slate-600 text-sm">نسخه:</span>
                          </div>
                          <span className="font-medium text-sm">{selectedUnloading.version}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* اطلاعات زمانی */}
                  <div className="space-y-3">
                    <h4 className="font-semibold text-slate-800 text-sm sm:text-base">اطلاعات زمانی</h4>
                    <div className="bg-slate-50 rounded-lg p-3 space-y-3 border border-slate-200">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          <Clock className="w-4 h-4 text-slate-500" />
                          <span className="text-slate-600 text-sm">زمان شروع:</span>
                        </div>
                        {selectedUnloading.start_time ? (
                          <div className="flex items-center gap-2">
                            <div className="flex items-center gap-1">
                              <Clock className="w-3 h-3 text-slate-400" />
                              <span className="font-medium text-sm">{formatTime(selectedUnloading.start_time)}</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Calendar className="w-3 h-3 text-slate-400" />
                              <span className="font-medium text-sm">{formatDate(selectedUnloading.start_time)}</span>
                            </div>
                          </div>
                        ) : null}
                      </div>
                      
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          <Clock className="w-4 h-4 text-slate-500" />
                          <span className="text-slate-600 text-sm">زمان پایان:</span>
                        </div>
                        {selectedUnloading.end_time ? (
                          <div className="flex items-center gap-2">
                            <div className="flex items-center gap-1">
                              <Clock className="w-3 h-3 text-slate-400" />
                              <span className="font-medium text-sm">{formatTime(selectedUnloading.end_time)}</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Calendar className="w-3 h-3 text-slate-400" />
                              <span className="font-medium text-sm">{formatDate(selectedUnloading.end_time)}</span>
                            </div>
                          </div>
                        ) : null}
                      </div>
                      
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          <CheckCircle className="w-4 h-4 text-slate-500" />
                          <span className="text-slate-600 text-sm">زمان تایید کاربر:</span>
                        </div>
                        {selectedUnloading.user_confirm_time ? (
                          <div className="flex items-center gap-2">
                            <div className="flex items-center gap-1">
                              <Clock className="w-3 h-3 text-slate-400" />
                              <span className="font-medium text-sm">{formatTime(selectedUnloading.user_confirm_time)}</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Calendar className="w-3 h-3 text-slate-400" />
                              <span className="font-medium text-sm">{formatDate(selectedUnloading.user_confirm_time)}</span>
                            </div>
                          </div>
                        ) : null}
                      </div>
                      
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          <Edit className="w-4 h-4 text-slate-500" />
                          <span className="text-slate-600 text-sm">زمان ویرایش:</span>
                        </div>
                        {selectedUnloading.edit_time ? (
                          <div className="flex items-center gap-2">
                            <div className="flex items-center gap-1">
                              <Clock className="w-3 h-3 text-slate-400" />
                              <span className="font-medium text-sm">{formatTime(selectedUnloading.edit_time)}</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Calendar className="w-3 h-3 text-slate-400" />
                              <span className="font-medium text-sm">{formatDate(selectedUnloading.edit_time)}</span>
                            </div>
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* آیتم‌های بارگیری */}
              <div className="space-y-3">
                <h4 className="font-semibold text-slate-800 text-sm sm:text-base">آیتم‌ها</h4>
                
                {getAllGroups(selectedUnloading.items || []).map((group, groupIndex) => (
                  <Card key={`${group.version}-${group.source}-${groupIndex}`} className="border border-slate-200 shadow-sm bg-white">
                    <CardHeader className="pb-0">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Badge className={`${getSourceBadgeColor(group.source)} text-sm`}>
                            {getGroupTitle(group)}
                          </Badge>
                          <span className="text-sm text-slate-500">
                            ({group.items.length} آیتم)
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          {renderIcon(getTypeIcon(group.source === 'vision' ? 'vision' : 'user'))}
                          <span className="text-xs text-slate-400">
                            نسخه {getVersionText(group.version)}
                          </span>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      {group.items.length > 0 ? (
                        <div className="space-y-1">
                          {group.items.map((item, index) => (
                            <div key={index} className="py-1">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <h5 className="font-semibold text-slate-900 truncate">
                                    {getPersianName ? getPersianName(item.name) : (item.name || "نامشخص")}
                                  </h5>
                                  <div className="flex items-center gap-1 text-slate-500">
                                    <HashIcon className="w-3 h-3" />
                                    <span className="text-sm font-bold text-slate-700">{item.count || "نامشخص"} مورد</span>
                                  </div>
                                </div>
                                <Badge className={
                                  item.type === 'loaded' ? 'bg-green-100 text-green-800 text-xs border-green-200' : 
                                  item.type === 'unloaded' ? 'bg-red-100 text-red-800 text-xs border-red-200' : 
                                  'bg-gray-100 text-gray-800 text-xs border-gray-200'
                                }>
                                  {item.type === 'loaded' ? 'بارگیری شده' : 
                                   item.type === 'unloaded' ? 'تخلیه شده' : item.type}
                                </Badge>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        // نمایش پیام مناسب برای گروه‌های خالی
                        <div className="text-center py-4">
                          {group.source === 'vision' ? (
                            <div className="flex flex-col items-center gap-2">
                              <EyeIcon className="w-8 h-8 text-slate-400" />
                              <p className="text-slate-500 text-sm">هیچ داده‌ای از سیستم بینایی دریافت نشده است</p>
                              <p className="text-slate-400 text-xs">اطلاعات بینایی در این بارگیری موجود نیست</p>
                            </div>
                          ) : (
                            <div className="flex flex-col items-center gap-2">
                              <UserIcon className="w-8 h-8 text-slate-400" />
                              <p className="text-slate-500 text-sm">هیچ آیتمی در نسخه {getVersionText(group.version)} کاربر وجود ندارد</p>
                            </div>
                          )}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>

              {/* اطلاعات اضافی */}
              {selectedUnloading.metadata && (
                <div className="space-y-2">
                  <h4 className="font-semibold text-slate-800">اطلاعات اضافی</h4>
                  <div className="bg-slate-50 p-4 rounded-lg border border-slate-200">
                    <pre className="text-sm text-slate-700 whitespace-pre-wrap">
                      {JSON.stringify(selectedUnloading.metadata, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </TabsContent>

            {/* Tab 2: Vision Details */}
            <TabsContent value="vision" className="space-y-6 mt-6">
              {visionData ? (
                <div className="space-y-6">
                  {/* Vision Summary - Only show if there are events */}
                  {visionData.summary && visionData.summary.events && Object.keys(visionData.summary.events).length > 0 && (
                    <div className="space-y-4">
                      <h4 className="font-semibold text-slate-800 text-sm sm:text-base">خلاصه عملیات بینایی</h4>
                      <div className="bg-slate-50 rounded-lg p-4 space-y-3 border border-slate-200">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                                     <div className="space-y-3">
                             <div className="flex items-center justify-between">
                               <div className="flex items-center gap-2">
                                 <Activity className="w-4 h-4 text-slate-500" />
                                 <span className="text-slate-600 text-sm">نوع عملیات:</span>
                               </div>
                               <Badge className="bg-blue-100 text-blue-800 text-xs border-blue-200">
                                 {getOperationTypeText(visionData.summary.operation_type || getOperationType(selectedUnloading))}
                               </Badge>
                             </div>
                             <div className="flex items-center justify-between">
                               <div className="flex items-center gap-2">
                                 <PackageIcon className="w-4 h-4 text-slate-500" />
                                 <span className="text-slate-600 text-sm">تعداد کل محصولات:</span>
                               </div>
                               <span className="font-medium text-sm">{visionData.summary.total_products || 0}</span>
                             </div>
                           </div>
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <Clock className="w-4 h-4 text-slate-500" />
                                <span className="text-slate-600 text-sm">زمان شروع:</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="flex items-center gap-1">
                                  <Clock className="w-3 h-3 text-slate-400" />
                                  <span className="font-medium text-sm">{formatVisionTimestamp(visionData.summary.start_time).split(' ')[1]}</span>
                                </div>
                                <div className="flex items-center gap-1">
                                  <Calendar className="w-3 h-3 text-slate-400" />
                                  <span className="font-medium text-sm">{formatVisionTimestamp(visionData.summary.start_time).split(' ')[0]}</span>
                                </div>
                              </div>
                            </div>
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <Clock className="w-4 h-4 text-slate-500" />
                                <span className="text-slate-600 text-sm">زمان پایان:</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="flex items-center gap-1">
                                  <Clock className="w-3 h-3 text-slate-400" />
                                  <span className="font-medium text-sm">{formatVisionTimestamp(visionData.summary.end_time).split(' ')[1]}</span>
                                </div>
                                <div className="flex items-center gap-1">
                                  <Calendar className="w-3 h-3 text-slate-400" />
                                  <span className="font-medium text-sm">{formatVisionTimestamp(visionData.summary.end_time).split(' ')[0]}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Vision Events */}
                  {visionData.summary?.events && Object.keys(visionData.summary.events).length > 0 ? (
                    <div className="space-y-4">
                      <h4 className="font-semibold text-slate-800 text-sm sm:text-base">رویدادهای بینایی</h4>
                      <div className="space-y-4">
                        {Object.entries(visionData.summary.events).map(([eventId, event]) => (
                                                     <Card key={eventId} className="border border-slate-200 shadow-sm bg-white">
                             <CardContent className="px-3">
                               <div className="flex flex-col md:flex-row gap-2">
                                 {/* Information Section */}
                                 <div className="w-full md:w-1/2 space-y-3 p-3 border border-slate-200 rounded-lg bg-slate-50">
                                   {/* Row 1: Product Type */}
                                   <div className="flex items-center justify-between">
                                     <div className="flex items-center gap-2">
                                       <PackageIcon className="w-4 h-4 text-slate-500" />
                                       <span className="text-slate-600 text-sm font-medium">نوع محصول:</span>
                                     </div>
                                     <Badge className="bg-purple-100 text-purple-800 text-xs border-purple-200 font-medium">
                                       {event.product_type || 'نامشخص'}
                                     </Badge>
                                   </div>
                                   
                                   {/* Row 2: Status */}
                                   <div className="flex items-center justify-between">
                                     <div className="flex items-center gap-2">
                                       <Activity className="w-4 h-4 text-slate-500" />
                                       <span className="text-slate-600 text-sm font-medium">وضعیت:</span>
                                     </div>
                                     <Badge className={
                                       event.status === 'loaded' ? 'bg-green-100 text-green-800 text-xs border-green-200' :
                                       event.status === 'unloaded' ? 'bg-red-100 text-red-800 text-xs border-red-200' :
                                       'bg-gray-100 text-gray-800 text-xs border-gray-200'
                                     }>
                                       {event.status === 'loaded' ? 'بارگیری شده' :
                                        event.status === 'unloaded' ? 'تخلیه شده' : event.status}
                                     </Badge>
                                   </div>
                                   
                                   {/* Row 3: Time */}
                                   <div className="flex items-center justify-between">
                                     <div className="flex items-center gap-2">
                                       <Clock className="w-4 h-4 text-slate-500" />
                                       <span className="text-slate-600 text-sm font-medium">زمان:</span>
                                     </div>
                                     <div className="flex items-center gap-2">
                                       <div className="flex items-center gap-1">
                                         <Clock className="w-3 h-3 text-slate-400" />
                                         <span className="font-medium text-sm text-slate-700">
                                           {formatVisionTimestamp(event.timestamp).split(' ')[1]}
                                         </span>
                                       </div>
                                       <div className="flex items-center gap-1">
                                         <Calendar className="w-3 h-3 text-slate-400" />
                                         <span className="font-medium text-sm text-slate-700">
                                           {formatVisionTimestamp(event.timestamp).split(' ')[0]}
                                         </span>
                                       </div>
                                     </div>
                                   </div>
                                 </div>
                                 
                                 {/* Image Section */}
                                 <div className="w-full md:w-1/2">
                                   {event.snapshot ? (
                                     <div className="w-full h-32 rounded-lg border border-slate-300 overflow-hidden py-2">
                                       <img 
                                         src={`http://172.16.6.79:5001/snapshots/${event.snapshot}`}
                                         alt={`Snapshot ${event.snapshot}`}
                                         className="w-full h-full object-contain"
                                         onError={(e) => {
                                           e.target.style.display = 'none';
                                           e.target.nextSibling.style.display = 'flex';
                                         }}
                                       />
                                       <div className="w-full h-full flex items-center justify-center bg-slate-100" style={{display: 'none'}}>
                                         <div className="text-center">
                                           <CameraIcon className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                                           <p className="text-slate-500 text-xs">تصویر در دسترس نیست</p>
                                         </div>
                                       </div>
                                     </div>
                                   ) : (
                                     <div className="w-full h-32 bg-slate-100 rounded-lg border-2 border-dashed border-slate-300 flex items-center justify-center">
                                       <div className="text-center">
                                         <CameraIcon className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                                         <p className="text-slate-500 text-xs">
                                           تصویر موجود نیست
                                         </p>
                                       </div>
                                     </div>
                                   )}
                                 </div>
                               </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <div className="flex flex-col items-center gap-3">
                        <CameraIcon className="w-12 h-12 text-slate-400" />
                        <p className="text-slate-500 text-sm">هیچ رویداد بینایی‌ای ثبت نشده است</p>
                        <p className="text-slate-400 text-xs">اطلاعات رویدادهای بینایی در این بارگیری موجود نیست</p>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="flex flex-col items-center gap-3">
                    <EyeIcon className="w-12 h-12 text-slate-400" />
                    <p className="text-slate-500 text-sm">هیچ داده بینایی‌ای موجود نیست</p>
                    <p className="text-slate-400 text-xs">فیلد vision_output در این بارگیری خالی است</p>
                  </div>
                </div>
              )}
            </TabsContent>

            {/* Tab 3: JSON Output */}
            <TabsContent value="json" className="space-y-6 mt-6">
              {selectedUnloading?.vision_output ? (
                <JsonDisplayBox 
                  data={visionData} 
                  title="خروجی سیستم بینایی (JSON)" 
                />
              ) : (
                <div className="text-center py-8">
                  <div className="flex flex-col items-center gap-3">
                    <FileText className="w-12 h-12 text-slate-400" />
                    <p className="text-slate-500 text-sm">هیچ داده بینایی‌ای موجود نیست</p>
                    <p className="text-slate-400 text-xs">فیلد vision_output در این بارگیری خالی است</p>
                  </div>
                </div>
              )}
            </TabsContent>
          </Tabs>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default UnloadingDetailsModal; 
