import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { API_ENDPOINTS } from '@/config';
import { toast } from 'sonner';
import {
  VisionServerHeader,
  VisionServerSearchCard,
  VisionServerCard,
  VisionServerAddDialog,
  VisionServerEditDialog
} from '@/components/vision-servers';

export default function VisionServersPage() {
  const [servers, setServers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedWarehouse, setSelectedWarehouse] = useState('all');
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);

  const [selectedServer, setSelectedServer] = useState(null);
  const [formData, setFormData] = useState({
    name: '',
    persian_name: '',
    url: '',
    type: '',
    is_active: true,
    video_source: 'picamera'
  });

  // بارگذاری سرورها
  useEffect(() => {
    loadServers();
  }, []);

  const loadServers = async () => {
    try {
      setLoading(true);
      const response = await fetch(API_ENDPOINTS.VISION_SERVERS);
      const data = await response.json();
      console.log('📡 Raw vision servers response:', data);
      
      if (data.success) {
        setServers(data.data || []);
      } else {
        toast.error('خطا در بارگذاری سرورهای بینایی');
      }
    } catch (error) {
      console.error('خطا در بارگذاری سرورهای بینایی:', error);
      toast.error('خطا در بارگذاری سرورهای بینایی');
    } finally {
      setLoading(false);
    }
  };

  // فیلتر کردن سرورها بر اساس جستجو و انبار
  const filteredServers = servers.filter(server => {
    // فیلتر بر اساس جستجو
    const matchesSearch = 
      server.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      server.persian_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      server.url?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      server.type?.toLowerCase().includes(searchTerm.toLowerCase());

    if (!matchesSearch) return false;

    // فیلتر بر اساس انبار
    if (selectedWarehouse === 'all') return true; // اگر همه انبارها انتخاب شده، همه را نمایش بده

    if (selectedWarehouse === 'mobile') {
      // نمایش دوربین‌های متحرک (سرورهایی که به چند انبار اختصاص داده شده‌اند)
      return server.warehouse_ids && server.warehouse_ids.length > 1;
    } else {
      // نمایش سرورهای مربوط به انبار انتخاب شده
      // selectedWarehouse یک string هست (مثل "Anbar_Akhal") و باید با warehouse_ids مقایسه بشه
      return server.warehouse_ids && server.warehouse_ids.includes(selectedWarehouse);
    }
  });

  // باز کردن دیالوگ افزودن
  const openAddDialog = () => {
    setFormData({
      name: '',
      persian_name: '',
      url: '',
      type: '',
      is_active: true,
      video_source: 'picamera'
    });
    setIsAddDialogOpen(true);
  };

  // باز کردن دیالوگ ویرایش
  const openEditDialog = (server) => {
    setSelectedServer(server);
    setFormData({
      name: server.name || '',
      persian_name: server.persian_name || '',
      url: server.url || '',
      type: server.type || '',
      is_active: server.is_active || false,
      video_source: server.video_source || 'picamera'
    });
    setIsEditDialogOpen(true);
  };



  // تغییر فرم
  const handleFormChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  // ذخیره سرور (افزودن یا ویرایش)
  const handleSaveServer = async () => {
    try {
      const url = selectedServer 
        ? API_ENDPOINTS.VISION_SERVER_UPDATE(selectedServer.id)
        : API_ENDPOINTS.VISION_SERVER_CREATE;
      
      const method = selectedServer ? 'PUT' : 'POST';

      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      });

      const data = await response.json();

      if (data.success) {
        toast.success(selectedServer ? 'سرور با موفقیت ویرایش شد' : 'سرور با موفقیت افزوده شد');
        loadServers();
        setIsAddDialogOpen(false);
        setIsEditDialogOpen(false);
        setSelectedServer(null);
      } else {
        toast.error(data.error || 'خطا در ذخیره سرور');
      }
    } catch (error) {
      console.error('خطا در ذخیره سرور:', error);
      toast.error('خطا در ذخیره سرور');
    }
  };

  // حذف سرور
  const handleDeleteServer = async (server) => {
    try {
      const response = await fetch(API_ENDPOINTS.VISION_SERVER_DELETE(server.id), {
        method: 'DELETE'
      });

      const data = await response.json();

      if (data.success) {
        toast.success('سرور با موفقیت حذف شد');
        loadServers();
      } else {
        toast.error(data.error || 'خطا در حذف سرور');
      }
    } catch (error) {
      console.error('خطا در حذف سرور:', error);
      toast.error('خطا در حذف سرور');
    }
  };

  return (
    <>
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <VisionServerHeader 
            onAddClick={openAddDialog}
            serversCount={servers.length}
          />
        </div>
      </div>

      {/* Main content */}
      <div className="min-h-screen bg-slate-50">
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {/* Search Card */}
          <VisionServerSearchCard 
            searchTerm={searchTerm}
            onSearchChange={setSearchTerm}
            selectedWarehouse={selectedWarehouse}
            onWarehouseChange={setSelectedWarehouse}
          />

          {/* Servers List */}
          <Card className="mt-6">
            <CardContent className="p-6">
              {/* Server Cards - Both Desktop and Mobile */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                 <VisionServerCard 
                   servers={filteredServers}
                   loading={loading}
                   searchTerm={searchTerm}
                   onEditClick={openEditDialog}
                   onDeleteClick={handleDeleteServer}
                 />
              </div>
            </CardContent>
          </Card>
        </main>
      </div>

      {/* Add Server Dialog */}
      <VisionServerAddDialog 
        open={isAddDialogOpen}
        onOpenChange={setIsAddDialogOpen}
        formData={formData}
        onFormChange={handleFormChange}
        onSave={handleSaveServer}
        onCancel={() => setIsAddDialogOpen(false)}
      />

      {/* Edit Server Dialog */}
      <VisionServerEditDialog 
        open={isEditDialogOpen}
        onOpenChange={setIsEditDialogOpen}
        formData={formData}
        onFormChange={handleFormChange}
        onSave={handleSaveServer}
        onCancel={() => setIsEditDialogOpen(false)}
      />

      
    </>
  );
} 