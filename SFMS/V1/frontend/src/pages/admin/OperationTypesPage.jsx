import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle 
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';

import { 
  Plus, 
  Settings
} from 'lucide-react';
import ActionButtons from '@/components/ui/action-buttons';

import { getIconComponent, iconOptions } from '@/components/IconSelector';
import { showSuccess, showError } from '@/lib/toast';
import { API_ENDPOINTS } from '@/config';
import { useNavigate } from 'react-router-dom';

export default function OperationTypesPage() {
  const navigate = useNavigate();
  const [operationTypes, setOperationTypes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [selectedOperation, setSelectedOperation] = useState(null);
  
  const [addForm, setAddForm] = useState({
    name: '',
    persian_name: '',
    icon: '',
    color: 'blue',
    description: '',
    order: 0,
    is_enabled: true,
    is_available: true
  });

  const [editForm, setEditForm] = useState({
    persian_name: '',
    icon: '',
    color: 'blue',
    description: '',
    order: 0,
    is_enabled: true,
    is_available: true
  });

  // Fetch operation types
  const fetchOperationTypes = async () => {
    try {
      setLoading(true);
      const response = await fetch(API_ENDPOINTS.OPERATION_TYPES);
      const data = await response.json();
      
      if (data.success) {
        setOperationTypes(data.data);
      } else {
        showError('خطا در بارگذاری انواع عملیات');
      }
    } catch (error) {
      console.error('Error fetching operation types:', error);
      showError('خطا در بارگذاری انواع عملیات');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOperationTypes();
  }, []);

  // Handle add operation type
  const handleAdd = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.OPERATION_TYPE_CREATE, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(addForm)
      });
      
      const data = await response.json();
      
      if (data.success) {
        showSuccess('نوع عملیات با موفقیت اضافه شد');
        setShowAddDialog(false);
        setAddForm({
          name: '',
          persian_name: '',
          icon: '',
          color: 'blue',
          description: '',
          order: 0,
          is_enabled: true,
          is_available: true
        });
        fetchOperationTypes();
      } else {
        showError(data.error || 'خطا در اضافه کردن نوع عملیات');
      }
    } catch (error) {
      console.error('Error adding operation type:', error);
      showError('خطا در اضافه کردن نوع عملیات');
    }
  };

  // Handle edit operation type
  const handleEdit = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.OPERATION_TYPE_UPDATE(selectedOperation.id), {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(editForm)
      });
      
      const data = await response.json();
      
      if (data.success) {
        showSuccess('نوع عملیات با موفقیت بروزرسانی شد');
        setShowEditDialog(false);
        setSelectedOperation(null);
        fetchOperationTypes();
      } else {
        showError(data.error || 'خطا در بروزرسانی نوع عملیات');
      }
    } catch (error) {
      console.error('Error updating operation type:', error);
      showError('خطا در بروزرسانی نوع عملیات');
    }
  };

  // Handle delete operation type
  const handleDelete = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.OPERATION_TYPE_DELETE(selectedOperation.id), {
        method: 'DELETE'
      });
      
      const data = await response.json();
      
      if (data.success) {
        showSuccess('نوع عملیات با موفقیت حذف شد');
        setShowDeleteDialog(false);
        setSelectedOperation(null);
        fetchOperationTypes();
      } else {
        showError(data.error || 'خطا در حذف نوع عملیات');
      }
    } catch (error) {
      console.error('Error deleting operation type:', error);
      showError('خطا در حذف نوع عملیات');
    }
  };

  // Handle edit click
  const handleEditClick = (operation) => {
    setSelectedOperation(operation);
    setEditForm({
      persian_name: operation.persian_name || '',
      icon: operation.icon || '',
      color: operation.color || 'blue',
      description: operation.description || '',
      order: operation.order || 0,
      is_enabled: operation.is_enabled,
      is_available: operation.is_available
    });
    setShowEditDialog(true);
  };

  // Handle delete click
  const handleDeleteClick = (operation) => {
    setSelectedOperation(operation);
    setShowDeleteDialog(true);
  };

  const colorOptions = [
    { value: 'blue', label: 'آبی' },
    { value: 'green', label: 'سبز' },
    { value: 'red', label: 'قرمز' },
    { value: 'yellow', label: 'زرد' },
    { value: 'purple', label: 'بنفش' },
    { value: 'pink', label: 'صورتی' },
    { value: 'indigo', label: 'نیلی' },
    { value: 'gray', label: 'خاکستری' }
  ];

     

  return (
    <div className="min-h-screen bg-slate-50">
                    {/* Header */}
       <div className="bg-white shadow-sm border-b border-gray-200">
         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
           <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
                           {/* دکمه پنل مدیریت در سطر اول وسط در موبایل */}
              <div className="flex justify-center sm:justify-end sm:hidden mb-2">
                <div 
                  className="p-2 bg-purple-100 rounded-lg cursor-pointer hover:bg-purple-200 transition-colors duration-200"
                  onClick={() => navigate('/admin')}
                >
                  <Settings className="w-6 h-6 text-purple-600" />
                </div>
              </div>
             
             {/* عنوان و زیرعنوان با دکمه خانه در دسکتاپ */}
             <div className="text-center sm:text-right sm:flex sm:items-center sm:gap-3">
               <div>
                 <h1 className="text-2xl font-bold">
                   مدیریت انواع عملیات
                 </h1>
                 <p className="text-slate-600 text-sm">
                   تنظیم دکمه‌های صفحه اصلی ({operationTypes.length} نوع عملیات)
                 </p>
               </div>
                               {/* دکمه پنل مدیریت در کنار عنوان در دسکتاپ */}
                <div className="hidden sm:block order-first">
                  <div 
                    className="p-2 bg-purple-100 rounded-lg cursor-pointer hover:bg-purple-200 transition-colors duration-200"
                    onClick={() => navigate('/admin')}
                  >
                    <Settings className="w-6 h-6 text-purple-600" />
                  </div>
                </div>
             </div>
             
             {/* دکمه‌ها در دسکتاپ - سمت راست */}
             <div className="hidden sm:flex items-center gap-3">
               <Button 
                 onClick={() => setShowAddDialog(true)}
                 className="bg-green-600 hover:bg-green-700 border border-green-600 hover:border-green-700 transition-all duration-200"
               >
                 <Plus className="h-4 w-4 ml-2" />
                 افزودن نوع عملیات
               </Button>
             </div>
           </div>
           
           {/* دکمه‌ها در موبایل - در همان سطر عنوان */}
           <div className="flex justify-center gap-2 sm:hidden mt-4">
             <Button 
               onClick={() => setShowAddDialog(true)}
               className="bg-green-600 hover:bg-green-700 border border-green-600 hover:border-green-700 transition-all duration-200"
             >
               <Plus className="h-4 w-4 ml-2" />
               افزودن نوع عملیات
             </Button>
           </div>
         </div>
       </div>

       {/* Main Content */}
       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
         <div className="space-y-6">

        

                                   {/* Operation Types Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {loading ? (
              <div className="col-span-full text-center py-8">
                <div className="flex items-center justify-center gap-2">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                  در حال بارگذاری...
                </div>
              </div>
            ) : (
              operationTypes.map((operation) => (
                <Card key={operation.id} className="bg-white shadow-sm hover:shadow-md transition-shadow duration-200">
                                     <CardContent className="px-6 py-3">
                                                               {/* Header with Icon and Title */}
                      <div className="mb-4">
                        <div className="flex items-center gap-4 mb-3">
                          <div className={`w-12 h-12 rounded-xl bg-${operation.color}-100 flex items-center justify-center`}>
                            {React.createElement(getIconComponent(operation.icon), {
                              className: `w-6 h-6 text-${operation.color}-600`
                            })}
                          </div>
                          <div>
                            <h3 className="text-lg font-semibold text-gray-900 mb-1">
                              {operation.persian_name}
                            </h3>
                            {operation.description && (
                              <p className="text-sm text-gray-500">
                                {operation.description}
                              </p>
                            )}
                          </div>
                        </div>
                        
                        {/* Status Badges - Horizontal */}
                        <div className="flex gap-2">
                          <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs ${
                            operation.is_enabled 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {operation.is_enabled ? 'فعال' : 'غیرفعال'}
                          </div>
                          <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs ${
                            operation.is_available 
                              ? 'bg-blue-100 text-blue-800' 
                              : 'bg-gray-100 text-gray-800'
                          }`}>
                            {operation.is_available ? 'در دسترس' : 'غیرفعال'}
                          </div>
                          <div className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-800">
                            ترتیب {operation.order}
                          </div>
                          <div className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-800">
                            {operation.name}
                          </div>
                        </div>
                      </div>
                    
                    
                    
                                                               {/* Details and Status */}
                      <div className="space-y-3 mb-4">
                        
                        
                      </div>
                     
                                           {/* Action Buttons */}
                      <ActionButtons
                        onEdit={() => handleEditClick(operation)}
                        onDelete={() => handleDeleteClick(operation)}
                      />
                  </CardContent>
                </Card>
              ))
            )}
          </div>
      </div>

                           {/* Add Dialog */}
        <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
          <DialogContent className="text-right [&>button]:right-auto [&>button]:left-4">
            <DialogHeader className="text-right">
              <DialogTitle className="text-right">افزودن نوع عملیات جدید</DialogTitle>
            </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="add_name">نام انگلیسی</Label>
              <Input
                id="add_name"
                value={addForm.name}
                onChange={(e) => setAddForm({...addForm, name: e.target.value})}
                placeholder="مثال: new_operation"
              />
            </div>
            <div>
              <Label htmlFor="add_persian_name">نام فارسی</Label>
              <Input
                id="add_persian_name"
                value={addForm.persian_name}
                onChange={(e) => setAddForm({...addForm, persian_name: e.target.value})}
                placeholder="نام فارسی عملیات"
              />
            </div>
                                                                                                                                                                                                                                               <div>
                   <Label htmlFor="add_icon">آیکون</Label>
                   <Select value={addForm.icon} onValueChange={(value) => setAddForm({...addForm, icon: value})}>
                     <SelectTrigger className="w-full">
                       <SelectValue placeholder="انتخاب آیکون" />
                     </SelectTrigger>
                     <SelectContent>
                       {iconOptions.map(icon => (
                         <SelectItem key={icon.value} value={icon.value}>
                           <div className="flex items-center gap-2">
                             {React.createElement(icon.icon, { className: "w-4 h-4" })}
                             <span>{icon.label}</span>
                           </div>
                         </SelectItem>
                       ))}
                     </SelectContent>
                   </Select>
                 </div>
                                                     <div>
                 <Label htmlFor="add_color">رنگ</Label>
                 <Select value={addForm.color} onValueChange={(value) => setAddForm({...addForm, color: value})}>
                   <SelectTrigger className="w-full">
                     <SelectValue placeholder="انتخاب رنگ" />
                   </SelectTrigger>
                   <SelectContent>
                     {colorOptions.map(color => (
                       <SelectItem key={color.value} value={color.value}>
                         <div className="flex items-center gap-2">
                           <div className={`w-4 h-4 rounded-full bg-${color.value}-500`}></div>
                           <span>{color.label}</span>
                         </div>
                       </SelectItem>
                     ))}
                   </SelectContent>
                 </Select>
               </div>
            <div>
              <Label htmlFor="add_description">توضیحات</Label>
              <Input
                id="add_description"
                value={addForm.description}
                onChange={(e) => setAddForm({...addForm, description: e.target.value})}
                placeholder="توضیحات عملیات"
              />
            </div>
            <div>
              <Label htmlFor="add_order">ترتیب</Label>
              <Input
                id="add_order"
                type="number"
                value={addForm.order}
                onChange={(e) => setAddForm({...addForm, order: parseInt(e.target.value) || 0})}
                placeholder="0"
              />
            </div>
                         <div className="flex items-center space-x-4">
               <div className="flex items-center space-x-2">
                 <Checkbox
                   id="add_is_enabled"
                   checked={addForm.is_enabled}
                   onCheckedChange={(checked) => setAddForm({...addForm, is_enabled: checked})}
                 />
                 <Label htmlFor="add_is_enabled">فعال</Label>
               </div>
               <div className="flex items-center space-x-2">
                 <Checkbox
                   id="add_is_available"
                   checked={addForm.is_available}
                   onCheckedChange={(checked) => setAddForm({...addForm, is_available: checked})}
                 />
                 <Label htmlFor="add_is_available">در دسترس</Label>
               </div>
             </div>
            <div className="flex gap-2 pt-4">
              <Button onClick={handleAdd} className="flex-1 bg-green-600 hover:bg-green-700">
                افزودن
              </Button>
              <Button 
                variant="outline" 
                onClick={() => setShowAddDialog(false)}
                className="flex-1"
              >
                انصراف
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

                           {/* Edit Dialog */}
        <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
          <DialogContent className="text-right [&>button]:right-auto [&>button]:left-4">
                        <DialogHeader className="text-right">
               <DialogTitle className="text-right">ویرایش نوع عملیات</DialogTitle>
             </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="edit_persian_name">نام فارسی</Label>
              <Input
                id="edit_persian_name"
                value={editForm.persian_name}
                onChange={(e) => setEditForm({...editForm, persian_name: e.target.value})}
                placeholder="نام فارسی عملیات"
              />
            </div>
                                                                                                                                                                                                                                                                                                                                                                                                                               <div>
                   <Label htmlFor="edit_icon">آیکون</Label>
                   <Select value={editForm.icon} onValueChange={(value) => setEditForm({...editForm, icon: value})}>
                     <SelectTrigger className="w-full">
                       <SelectValue placeholder="انتخاب آیکون" />
                     </SelectTrigger>
                     <SelectContent>
                       {iconOptions.map(icon => (
                         <SelectItem key={icon.value} value={icon.value}>
                           <div className="flex items-center gap-2">
                             {React.createElement(icon.icon, { className: "w-4 h-4" })}
                             <span>{icon.label}</span>
                           </div>
                         </SelectItem>
                       ))}
                     </SelectContent>
                   </Select>
                 </div>
                                                     <div>
                 <Label htmlFor="edit_color">رنگ</Label>
                 <Select value={editForm.color} onValueChange={(value) => setEditForm({...editForm, color: value})}>
                   <SelectTrigger className="w-full">
                     <SelectValue placeholder="انتخاب رنگ" />
                   </SelectTrigger>
                   <SelectContent>
                     {colorOptions.map(color => (
                       <SelectItem key={color.value} value={color.value}>
                         <div className="flex items-center gap-2">
                           <div className={`w-4 h-4 rounded-full bg-${color.value}-500`}></div>
                           <span>{color.label}</span>
                         </div>
                       </SelectItem>
                     ))}
                   </SelectContent>
                 </Select>
               </div>
            <div>
              <Label htmlFor="edit_description">توضیحات</Label>
              <Input
                id="edit_description"
                value={editForm.description}
                onChange={(e) => setEditForm({...editForm, description: e.target.value})}
                placeholder="توضیحات عملیات"
              />
            </div>
            <div>
              <Label htmlFor="edit_order">ترتیب</Label>
              <Input
                id="edit_order"
                type="number"
                value={editForm.order}
                onChange={(e) => setEditForm({...editForm, order: parseInt(e.target.value) || 0})}
                placeholder="0"
              />
            </div>
                         <div className="flex items-center space-x-4">
               <div className="flex items-center space-x-2">
                 <Checkbox
                   id="edit_is_enabled"
                   checked={editForm.is_enabled}
                   onCheckedChange={(checked) => setEditForm({...editForm, is_enabled: checked})}
                 />
                 <Label htmlFor="edit_is_enabled">فعال</Label>
               </div>
               <div className="flex items-center space-x-2">
                 <Checkbox
                   id="edit_is_available"
                   checked={editForm.is_available}
                   onCheckedChange={(checked) => setEditForm({...editForm, is_available: checked})}
                 />
                 <Label htmlFor="edit_is_available">در دسترس</Label>
               </div>
             </div>
            <div className="flex gap-2 pt-4">
              <Button onClick={handleEdit} className="flex-1 bg-blue-600 hover:bg-blue-700">
                ذخیره
              </Button>
              <Button 
                variant="outline" 
                onClick={() => setShowEditDialog(false)}
                className="flex-1"
              >
                انصراف
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent className="text-right [&>button]:right-auto [&>button]:left-4">
          <DialogHeader>
            <DialogTitle>حذف نوع عملیات</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <p className="text-gray-600">
              آیا از حذف نوع عملیات "{selectedOperation?.persian_name}" اطمینان دارید؟
            </p>
            <p className="text-sm text-gray-500">
              این عملیات قابل بازگشت نیست.
            </p>
            <div className="flex gap-2 pt-4">
              <Button 
                onClick={handleDelete} 
                className="flex-1 bg-red-600 hover:bg-red-700"
              >
                حذف
              </Button>
              <Button 
                variant="outline" 
                onClick={() => setShowDeleteDialog(false)}
                className="flex-1"
              >
                انصراف
              </Button>
            </div>
                     </div>
                  </DialogContent>
        </Dialog>
        </div>
      </div>
    );
  } 