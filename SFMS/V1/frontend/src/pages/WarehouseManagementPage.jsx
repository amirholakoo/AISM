import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { showSuccess, showError } from '@/lib/toast';
import { API_ENDPOINTS } from '@/config';
import {
  WarehouseHeader,
  WarehouseSearchCard,
  WarehouseCard,
  WarehouseEditDialog,
  WarehouseDeleteDialog,
  WarehouseAddDialog
} from '@/components/warehouse';

const WarehouseManagementPage = () => {
  const [warehouses, setWarehouses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [editingWarehouse, setEditingWarehouse] = useState(null);
  const [editForm, setEditForm] = useState({
    persian_name: '',
    is_active: true
  });
  const [warehouseToDelete, setWarehouseToDelete] = useState(null);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [addForm, setAddForm] = useState({
    id: '',
    name: '',
    persian_name: '',
    is_active: true
  });

  // فیلتر کردن انبارها بر اساس جستجو
  const filteredWarehouses = warehouses.filter(warehouse =>
    warehouse.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    warehouse.persian_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    warehouse.id?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // دریافت لیست انبارها
  const fetchWarehouses = async () => {
    try {
      setLoading(true);
      const response = await fetch(API_ENDPOINTS.WAREHOUSES);
      const data = await response.json();
      setWarehouses(data.warehouses || []);
    } catch (error) {
      showError('خطا در دریافت لیست انبارها');
      console.error('Error fetching warehouses:', error);
    } finally {
      setLoading(false);
    }
  };

  // همگام‌سازی انبارها از external database
  const syncWarehouses = async () => {
    try {
      setSyncing(true);
      const response = await fetch(API_ENDPOINTS.WAREHOUSES_SYNC, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      const data = await response.json();
      
      if (data.success) {
        showSuccess(data.message);
        fetchWarehouses(); // بروزرسانی لیست
      } else {
        showError(data.message);
      }
    } catch (error) {
      showError('خطا در همگام‌سازی انبارها');
      console.error('Error syncing warehouses:', error);
    } finally {
      setSyncing(false);
    }
  };

  // ویرایش انبار
  const handleEdit = (warehouse) => {
    setEditingWarehouse(warehouse);
    setEditForm({
      persian_name: warehouse.persian_name || '',
      is_active: warehouse.is_active
    });
    setShowEditDialog(true);
  };

  // ذخیره تغییرات
  const handleSave = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.WAREHOUSE_UPDATE(editingWarehouse.id), {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(editForm)
      });
      
      const data = await response.json();
      
      if (data.success) {
        showSuccess(data.message);
        setEditingWarehouse(null);
        setShowEditDialog(false);
        fetchWarehouses(); // بروزرسانی لیست
      } else {
        showError(data.message);
      }
    } catch (error) {
      showError('خطا در ذخیره تغییرات');
      console.error('Error updating warehouse:', error);
    }
  };

  // حذف انبار
  const handleDelete = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.WAREHOUSE_DELETE(warehouseToDelete.id), {
        method: 'DELETE'
      });
      
      const data = await response.json();
      
      if (data.success) {
        showSuccess(data.message);
        setWarehouseToDelete(null);
        setShowDeleteDialog(false);
        fetchWarehouses(); // بروزرسانی لیست
      } else {
        showError(data.message);
      }
    } catch (error) {
      showError('خطا در حذف انبار');
      console.error('Error deleting warehouse:', error);
    }
  };

  // حذف انبار (trigger)
  const handleDeleteClick = (warehouse) => {
    setWarehouseToDelete(warehouse);
    setShowDeleteDialog(true);
  };

  // افزودن انبار جدید
  const handleAdd = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.WAREHOUSES, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(addForm)
      });
      
      const data = await response.json();
      
      if (data.success) {
        showSuccess(data.message);
        setShowAddDialog(false);
        setAddForm({
          id: '',
          name: '',
          persian_name: '',
          is_active: true,
          vision_server_url: 'http://localhost:5001'
        });
        fetchWarehouses(); // بروزرسانی لیست
      } else {
        showError(data.message);
      }
    } catch (error) {
      showError('خطا در افزودن انبار');
      console.error('Error adding warehouse:', error);
    }
  };

  useEffect(() => {
    fetchWarehouses();
  }, []);

  return (
    <>
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <WarehouseHeader 
            onAddClick={() => setShowAddDialog(true)}
            onSyncClick={syncWarehouses}
            syncing={syncing}
            warehouseCount={warehouses.length}
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="min-h-screen bg-slate-50">
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Search Card */}
        <WarehouseSearchCard 
          searchTerm={searchTerm}
          onSearchChange={setSearchTerm}
        />

        {/* Warehouse List */}
        <Card>
          <CardContent>
            {/* Warehouse Cards - Both Desktop and Mobile */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <WarehouseCard 
                warehouses={filteredWarehouses}
                loading={loading}
                searchTerm={searchTerm}
                onEditClick={handleEdit}
                onDeleteClick={handleDeleteClick}
              />
            </div>
          </CardContent>
        </Card>
      </main>

      {/* Edit Warehouse Dialog */}
      <WarehouseEditDialog 
        open={showEditDialog}
        onOpenChange={setShowEditDialog}
        editForm={editForm}
        onEditFormChange={setEditForm}
        onSave={handleSave}
        onCancel={() => setShowEditDialog(false)}
      />

      {/* Delete Warehouse Dialog */}
      <WarehouseDeleteDialog 
        open={showDeleteDialog}
        onOpenChange={setShowDeleteDialog}
        warehouseToDelete={warehouseToDelete}
        onDelete={handleDelete}
        onCancel={() => setShowDeleteDialog(false)}
      />

      {/* Add Warehouse Dialog */}
      <WarehouseAddDialog 
        open={showAddDialog}
        onOpenChange={setShowAddDialog}
        addForm={addForm}
        onAddFormChange={setAddForm}
        onAdd={handleAdd}
        onCancel={() => setShowAddDialog(false)}
      />
        </div>
      </>
    );
  };

export default WarehouseManagementPage; 