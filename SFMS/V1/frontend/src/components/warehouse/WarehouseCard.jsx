import React from 'react';
import { Card } from '@/components/ui/card';
import ActionButtons from '@/components/ui/action-buttons';
import Spinner from '@/components/Spinner';

const WarehouseCard = ({ 
  warehouses, 
  loading, 
  searchTerm, 
  onEditClick, 
  onDeleteClick 
}) => {
  if (loading) {
    return (
      <div className="col-span-full text-center py-8">
        <div className="flex items-center justify-center gap-2">
          <Spinner className="w-4 h-4 text-blue-600" />
          در حال بارگذاری...
        </div>
      </div>
    );
  }

  if (warehouses.length === 0) {
    return (
      <div className="col-span-full text-center py-8 text-gray-500">
        {searchTerm ? 'هیچ انباری با این جستجو یافت نشد.' : 'هیچ انباری یافت نشد.'}
      </div>
    );
  }

  return warehouses.map((warehouse) => (
    <Card key={warehouse.id} className="p-4 hover:shadow-md transition-shadow duration-200">
      <div className="space-y-3">
        {/* نام فارسی */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-500">نام فارسی:</span>
          <span className="font-medium text-right">{warehouse.persian_name || warehouse.name}</span>
        </div>
        
        {/* نام انگلیسی */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-500">نام انگلیسی:</span>
          <span className="text-sm text-gray-700 text-right">{warehouse.name}</span>
        </div>
        
        {/* جدول خارجی */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-500">جدول خارجی:</span>
          <span className="text-sm text-gray-700 text-right">{warehouse.id}</span>
        </div>
        

        
        {/* وضعیت */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-500">وضعیت:</span>
          <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs ${
            warehouse.is_active 
              ? 'bg-green-100 text-green-800' 
              : 'bg-red-100 text-red-800'
          }`}>
            {warehouse.is_active ? 'فعال' : 'غیرفعال'}
          </div>
        </div>
        
        {/* دکمه‌های عملیات */}
        <ActionButtons
          onEdit={() => onEditClick(warehouse)}
          onDelete={() => onDeleteClick(warehouse)}
          className="pt-2"
        />
      </div>
    </Card>
  ));
};

export default WarehouseCard; 