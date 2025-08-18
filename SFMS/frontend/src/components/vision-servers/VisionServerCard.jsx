import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import DeleteConfirmDialog from '@/components/DeleteConfirmDialog';
import { Eye, Globe, Tag, Edit } from 'lucide-react';

const VisionServerCard = ({ 
  servers, 
  loading, 
  searchTerm, 
  onEditClick, 
  onDeleteClick 
}) => {
  if (loading) {
    return (
      <div className="col-span-full text-center py-8">
        <div className="flex items-center justify-center gap-2">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-600"></div>
          در حال بارگذاری...
        </div>
      </div>
    );
  }

  if (servers.length === 0) {
    return (
      <div className="col-span-full text-center py-8 text-gray-500">
        {searchTerm ? 'هیچ سروری با این جستجو یافت نشد.' : 'هیچ سروری یافت نشد.'}
      </div>
    );
  }

  return servers.map((server) => (
    <Card key={server.id} className="p-4 hover:shadow-md transition-shadow duration-200">
      <div className="space-y-3">
        {/* نام انگلیسی */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-500">نام انگلیسی:</span>
          <div className="flex items-center gap-2">
            <Eye className="w-4 h-4 text-slate-400" />
            <span className="font-medium text-right">{server.name}</span>
          </div>
        </div>
        
        {/* نام فارسی */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-500">نام فارسی:</span>
          <div className="flex items-center gap-2">
            <Tag className="w-4 h-4 text-slate-400" />
            <span className="text-sm text-gray-700 text-right">{server.persian_name || '-'}</span>
          </div>
        </div>
        
        {/* URL */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-500">آدرس سرور:</span>
          <div className="flex items-center gap-2">
            <Globe className="w-4 h-4 text-slate-400" />
            <span className="text-sm text-gray-700 text-right max-w-32 truncate" title={server.url}>
              {server.url}
            </span>
          </div>
        </div>
        
        {/* وضعیت */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-500">وضعیت:</span>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${server.is_active ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className={`text-sm ${server.is_active ? 'text-green-600' : 'text-red-600'}`}>
              {server.is_active ? 'فعال' : 'غیرفعال'}
            </span>
          </div>
        </div>
        
        {/* دکمه‌های عملیات */}
        <div className="flex gap-2 pt-4 border-t border-gray-100">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onEditClick(server)}
            className="flex-1 flex items-center justify-center gap-2 px-3 py-1 h-8 border-blue-300 text-blue-600 hover:bg-blue-600 hover:text-white hover:border-blue-600 hover:shadow-md transition-all duration-200"
          >
            <Edit className="h-4 w-4" />
            <span className="text-xs">ویرایش</span>
          </Button>
          <DeleteConfirmDialog
            itemName={server.name}
            itemType="vision-server"
            onConfirm={() => onDeleteClick(server)}
          >
            <Button
              variant="outline"
              size="sm"
              className="flex-1 flex items-center justify-center gap-2 px-3 py-1 h-8 border-red-300 text-red-600 hover:bg-red-600 hover:text-white hover:border-red-600 hover:shadow-md transition-all duration-200"
            >
              <span className="text-xs">حذف</span>
            </Button>
          </DeleteConfirmDialog>
        </div>
      </div>
    </Card>
  ));
};

export default VisionServerCard; 