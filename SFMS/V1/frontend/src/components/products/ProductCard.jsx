import React from 'react';
import { Card } from '@/components/ui/card';
import ActionButtons from '@/components/ui/action-buttons';
import { Hash, FileText, Eye, Ruler } from 'lucide-react';
import Spinner from '@/components/Spinner';

const ProductCard = ({ 
  products, 
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

  if (products.length === 0) {
    return (
      <div className="col-span-full text-center py-8 text-gray-500">
        {searchTerm ? 'هیچ محصولی با این جستجو یافت نشد.' : 'هیچ محصولی یافت نشد.'}
      </div>
    );
  }

  return products.map((product) => (
    <Card key={product.id} className="p-4 hover:shadow-md transition-shadow duration-200">
      <div className="space-y-3">
        {/* نام انگلیسی */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-500">نام انگلیسی:</span>
          <div className="flex items-center gap-2">
            <Hash className="w-4 h-4 text-slate-400" />
            <span className="font-medium text-right">{product.name}</span>
          </div>
        </div>
        
        {/* نام فارسی */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-500">نام فارسی:</span>
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-slate-400" />
            <span className="text-sm text-gray-700 text-right">{product.persian_name}</span>
          </div>
        </div>
        
        {/* نام بینایی */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-500">نام بینایی:</span>
          <div className="flex items-center gap-2">
            <Eye className="w-4 h-4 text-slate-400" />
            <span className="text-sm text-gray-700 text-right">{product.vision_name}</span>
          </div>
        </div>
        
        {/* ابعاد */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-gray-500">ابعاد:</span>
          <div className="flex items-center gap-2">
            <Ruler className="w-4 h-4 text-slate-400" />
            <span className="text-sm text-gray-700 text-right">
              {product.width}×{product.length} - {product.gsm} گرم
            </span>
          </div>
        </div>
        
        {/* دکمه‌های عملیات */}
        <ActionButtons
          onEdit={() => onEditClick(product)}
          onDelete={() => onDeleteClick(product)}
          className="pt-2"
        />
      </div>
    </Card>
  ));
};

export default ProductCard; 