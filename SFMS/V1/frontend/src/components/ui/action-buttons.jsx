import React from 'react';
import { Button } from '@/components/ui/button';
import { Edit, X } from 'lucide-react';

const ActionButtons = ({ 
  onEdit, 
  onDelete, 
  editText = "ویرایش", 
  deleteText = "حذف",
  className = ""
}) => {
  return (
    <div className={`flex gap-2 pt-4 border-t border-gray-100 ${className}`}>
      <Button
        variant="outline"
        size="sm"
        onClick={onEdit}
        className="flex-1 flex items-center justify-center gap-2 px-3 py-1 h-8 border-blue-300 text-blue-600 hover:bg-blue-600 hover:text-white hover:border-blue-600 hover:shadow-md transition-all duration-200"
      >
        <Edit className="h-4 w-4" />
        <span className="text-xs">{editText}</span>
      </Button>
      <Button
        variant="outline"
        size="sm"
        onClick={onDelete}
        className="flex-1 flex items-center justify-center gap-2 px-3 py-1 h-8 border-red-300 text-red-600 hover:bg-red-600 hover:text-white hover:border-red-600 hover:shadow-md transition-all duration-200"
      >
        <X className="h-4 w-4" />
        <span className="text-xs">{deleteText}</span>
      </Button>
    </div>
  );
};

export default ActionButtons; 