import React from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';

const WarehouseDeleteDialog = ({ 
  open, 
  onOpenChange, 
  warehouseToDelete, 
  onDelete, 
  onCancel 
}) => {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent className="text-right">
        <AlertDialogHeader className="text-right">
          <AlertDialogTitle className="text-right">حذف انبار</AlertDialogTitle>
          <AlertDialogDescription className="text-right">
            آیا از حذف انبار <strong>{warehouseToDelete?.persian_name || warehouseToDelete?.name}</strong> اطمینان دارید؟
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter className="flex gap-2">
          <AlertDialogCancel 
            onClick={onCancel}
            className="border-gray-300 hover:border-gray-400 hover:bg-gray-50 transition-all duration-200"
          >
            انصراف
          </AlertDialogCancel>
          <AlertDialogAction 
            onClick={onDelete}
            className="bg-red-600 hover:bg-red-700 border border-red-600 hover:border-red-700 transition-all duration-200"
          >
            حذف
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};

export default WarehouseDeleteDialog; 