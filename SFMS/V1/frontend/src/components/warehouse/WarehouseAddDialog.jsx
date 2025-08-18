import React from 'react';
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle 
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

const WarehouseAddDialog = ({ 
  open, 
  onOpenChange, 
  addForm, 
  onAddFormChange, 
  onAdd, 
  onCancel 
}) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="text-right [&>button]:right-auto [&>button]:left-4">
        <DialogHeader className="text-right">
          <DialogTitle className="text-right">افزودن انبار جدید</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label htmlFor="add_id">شناسه انبار (نام جدول خارجی)</Label>
            <Input
              id="add_id"
              value={addForm.id}
              onChange={(e) => onAddFormChange({
                ...addForm,
                id: e.target.value
              })}
              placeholder="مثال: Anbar_New"
            />
          </div>
          <div>
            <Label htmlFor="add_name">نام انگلیسی</Label>
            <Input
              id="add_name"
              value={addForm.name}
              onChange={(e) => onAddFormChange({
                ...addForm,
                name: e.target.value
              })}
              placeholder="نام انگلیسی انبار"
            />
          </div>
          <div>
            <Label htmlFor="add_persian_name">نام فارسی</Label>
            <Input
              id="add_persian_name"
              value={addForm.persian_name}
              onChange={(e) => onAddFormChange({
                ...addForm,
                persian_name: e.target.value
              })}
              placeholder="نام فارسی انبار"
            />
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="add_is_active"
              checked={addForm.is_active}
              onChange={(e) => onAddFormChange({
                ...addForm,
                is_active: e.target.checked
              })}
              className="rounded"
            />
            <Label htmlFor="add_is_active">فعال</Label>
          </div>
          <div className="flex gap-2 pt-4">
            <Button 
              onClick={onAdd} 
              className="flex-1 bg-green-600 hover:bg-green-700 border border-green-600 hover:border-green-700 transition-all duration-200"
            >
              افزودن
            </Button>
            <Button 
              variant="outline" 
              onClick={onCancel}
              className="flex-1 border-gray-300 hover:border-gray-400 hover:bg-gray-50 transition-all duration-200"
            >
              انصراف
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default WarehouseAddDialog; 