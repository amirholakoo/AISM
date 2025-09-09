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
import { Checkbox } from '@/components/ui/checkbox';

const WarehouseEditDialog = ({ 
  open, 
  onOpenChange, 
  editForm, 
  onEditFormChange, 
  onSave, 
  onCancel 
}) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="text-right [&>button]:right-auto [&>button]:left-4">
        <DialogHeader className="text-right">
          <DialogTitle className="text-right">ویرایش انبار</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label htmlFor="persian_name">نام فارسی</Label>
            <Input
              id="persian_name"
              value={editForm.persian_name}
              onChange={(e) => onEditFormChange({
                ...editForm,
                persian_name: e.target.value
              })}
              placeholder="نام فارسی انبار"
            />
          </div>



          <div className="flex items-center space-x-2">
            <Checkbox
              id="is_active"
              checked={editForm.is_active}
              onCheckedChange={(checked) => onEditFormChange({
                ...editForm,
                is_active: checked
              })}
            />
            <Label htmlFor="is_active">فعال</Label>
          </div>
          <div className="flex gap-2 pt-4">
            <Button 
              onClick={onSave} 
              className="flex-1 bg-blue-600 hover:bg-blue-700 border border-blue-600 hover:border-blue-700 transition-all duration-200"
            >
              ذخیره
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

export default WarehouseEditDialog; 