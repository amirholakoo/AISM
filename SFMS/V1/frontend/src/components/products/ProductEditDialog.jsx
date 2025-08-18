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

const ProductEditDialog = ({ 
  open, 
  onOpenChange, 
  formData, 
  onFormChange, 
  onSave, 
  onCancel 
}) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-white border-slate-200">
        <DialogHeader>
          <DialogTitle className="text-right">ویرایش محصول</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label htmlFor="edit-name">نام انگلیسی</Label>
            <Input
              id="edit-name"
              value={formData.name}
              onChange={(e) => onFormChange("name", e.target.value)}
              className="bg-white border-slate-300"
            />
          </div>
          <div>
            <Label htmlFor="edit-persian_name">نام فارسی</Label>
            <Input
              id="edit-persian_name"
              value={formData.persian_name}
              onChange={(e) => onFormChange("persian_name", e.target.value)}
              className="bg-white border-slate-300"
            />
          </div>
          <div>
            <Label htmlFor="edit-vision_name">نام بینایی</Label>
            <Input
              id="edit-vision_name"
              value={formData.vision_name}
              onChange={(e) => onFormChange("vision_name", e.target.value)}
              className="bg-white border-slate-300"
            />
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <Label htmlFor="edit-width">عرض (cm)</Label>
              <Input
                id="edit-width"
                type="number"
                value={formData.width}
                onChange={(e) => onFormChange("width", e.target.value)}
                className="bg-white border-slate-300"
              />
            </div>
            <div>
              <Label htmlFor="edit-length">طول (cm)</Label>
              <Input
                id="edit-length"
                type="number"
                value={formData.length}
                onChange={(e) => onFormChange("length", e.target.value)}
                className="bg-white border-slate-300"
              />
            </div>
            <div>
              <Label htmlFor="edit-gsm">گرم (g/m²)</Label>
              <Input
                id="edit-gsm"
                type="number"
                value={formData.gsm}
                onChange={(e) => onFormChange("gsm", e.target.value)}
                className="bg-white border-slate-300"
              />
            </div>
          </div>
        </div>
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={onCancel}>
            انصراف
          </Button>
          <Button onClick={onSave} className="bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700">
            ذخیره
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ProductEditDialog; 