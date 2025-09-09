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

const ProductAddDialog = ({ 
  open, 
  onOpenChange, 
  formData, 
  onFormChange, 
  onSave, 
  onCancel 
}) => {
  // تغییر نام انگلیسی و تنظیم خودکار نام بینایی
  const handleNameChange = (value) => {
    onFormChange("name", value);
    onFormChange("vision_name", value.toLowerCase()); // تنظیم خودکار نام بینایی
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-white border-slate-200">
        <DialogHeader>
          <DialogTitle>افزودن محصول جدید</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label htmlFor="name">نام انگلیسی *</Label>
            <Input
              id="name"
              value={formData.name}
              onChange={(e) => handleNameChange(e.target.value)}
              placeholder="مثال: sulfat"
              className="bg-white border-slate-300"
            />
          </div>
          <div>
            <Label htmlFor="persian_name">نام فارسی *</Label>
            <Input
              id="persian_name"
              value={formData.persian_name}
              onChange={(e) => onFormChange("persian_name", e.target.value)}
              placeholder="مثال: سولفات"
              className="bg-white border-slate-300"
            />
          </div>
          <div>
            <Label htmlFor="vision_name">نام در سیستم بینایی</Label>
            <Input
              id="vision_name"
              value={formData.vision_name}
              onChange={(e) => onFormChange("vision_name", e.target.value)}
              placeholder="مثال: sulfat_vision"
              className="bg-white border-slate-300"
            />
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <Label htmlFor="width">عرض (mm)</Label>
              <Input
                id="width"
                type="number"
                value={formData.width}
                onChange={(e) => onFormChange("width", e.target.value)}
                placeholder="0"
                className="bg-white border-slate-300"
              />
            </div>
            <div>
              <Label htmlFor="gsm">گرماژ (gsm)</Label>
              <Input
                id="gsm"
                type="number"
                value={formData.gsm}
                onChange={(e) => onFormChange("gsm", e.target.value)}
                placeholder="0"
                className="bg-white border-slate-300"
              />
            </div>
            <div>
              <Label htmlFor="length">طول (m)</Label>
              <Input
                id="length"
                type="number"
                value={formData.length}
                onChange={(e) => onFormChange("length", e.target.value)}
                placeholder="0"
                className="bg-white border-slate-300"
              />
            </div>
          </div>
        </div>
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={onCancel}>
            انصراف
          </Button>
          <Button onClick={onSave} className="bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700">
            افزودن
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ProductAddDialog; 