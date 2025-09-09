import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';

const VisionServerAddDialog = ({
  open,
  onOpenChange,
  formData,
  onFormChange,
  onSave,
  onCancel,
  loading = false
}) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    onSave();
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle className="text-right">افزودن سرور بینایی جدید</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* نام انگلیسی */}
          <div className="space-y-2">
            <Label htmlFor="name" className="text-right">نام انگلیسی *</Label>
            <Input
              id="name"
              value={formData.name}
              onChange={(e) => onFormChange('name', e.target.value)}
              placeholder="نام انگلیسی سرور"
              required
            />
          </div>

          {/* نام فارسی */}
          <div className="space-y-2">
            <Label htmlFor="persian_name" className="text-right">نام فارسی</Label>
            <Input
              id="persian_name"
              value={formData.persian_name}
              onChange={(e) => onFormChange('persian_name', e.target.value)}
              placeholder="نام فارسی سرور"
            />
          </div>

          {/* URL */}
          <div className="space-y-2">
            <Label htmlFor="url" className="text-right">آدرس سرور *</Label>
            <Input
              id="url"
              value={formData.url}
              onChange={(e) => onFormChange('url', e.target.value)}
              placeholder="http://example.com:8080"
              required
            />
          </div>

          {/* نوع عملیات */}
          <div className="space-y-2">
            <Label htmlFor="type" className="text-right">نوع عملیات *</Label>
            <Select value={formData.type} onValueChange={(value) => onFormChange('type', value)}>
              <SelectTrigger>
                <SelectValue placeholder="انتخاب نوع عملیات" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="unloading">تخلیه</SelectItem>
                <SelectItem value="loading">بارگیری</SelectItem>
                <SelectItem value="consumption">مصرف</SelectItem>
                <SelectItem value="transfer">انتقال</SelectItem>
                <SelectItem value="return">بازگشت</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* وضعیت فعال */}
          <div className="flex items-center space-x-2 space-x-reverse">
            <Checkbox
              id="is_active"
              checked={formData.is_active}
              onCheckedChange={(checked) => onFormChange('is_active', checked)}
            />
            <Label htmlFor="is_active" className="text-sm">فعال</Label>
          </div>

          {/* دکمه‌ها */}
          <div className="flex justify-end gap-2 pt-4">
            <Button type="button" variant="outline" onClick={onCancel} disabled={loading}>
              انصراف
            </Button>
            <Button type="submit" disabled={loading}>
              {loading ? 'در حال ذخیره...' : 'ذخیره'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default VisionServerAddDialog; 