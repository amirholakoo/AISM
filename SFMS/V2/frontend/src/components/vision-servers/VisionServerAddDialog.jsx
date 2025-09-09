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

          {/* منبع ویدیو */}
          <div className="space-y-2">
            <Label htmlFor="video_source" className="text-right">منبع ویدیو *</Label>
            <div className="space-y-2">
              {/* Quick Select Buttons */}
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => onFormChange('video_source', 'picamera')}
                  className={`px-3 py-1 text-xs rounded-md border transition-colors ${
                    formData.video_source === 'picamera' 
                      ? 'bg-blue-100 border-blue-300 text-blue-700' 
                      : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  PiCamera
                </button>
                <button
                  type="button"
                  onClick={() => onFormChange('video_source', '0')}
                  className={`px-3 py-1 text-xs rounded-md border transition-colors ${
                    formData.video_source === '0' 
                      ? 'bg-blue-100 border-blue-300 text-blue-700' 
                      : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  USB (0)
                </button>
                <button
                  type="button"
                  onClick={() => onFormChange('video_source', '1')}
                  className={`px-3 py-1 text-xs rounded-md border transition-colors ${
                    formData.video_source === '1' 
                      ? 'bg-blue-100 border-blue-300 text-blue-700' 
                      : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  USB (1)
                </button>
                <button
                  type="button"
                  onClick={() => onFormChange('video_source', 'rtsp://192.168.1.100:554/stream')}
                  className={`px-3 py-1 text-xs rounded-md border transition-colors ${
                    formData.video_source?.startsWith('rtsp://') 
                      ? 'bg-blue-100 border-blue-300 text-blue-700' 
                      : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  RTSP Template
                </button>
              </div>
              
              {/* Custom Input Field */}
              <Input
                id="video_source"
                value={formData.video_source}
                onChange={(e) => onFormChange('video_source', e.target.value)}
                placeholder="مثال: rtsp://192.168.1.100:554/stream یا picamera یا 0"
                required
                className="text-left"
                dir="ltr"
              />
              
              {/* Help Text */}
              <div className="text-xs text-gray-500 text-right">
                <p>نمونه‌ها:</p>
                <ul className="list-disc list-inside mr-2 space-y-1">
                  <li><code className="bg-gray-100 px-1 rounded">picamera</code> - دوربین رزبری پای</li>
                  <li><code className="bg-gray-100 px-1 rounded">0</code> یا <code className="bg-gray-100 px-1 rounded">1</code> - دوربین USB</li>
                  <li><code className="bg-gray-100 px-1 rounded">rtsp://IP:PORT/stream</code> - جریان RTSP</li>
                  <li><code className="bg-gray-100 px-1 rounded">/path/to/video.mp4</code> - فایل ویدیو</li>
                </ul>
              </div>
            </div>
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