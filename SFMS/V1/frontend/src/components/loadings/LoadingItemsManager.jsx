import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Plus, X, Edit, Trash2, QrCode, PackageIcon } from "lucide-react";
import { API_ENDPOINTS } from "@/config";
import AlertManager from "@/components/AlertManager";

export default function LoadingItemsManager({ 
  loadingId, 
  canEdit = true,
  onItemsChange 
}) {
  const [items, setItems] = useState([]);
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [editingIndex, setEditingIndex] = useState(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    reel_number: '',
    width: '',
    gsm: '',
    length: '',
    breaks: 0,
    grade: ''
  });
  const [alert, setAlert] = useState({ show: false, message: '', type: 'info' });

  // Load products for the dropdown
  useEffect(() => {
    const loadProducts = async () => {
      try {
        const response = await fetch(API_ENDPOINTS.PRODUCTS);
        const data = await response.json();
        if (data.success) {
          setProducts(data.products || []);
        }
      } catch (error) {
        console.error('Error loading products:', error);
      }
    };
    loadProducts();
  }, []);

  // Load loading items
  useEffect(() => {
    if (loadingId) {
      loadItems();
    }
  }, [loadingId]);

  const loadItems = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.LOADING_ITEMS(loadingId));
      const data = await response.json();
      if (data.success) {
        setItems(data.items || []);
        if (onItemsChange) {
          onItemsChange(data.items || []);
        }
      }
    } catch (error) {
      console.error('Error loading items:', error);
      showAlert('خطا در بارگذاری آیتم‌ها', 'error');
    }
  };

  const showAlert = (message, type = 'info') => {
    setAlert({ show: true, message, type });
    setTimeout(() => setAlert({ show: false, message: '', type: 'info' }), 5000);
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.name.trim()) {
      showAlert('نام محصول الزامی است', 'error');
      return;
    }
    
    setLoading(true);
    
    try {
      if (editingIndex !== null) {
        // Edit existing item
        const item = items[editingIndex];
        const response = await fetch(API_ENDPOINTS.LOADING_ITEM_UPDATE(item.id), {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        if (data.success) {
          showAlert('آیتم با موفقیت ویرایش شد', 'success');
          await loadItems();
          setEditingIndex(null);
        } else {
          showAlert(data.message || 'خطا در ویرایش آیتم', 'error');
        }
      } else {
        // Add new item
        const response = await fetch(API_ENDPOINTS.LOADING_ITEMS(loadingId), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        if (data.success) {
          showAlert('آیتم با موفقیت اضافه شد', 'success');
          await loadItems();
          setShowAddForm(false);
        } else {
          showAlert(data.message || 'خطا در اضافه کردن آیتم', 'error');
        }
      }
      
      // Reset form
      setFormData({
        name: '',
        reel_number: '',
        width: '',
        gsm: '',
        length: '',
        breaks: 0,
        grade: ''
      });
    } catch (error) {
      console.error('Error saving item:', error);
      showAlert('خطا در ذخیره آیتم', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = (index) => {
    setEditingIndex(index);
    setFormData({ ...items[index] });
    setShowAddForm(true);
  };

  const handleCancelEdit = () => {
    setEditingIndex(null);
    setFormData({
      name: '',
      reel_number: '',
      width: '',
      gsm: '',
      length: '',
      breaks: 0,
      grade: ''
    });
    setShowAddForm(false);
  };

  const handleDelete = async (index) => {
    const item = items[index];
    if (!confirm(`آیا از حذف آیتم "${item.name}" اطمینان دارید؟`)) {
      return;
    }
    
    setLoading(true);
    
    try {
      const response = await fetch(API_ENDPOINTS.LOADING_ITEM_DELETE(item.id), {
        method: 'DELETE'
      });
      
      const data = await response.json();
      if (data.success) {
        showAlert('آیتم با موفقیت حذف شد', 'success');
        await loadItems();
      } else {
        showAlert(data.message || 'خطا در حذف آیتم', 'error');
      }
    } catch (error) {
      console.error('Error deleting item:', error);
      showAlert('خطا در حذف آیتم', 'error');
    } finally {
      setLoading(false);
    }
  };

  const getSourceBadge = (source) => {
    if (source === 'vision') {
      return (
        <div className="flex items-center gap-1">
          <QrCode className="w-3 h-3" />
          <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">بینایی</span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-1">
        <PackageIcon className="w-3 h-3" />
        <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">کاربر</span>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <AlertManager alert={alert} setAlert={setAlert} />
      
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-slate-800">مدیریت آیتم‌های بارگیری</h3>
        {canEdit && !showAddForm && (
          <Button
            onClick={() => setShowAddForm(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white"
          >
            <Plus className="w-4 h-4 ml-2" />
            افزودن آیتم جدید
          </Button>
        )}
      </div>

      {/* Add/Edit Form */}
      {canEdit && showAddForm && (
        <Card className="bg-blue-50/50 border-2 border-blue-700">
          <CardHeader className="pb-3">
            <CardTitle className="text-center text-blue-800 text-lg font-bold flex items-center justify-center gap-2">
              <Plus className="w-5 h-5" />
              {editingIndex !== null ? 'ویرایش آیتم بارگیری' : 'افزودن آیتم بارگیری جدید'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Product Name */}
                <div className="space-y-2">
                  <Label htmlFor="name" className="text-blue-800 font-semibold">
                    نام محصول *
                  </Label>
                  <Select
                    value={formData.name}
                    onValueChange={(value) => handleInputChange('name', value)}
                    disabled={editingIndex !== null}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="انتخاب محصول" />
                    </SelectTrigger>
                    <SelectContent>
                      {products.map((product) => (
                        <SelectItem key={product.id} value={product.name}>
                          {product.persian_name || product.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>



                {/* Reel Number */}
                <div className="space-y-2">
                  <Label htmlFor="reel_number" className="text-blue-800 font-semibold">
                    شماره رول
                  </Label>
                  <Input
                    id="reel_number"
                    type="text"
                    value={formData.reel_number}
                    onChange={(e) => handleInputChange('reel_number', e.target.value)}
                    placeholder="مثال: 1482"
                    className="border-blue-300 focus:border-blue-500"
                  />
                </div>

                {/* Width */}
                <div className="space-y-2">
                  <Label htmlFor="width" className="text-blue-800 font-semibold">
                    عرض (mm)
                  </Label>
                  <Input
                    id="width"
                    type="number"
                    min="0"
                    value={formData.width}
                    onChange={(e) => handleInputChange('width', parseInt(e.target.value) || '')}
                    placeholder="مثال: 240"
                    className="border-blue-300 focus:border-blue-500"
                  />
                </div>

                {/* GSM */}
                <div className="space-y-2">
                  <Label htmlFor="gsm" className="text-blue-800 font-semibold">
                    گرماژ (gsm)
                  </Label>
                  <Input
                    id="gsm"
                    type="number"
                    min="0"
                    value={formData.gsm}
                    onChange={(e) => handleInputChange('gsm', parseInt(e.target.value) || '')}
                    placeholder="مثال: 130"
                    className="border-blue-300 focus:border-blue-500"
                  />
                </div>

                {/* Length */}
                <div className="space-y-2">
                  <Label htmlFor="length" className="text-blue-800 font-semibold">
                    طول (m)
                  </Label>
                  <Input
                    id="length"
                    type="number"
                    min="0"
                    value={formData.length}
                    onChange={(e) => handleInputChange('length', parseInt(e.target.value) || '')}
                    placeholder="مثال: 6300"
                    className="border-blue-300 focus:border-blue-500"
                  />
                </div>

                {/* Breaks */}
                <div className="space-y-2">
                  <Label htmlFor="breaks" className="text-blue-800 font-semibold">
                    تعداد توقف
                  </Label>
                  <Input
                    id="breaks"
                    type="number"
                    min="0"
                    value={formData.breaks}
                    onChange={(e) => handleInputChange('breaks', parseInt(e.target.value) || '')}
                    placeholder="مثال: 2"
                    className="border-blue-300 focus:border-blue-500"
                  />
                </div>


                {/* Grade */}
                <div className="space-y-2">
                  <Label htmlFor="grade" className="text-blue-800 font-semibold">
                    درجه/کیفیت
                  </Label>
                  <Input
                    id="grade"
                    type="text"
                    value={formData.grade}
                    onChange={(e) => handleInputChange('grade', e.target.value)}
                    placeholder="مثال: Testliner HOMAYOUN"
                    className="border-blue-300 focus:border-blue-500"
                  />
                </div>
              </div>

              <div className="flex gap-3 justify-center pt-4">
                <Button
                  type="submit"
                  disabled={loading}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-6"
                >
                  {loading ? 'در حال ذخیره...' : (editingIndex !== null ? 'ویرایش آیتم' : 'افزودن آیتم')}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleCancelEdit}
                  className="border-blue-300 text-blue-700 hover:bg-blue-50"
                >
                  لغو
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      )}

      {/* Display existing items */}
      {items.length > 0 && (
        <Card className="bg-white border-2 border-blue-200">
          <CardHeader className="pb-3">
            <CardTitle className="text-center text-blue-800 text-lg font-bold">
              آیتم‌های بارگیری ({items.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {items.map((item, index) => (
                <div
                  key={`${item.id}-${item.name}-${index}`}
                  className="bg-blue-50 rounded-lg border border-blue-200 p-4"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <h3 className="font-semibold text-blue-800">
                        {item.name}
                      </h3>
                      {getSourceBadge(item.source)}
                    </div>
                    {canEdit && (
                      <div className="flex gap-2">
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => handleEdit(index)}
                          className="text-blue-600 hover:bg-blue-100"
                        >
                          <Edit className="w-4 h-4" />
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDelete(index)}
                          className="text-red-600 hover:bg-red-100"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    {item.reel_number && (
                      <div>
                        <span className="font-medium text-blue-700">شماره رول:</span>
                        <span className="mr-2 text-blue-800">{item.reel_number}</span>
                      </div>
                    )}
                    {item.width && (
                      <div>
                        <span className="font-medium text-blue-700">عرض:</span>
                        <span className="mr-2 text-blue-800">{item.width} mm</span>
                      </div>
                    )}
                    {item.gsm && (
                      <div>
                        <span className="font-medium text-blue-700">گرماژ:</span>
                        <span className="mr-2 text-blue-800">{item.gsm} gsm</span>
                      </div>
                    )}
                    {item.length && (
                      <div>
                        <span className="font-medium text-blue-700">طول:</span>
                        <span className="mr-2 text-blue-800">{item.length} m</span>
                      </div>
                    )}
                    {item.breaks && (
                      <div>
                        <span className="font-medium text-blue-700">تعداد توقف:</span>
                        <span className="mr-2 text-blue-800">{item.breaks}</span>
                      </div>
                    )}

                    {item.grade && (
                      <div>
                        <span className="font-medium text-blue-700">درجه:</span>
                        <span className="mr-2 text-blue-800">{item.grade}</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Empty state */}
      {items.length === 0 && !showAddForm && (
        <Card className="bg-slate-50 border-2 border-dashed border-slate-300">
          <CardContent className="text-center py-8">
            <QrCode className="w-12 h-12 text-slate-400 mx-auto mb-3" />
            <p className="text-slate-500 text-sm">هیچ آیتم بارگیری‌ای موجود نیست</p>
            <p className="text-slate-400 text-xs">آیتم‌ها از QR codes یا به صورت دستی اضافه می‌شوند</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
