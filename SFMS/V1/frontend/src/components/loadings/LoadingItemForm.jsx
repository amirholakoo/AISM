import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Plus, Trash2, Package } from "lucide-react";
import DeleteConfirmDialog from "@/components/DeleteConfirmDialog";

export default function LoadingItemForm({ 
  items, 
  onItemsChange, 
  products, 
  canEdit = true,
  onAddItem,
  onDeleteItem,
  onEditItem
}) {
  const handleDelete = (index) => {
    onDeleteItem(index);
  };

  const handleItemFieldChange = (index, field, value) => {
    const newItems = [...items];
    newItems[index] = {
      ...newItems[index],
      [field]: value
    };
    onItemsChange(newItems);
  };

  const handleAddEmptyItem = () => {
    const newItem = {
      name: '',
      reel_number: '',
      width: '',
      gsm: '',
      length: '',
      breaks: 0,
      grade: '',
      version: 1
    };
    onAddItem(newItem);
  };

  return (
    <div className="space-y-6">
      {/* Display existing items */}
      {console.log('🔍 LoadingItemForm items:', items)}
      {items.length > 0 ? (
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
                  key={`${item.reel_number}-${index}`}
                  className="bg-blue-50 rounded-lg border border-blue-200 p-4"
                >
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold text-blue-800">
                      {item.reel_number || 'بدون شماره رول'}
                    </h3>
                    {canEdit && (
                      <div className="flex gap-2">
                        <DeleteConfirmDialog
                          itemName={item.reel_number || 'بدون شماره رول'}
                          itemType="loaded"
                          onConfirm={() => handleDelete(index)}
                        >
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="text-red-600 hover:bg-red-100"
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </DeleteConfirmDialog>
                      </div>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {/* Reel Number */}
                    <div className="space-y-2">
                      <Label className="text-blue-700 font-medium text-sm">شماره رول</Label>
                      <Input
                        type="text"
                        value={item.reel_number || ''}
                        onChange={(e) => handleItemFieldChange(index, 'reel_number', e.target.value)}
                        placeholder="مثال: 1482"
                        className="h-9 border-blue-300 focus:border-blue-500"
                        disabled={!canEdit}
                      />
                    </div>

                    {/* Width */}
                    <div className="space-y-2">
                      <Label className="text-blue-700 font-medium text-sm">عرض (mm)</Label>
                      <Input
                        type="number"
                        min="0"
                        value={item.width || ''}
                        onChange={(e) => handleItemFieldChange(index, 'width', parseInt(e.target.value) || '')}
                        placeholder="مثال: 240"
                        className="h-9 border-blue-300 focus:border-blue-500"
                        disabled={!canEdit}
                      />
                    </div>

                    {/* GSM */}
                    <div className="space-y-2">
                      <Label className="text-blue-700 font-medium text-sm">گرماژ (gsm)</Label>
                      <Input
                        type="number"
                        min="0"
                        value={item.gsm || ''}
                        onChange={(e) => handleItemFieldChange(index, 'gsm', parseInt(e.target.value) || '')}
                        placeholder="مثال: 130"
                        className="h-9 border-blue-300 focus:border-blue-500"
                        disabled={!canEdit}
                      />
                    </div>

                    {/* Length */}
                    <div className="space-y-2">
                      <Label className="text-blue-700 font-medium text-sm">طول (m)</Label>
                      <Input
                        type="number"
                        min="0"
                        value={item.length || ''}
                        onChange={(e) => handleItemFieldChange(index, 'length', parseInt(e.target.value) || '')}
                        placeholder="مثال: 6300"
                        className="h-9 border-blue-300 focus:border-blue-500"
                        disabled={!canEdit}
                      />
                    </div>

                    {/* Breaks */}
                    <div className="space-y-2">
                      <Label className="text-blue-700 font-medium text-sm">تعداد توقف</Label>
                      <Input
                        type="number"
                        min="0"
                        value={item.breaks || ''}
                        onChange={(e) => handleItemFieldChange(index, 'breaks', parseInt(e.target.value) || '')}
                        placeholder="مثال: 2"
                        className="h-9 border-blue-300 focus:border-blue-500"
                        disabled={!canEdit}
                      />
                    </div>

                    {/* Grade */}
                    <div className="space-y-2">
                      <Label className="text-blue-700 font-medium text-sm">درجه/کیفیت</Label>
                      <Input
                        type="text"
                        value={item.grade || ''}
                        onChange={(e) => handleItemFieldChange(index, 'grade', e.target.value)}
                        placeholder="مثال: Testliner HOMAYOUN"
                        className="h-9 border-blue-300 focus:border-blue-500"
                        disabled={!canEdit}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card className="bg-white border-2 border-blue-200">
          <CardContent className="p-6">
            <div className="text-center space-y-4">
              <div className="text-blue-600">
                <Package className="w-16 h-16 mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">آیتم‌های بارگیری</h3>
                <p className="text-sm text-gray-600">
                  در حال حاضر هیچ آیتمی وجود ندارد.
                </p>
                <p className="text-sm text-gray-600 mt-2">
                  می‌توانید آیتم‌ها را به صورت دستی اضافه کنید.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Add New Item Button */}
      {canEdit && (
        <div className="flex justify-center">
          <Button
            onClick={handleAddEmptyItem}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 flex items-center gap-2"
          >
            <Plus className="w-5 h-4" />
            افزودن آیتم جدید
          </Button>
        </div>
      )}
    </div>
  );
}
