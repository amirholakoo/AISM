import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { PlusCircleIcon, EditIcon, CheckIcon, PackageXIcon, Package } from "lucide-react";
import DeleteConfirmDialog from "@/components/DeleteConfirmDialog";
import ProductAdder from "@/components/ProductAdder";
import Spinner from "@/components/Spinner";

const ItemsTable = ({
  items,
  products,
  editingLoading,
  loading,
  canEdit,
  onItemChange,
  onDeleteItem,
  onRestoreItem,
  onAddItem,
  onEdit,
  onSave,
  showEditExpiredAlert = false,
  getPersianName
}) => {
  // اگر در حالت ویرایش مجدد هستیم و آیتمی نیست، فرم خالی نمایش دهیم
  if (editingLoading && items.length === 0) {
    return (
      <form
        onSubmit={e => {
          e.preventDefault();
          onEdit();
        }}
      >
        <div className="space-y-6 mb-4">
          {/* بارگیری‌ها */}
          <Card className="bg-blue-50/50 border-2 border-blue-700">
            <CardHeader className="pb-3">
              <CardTitle className="text-center text-blue-700">
                بارگیری‌ها (0)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-2">
                <div className="flex flex-col items-center gap-1">
                  <PackageXIcon className="w-6 h-6 text-gray-400" />
                  <p className="text-gray-500 mb-1">هیچ آیتمی برای ویرایش وجود ندارد</p>
                  <p className="text-sm text-gray-400">می‌توانید آیتم‌های جدید اضافه کنید</p>
                </div>
              </div>
              <ProductAdder
                type="loaded"
                products={products}
                items={items}
                onAddItem={onAddItem}
              />
            </CardContent>
          </Card>

          {/* تخلیه‌ها */}
          <Card className="bg-red-50/50 border-2 border-red-700">
            <CardHeader className="pb-3">
              <CardTitle className="text-center text-red-700">
                تخلیه‌ها (0)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-2">
                <div className="flex flex-col items-center gap-1">
                  <PackageXIcon className="w-6 h-6 text-gray-400" />
                  <p className="text-gray-500 mb-1">هیچ آیتمی برای ویرایش وجود ندارد</p>
                  <p className="text-sm text-gray-400">می‌توانید آیتم‌های جدید اضافه کنید</p>
                </div>
              </div>
              <ProductAdder
                type="unloaded"
                products={products}
                items={items}
                onAddItem={onAddItem}
              />
            </CardContent>
          </Card>
        </div>
        
        <hr className="my-4 border-t border-gray-300" />
        
        <Button 
          type="submit" 
          className="w-full bg-orange-800 hover:bg-orange-900 text-white"
          disabled={loading}
        >
          {loading && <Spinner />}
          {!loading && <EditIcon className="w-4 h-4 ml-2" />}
          ویرایش و ذخیره
        </Button>
      </form>
    );
  }

  // اگر آیتمی نیست و در حالت ویرایش مجدد هم نیستیم، هیچ چیزی نمایش ندهیم
  if (items.length === 0) return null;

  // Remove duplicates and create unique items with proper identification
  const uniqueItems = items.reduce((acc, item, index) => {
    const key = `${item.name}-${item.type}`;
    if (!acc.find(existing => `${existing.name}-${existing.type}` === key)) {
      acc.push({ ...item, uniqueId: index });
    }
    return acc;
  }, []);

  const loadedItems = uniqueItems.filter(item => item.type === "loaded");
  const unloadedItems = uniqueItems.filter(item => item.type === "unloaded");

  // Helper function to find item index by unique identifier
  const findItemIndex = (itemName, itemType) => {
    return items.findIndex(item => item.name === itemName && item.type === itemType);
  };

  return (
    <form
             onSubmit={e => {
         e.preventDefault();
         if (editingLoading) {
           onEdit();
         } else {
           onSave();
         }
       }}
    >
      <div className="space-y-8 mb-6">
        {/* بارگیری‌ها */}
        <Card className="bg-gradient-to-br from-blue-50 to-blue-100/50 border-2 border-blue-600 shadow-lg">
          <CardHeader className="pb-4">
            <CardTitle className="text-center text-blue-800 text-xl font-bold flex items-center justify-center gap-2">
              <Package className="w-6 h-6" />
              بارگیری‌ها ({loadedItems.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loadedItems.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-6">
                {loadedItems.map((item) => (
                  <div key={`${item.name}-${item.type}-${item.uniqueId}`} className="bg-white rounded-lg border border-blue-200 p-4 shadow-sm hover:shadow-md transition-shadow duration-200">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-semibold text-blue-800 text-sm">
                        {getPersianName ? getPersianName(item.name) : item.name}
                      </h3>
                                             <DeleteConfirmDialog
                         itemName={item.name}
                         itemType="loaded"
                         onConfirm={() => onDeleteItem(item.name, "loaded")}
                       />
                    </div>
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        min={0}
                        value={item.count}
                        onChange={e => {
                          const index = findItemIndex(item.name, "loaded");
                          if (index !== -1) {
                            onItemChange(index, e.target.value);
                          }
                        }}
                                                 className="flex-1 text-center font-bold text-blue-700 border-blue-300 focus:border-blue-500 focus:ring-blue-500"
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center pb-8">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
                    <PackageXIcon className="w-8 h-8 text-blue-400" />
                  </div>
                  <div>
                    <p className="text-gray-600 font-medium mb-1">هیچ بارگیری‌ای ثبت نشده است</p>
                    <p className="text-sm text-gray-500">می‌توانید از لیست زیر آیتم اضافه کنید</p>
                  </div>
                </div>
              </div>
            )}
            
            {/* Available products for loaded */}
            <div className="border-t border-blue-200 pt-4">
              <ProductAdder
                type="loaded"
                products={products}
                items={items}
                onAddItem={onAddItem}
              />
            </div>
          </CardContent>
        </Card>

        {/* تخلیه‌ها */}
        <Card className="bg-gradient-to-br from-red-50 to-red-100/50 border-2 border-red-600 shadow-lg">
          <CardHeader className="pb-4">
            <CardTitle className="text-center text-red-800 text-xl font-bold flex items-center justify-center gap-2">
              <Package className="w-6 h-6" />
              تخلیه‌ها ({unloadedItems.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            {unloadedItems.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-6">
                {unloadedItems.map((item) => (
                  <div key={`${item.name}-${item.type}-${item.uniqueId}`} className="bg-white rounded-lg border border-red-200 p-4 shadow-sm hover:shadow-md transition-shadow duration-200">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-semibold text-red-800 text-sm">
                        {getPersianName ? getPersianName(item.name) : item.name}
                      </h3>
                      <DeleteConfirmDialog
                        itemName={item.name}
                        itemType="unloaded"
                        onConfirm={() => onDeleteItem(item.name, "unloaded")}
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        min={0}
                        value={item.count}
                        onChange={e => {
                          const index = findItemIndex(item.name, "unloaded");
                          if (index !== -1) {
                            onItemChange(index, e.target.value);
                          }
                        }}
                        className="flex-1 text-center font-bold text-red-700 border-red-300 focus:border-red-500 focus:ring-red-500"
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center pb-8">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center">
                    <PackageXIcon className="w-8 h-8 text-red-400" />
                  </div>
                  <div>
                    <p className="text-gray-600 font-medium mb-1">هیچ تخلیه‌ای ثبت نشده است</p>
                    <p className="text-sm text-gray-500">می‌توانید از لیست زیر آیتم اضافه کنید</p>
                  </div>
                </div>
              </div>
            )}
            
            {/* Available products for unloaded */}
            <div className="border-t border-red-200 pt-4">
              <ProductAdder
                type="unloaded"
                products={products}
                items={items}
                onAddItem={onAddItem}
              />
            </div>
          </CardContent>
        </Card>
      </div>
      
      <hr className="my-6 border-t-2 border-gray-300" />
      
      <Button 
        type="submit" 
        className={`w-full py-4 text-lg font-semibold ${
          editingLoading 
            ? 'bg-gradient-to-r from-orange-600 to-orange-700 hover:from-orange-700 hover:to-orange-800 text-white shadow-lg' 
            : 'bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white shadow-lg'
        }`}
        disabled={loading}
        onClick={editingLoading ? onEdit : undefined}
        title={editingLoading ? 'ویرایش و ذخیره' : 'تایید و ذخیره'}
      >
        {loading && <Spinner />}
        {!loading && (
          editingLoading ? (
            <EditIcon className="w-5 h-5 ml-2" />
          ) : (
            <CheckIcon className="w-5 h-5 ml-2" />
          )
        )}
        {editingLoading ? 'ویرایش و ذخیره' : 'تایید و ذخیره'}
      </Button>
    </form>
  );
};

export default ItemsTable;