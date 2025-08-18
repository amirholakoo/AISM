import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import WarehouseButton from "./WarehouseButton";

// Skeleton component for warehouse loading
const WarehouseSkeleton = () => (
  <div className="w-full h-20 bg-gray-200 rounded-lg animate-pulse"></div>
);


const WarehouseButtons = ({
  warehouses,
  selectedWarehouseId,
  setSelectedWarehouseId,
  started,
  loading,
  onStart,
  onEnd,
  onLoadLastCompleted
}) => {

  // اگر در حالت بارگیری یا ویرایش هستیم، دکمه‌های انبار رو نمایش ندهیم
  if (started) {
    return null;
  }

  return (
    <div className="space-y-6">
      <Card className="bg-white shadow-md">
        <CardHeader>
          <CardTitle className="text-xl font-semibold text-slate-900">
            انبارهای موجود
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Warehouse Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
            {loading ? (
              // Skeleton loading
              Array.from({ length: 8 }).map((_, index) => (
                <WarehouseSkeleton key={index} />
              ))
            ) : (
              warehouses.map(warehouse => (
                <WarehouseButton
                  key={warehouse.id}
                  warehouse={warehouse}
                  selectedWarehouseId={selectedWarehouseId}
                  setSelectedWarehouseId={setSelectedWarehouseId}
                  onStart={onStart}
                  onEnd={onEnd}
                />
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default WarehouseButtons; 