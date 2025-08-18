import React from "react";
import { Package } from "lucide-react";
import ShipmentCard from "./ShipmentCard";

const ShipmentCardGrid = ({ 
  shipments, 
  loading, 
  selectedShipmentId, 
  onShipmentSelect,
  onWarehouseSelect,
  operationType = "unloading"
}) => {
  // اگر loading است، چیزی نمایش نده (loading در ShipmentSelectionContent نمایش داده می‌شود)
  if (loading) {
    return null;
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-5xl mx-auto">
      {shipments.map(shipment => (
        <div key={shipment.id} className="relative">
          <ShipmentCard
            shipment={shipment}
            isSelected={selectedShipmentId === shipment.id.toString()}
            onClick={() => onShipmentSelect(shipment.id.toString())}
            onWarehouseSelect={onWarehouseSelect}
            variant="compact"
            operationType={operationType}
          />
        </div>
      ))}
      
      {shipments.length === 0 && (
        <div className="col-span-full text-center text-gray-600 py-8">
          <Package className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <p>هیچ محموله‌ای یافت نشد.</p>
        </div>
      )}
    </div>
  );
};

export default ShipmentCardGrid; 