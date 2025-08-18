import React from "react";
import { PackageIcon } from "lucide-react";
import UnloadingCard from "./UnloadingCard";
import { getOperationTypeText, extractIdFromToken } from "./UnloadingUtils";

const UnloadingList = ({ 
  unloadings, 
  onShowDetails, 
  getWarehouseName, 
  getStatusColor, 
  getStatusText, 
  getStatusIcon, 
  getTypeIcon, 
  getVersionText, 
  formatDate, 
  formatTime,
  selectedOperationType = 'all',
  operationTypes = []
}) => {
  if (unloadings.length === 0) {
    const getEmptyMessage = () => {
      if (selectedOperationType === 'all') {
        return "هیچ عملیاتی یافت نشد.";
      }
      
      const selectedType = operationTypes.find(type => type.id.toString() === selectedOperationType);
      if (selectedType) {
        return `هیچ عملیات ${selectedType.persian_name} یافت نشد.`;
      }
      
      return "هیچ عملیاتی یافت نشد.";
    };

    return (
      <div className="text-center text-slate-600 py-12">
        <PackageIcon className="w-16 h-16 mx-auto mb-4 text-slate-400" />
        <p className="text-lg">{getEmptyMessage()}</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {unloadings.map((unloading) => (
        <UnloadingCard
          key={unloading.token}
          unloading={unloading}
          onShowDetails={onShowDetails}
          getWarehouseName={getWarehouseName}
          getStatusColor={getStatusColor}
          getStatusText={getStatusText}
          getStatusIcon={getStatusIcon}
          getTypeIcon={getTypeIcon}
          getVersionText={getVersionText}
          formatDate={formatDate}
          formatTime={formatTime}
          getOperationTypeText={getOperationTypeText}
        />
      ))}
    </div>
  );
};

export default UnloadingList; 