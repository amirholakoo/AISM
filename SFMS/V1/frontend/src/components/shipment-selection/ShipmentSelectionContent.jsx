import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import Spinner from "@/components/Spinner";
import AlertManager from "@/components/AlertManager";
import ShipmentCardGrid from "./ShipmentCardGrid";

const ShipmentSelectionContent = ({ 
  pageTransitionLoading,
  message,
  error,
  setMessage,
  setError,
  shipments,
  loading,
  selectedShipmentId,
  onShipmentSelect,
  onWarehouseSelect,
  operationType = "unloading"
}) => {
  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <AlertManager
        showEditExpiredAlert={false}
        setShowEditExpiredAlert={() => {}}
        editingLoading={null}
        canEdit={false}
        remainingMinutes={0}
        showRemainingTimeAlert={false}
        setShowRemainingTimeAlert={() => {}}
        connectedToExisting={false}
        setConnectedToExisting={() => {}}
        started={false}
        message={message}
        error={error}
        setMessage={setMessage}
        setError={setError}
      />

      {(loading || pageTransitionLoading) ? (
        <div className="space-y-6">
          {/* Skeleton Loading Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-5xl mx-auto">
                         {Array.from({ length: 6 }).map((_, index) => (
               <Card key={index} className="bg-white shadow-md animate-pulse border py-3">
                 <CardContent className="p-3 py-0">
                  <div className="space-y-2">
                                         {/* Header with icon and license skeleton */}
                     <div className="flex items-center justify-between">
                       <div className="p-2 bg-gray-200 rounded-lg w-9 h-9"></div>
                       <div className="text-left">
                         <div className="w-20 h-5 bg-gray-200 rounded mb-1"></div>
                         <div className="flex items-center gap-1">
                           <div className="w-3 h-4 bg-gray-200 rounded"></div>
                           <div className="w-12 h-4 bg-gray-200 rounded"></div>
                           <div className="w-3 h-4 bg-gray-200 rounded"></div>
                           <div className="w-16 h-4 bg-gray-200 rounded"></div>
                         </div>
                       </div>
                     </div>
                    
                    {/* Divider skeleton */}
                    <div className="border-t border-gray-200"></div>
                    
                                         {/* Details skeleton */}
                     <div className="space-y-1">
                       <div className="flex justify-between items-center">
                         <div className="w-16 h-4 bg-gray-200 rounded"></div>
                         <div className="w-20 h-4 bg-gray-200 rounded"></div>
                       </div>
                       <div className="flex justify-between items-center">
                         <div className="w-12 h-4 bg-gray-200 rounded"></div>
                         <div className="w-16 h-4 bg-gray-200 rounded"></div>
                       </div>
                       <div className="flex justify-between items-center">
                         <div className="w-14 h-4 bg-gray-200 rounded"></div>
                         <div className="w-18 h-4 bg-gray-200 rounded"></div>
                       </div>
                       <div className="flex justify-between items-center">
                         <div className="w-8 h-4 bg-gray-200 rounded"></div>
                         <div className="w-12 h-4 bg-gray-200 rounded"></div>
                       </div>
                     </div>
                    
                    {/* Button skeleton */}
                    <div className="mt-4 text-center w-full">
                      <div className="w-full h-7 bg-gray-200 rounded-sm"></div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      ) : (
        <div className="space-y-6">
                  <ShipmentCardGrid
          shipments={shipments}
          loading={loading}
          selectedShipmentId={selectedShipmentId}
          onShipmentSelect={onShipmentSelect}
          onWarehouseSelect={onWarehouseSelect}
          operationType={operationType}
        />
        </div>
      )}
    </main>
  );
};

export default ShipmentSelectionContent; 