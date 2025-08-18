import React from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectItem, SelectTrigger, SelectContent, SelectValue } from "@/components/ui/select";
import { PlayIcon, SquareIcon } from "lucide-react";
import Spinner from "@/components/Spinner";
import EditLastLoadingButton from "./EditLastLoadingButton";

const ControlPanel = ({
  warehouses,
  selectedWarehouseId,
  setSelectedWarehouseId,
  started = false,
  loading = false,
  editingLoading = false,
  items = [],
  onStart,
  onEnd,
  onLoadLastCompleted,
  // New props for QR scanner
  warehouseId,
  operationType,
  showConfirmation,
  onCancelConfirmation,
  onConfirmEnd
}) => {
  return (
    <>
      {/* دکمه شروع بارگیری - فقط در حالت اولیه */}
      {!started && !editingLoading && (!items || items.length === 0) && (
        <Button
          className={`w-full mb-2 ${loading ? 'bg-gray-50 hover:bg-gray-100 border-gray-300 text-gray-600' : ''}`}
          onClick={onStart}
          disabled={(!selectedWarehouseId && !warehouseId) || loading}
        >
          {loading && <Spinner />}
          {!loading && <PlayIcon className="w-4 h-4 ml-2" />}
          شروع بارگیری
        </Button>
      )}
        
      {started && (!items || items.length === 0) && (
        <Button
          className={`w-full mb-2 ${loading ? 'bg-gray-50 hover:bg-gray-100 border-gray-300 text-gray-600' : 'bg-blue-800 hover:bg-blue-900 text-white'}`}
          onClick={onEnd}
          disabled={loading}
          variant="default"
        >
          {loading && <Spinner />}
          {!loading && <SquareIcon className="w-4 h-4 ml-2" />}
          پایان بارگیری
        </Button>
      )}

      {/* دکمه ویرایش مجدد */}
      {!started && (!items || items.length === 0) && !editingLoading && onLoadLastCompleted && (
        <div className="mb-4">
          <EditLastLoadingButton
            onClick={onLoadLastCompleted}
          />
        </div>
      )}
    </>
  );
};

export default ControlPanel;