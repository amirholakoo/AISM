import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  EyeIcon, 
  ClockIcon, 
  PackageIcon,
  UserIcon,
  BotIcon,
  HistoryIcon,
  CalendarIcon,
  WarehouseIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  EditIcon,
  CheckCircleIcon,
  AlertCircleIcon
} from "lucide-react";
import { extractIdFromToken } from "./UnloadingUtils";

// Map icon names to components
const iconMap = {
  BotIcon,
  UserIcon,
  HistoryIcon,
  PackageIcon,
  CheckCircleIcon,
  AlertCircleIcon,
  EditIcon
};

const UnloadingCard = ({ 
  unloading, 
  onShowDetails, 
  getWarehouseName, 
  getStatusColor, 
  getStatusText, 
  getStatusIcon, 
  getTypeIcon, 
  getVersionText, 
  formatDate, 
  formatTime,
  getOperationTypeText
}) => {
  // Helper function to render icons
  const renderIcon = (iconName) => {
    const IconComponent = iconMap[iconName];
    return IconComponent ? <IconComponent className="w-4 h-4" /> : <PackageIcon className="w-4 h-4" />;
  };

  return (
    <Card className="bg-white hover:shadow-md transition-shadow border border-slate-200">
      <CardContent className="px-3">
        {/* هدر کارت */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {renderIcon(getTypeIcon(unloading.type))}
            <h3 className="font-semibold text-slate-900 text-lg">
              {getOperationTypeText(unloading.type)} {extractIdFromToken(unloading.token)}
            </h3>
          </div>
          <Badge className={getStatusColor(unloading.status)}>
            {getStatusText(unloading.status)}
          </Badge>
        </div>

        {/* اطلاعات انبار */}
        <div className="flex items-center gap-2 mb-2 text-sm text-slate-600">
          <WarehouseIcon className="w-4 h-4" />
          <span>{getWarehouseName(unloading.warehouse_id)}</span>
        </div>

        {/* آمار آیتم‌ها */}
        <div className="mb-2">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <PackageIcon className="w-4 h-4 text-slate-500" />
              <span className="font-medium">{unloading.items_count || 0}</span>
              <span className="text-slate-600">کل آیتم</span>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                <span className="text-green-600 font-medium">{unloading.loaded_count || 0}</span>
                <span className="text-slate-600 text-xs">بارگیری</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 bg-red-500 rounded-full"></span>
                <span className="text-red-600 font-medium">{unloading.unloaded_count || 0}</span>
                <span className="text-slate-600 text-xs">تخلیه</span>
              </div>
            </div>
          </div>
        </div>

        {/* اطلاعات زمانی و نسخه */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <CalendarIcon className="w-3 h-3" />
            <span>{formatDate(unloading.created_at)}</span>
            <ClockIcon className="w-3 h-3" />
            <span>{formatTime(unloading.created_at)}</span>
          </div>
          {unloading.version && (
            <Badge variant="outline" className="text-xs">
              نسخه {getVersionText(unloading.version)}
            </Badge>
          )}
        </div>

        {/* دکمه عملیات */}
        <Button
          onClick={() => onShowDetails(unloading)}
          variant="outline"
          size="sm"
          className="w-full flex items-center justify-center gap-2 bg-white hover:bg-slate-50 border-slate-300 hover:border-slate-400 text-slate-700 hover:text-slate-800 shadow-sm hover:shadow-md transition-all duration-200"
        >
          <EyeIcon className="w-4 h-4" />
          مشاهده جزئیات
        </Button>
      </CardContent>
    </Card>
  );
};

export default UnloadingCard; 