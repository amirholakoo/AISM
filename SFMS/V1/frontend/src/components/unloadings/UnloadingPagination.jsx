import React from "react";
import { Button } from "@/components/ui/button";
import { ChevronLeftIcon, ChevronRightIcon } from "lucide-react";

const UnloadingPagination = ({ 
  pagination, 
  onPageChange, 
  onNextPage, 
  onPrevPage 
}) => {
  // Show pagination even if there's only one page, but with different styling
  if (!pagination) {
    return (
      <div className="mt-8 pt-6 border-t border-slate-200 text-center">
        <p className="text-slate-500">خطا: اطلاعات صفحه‌بندی موجود نیست</p>
      </div>
    );
  }
  
  // Show message when no operations exist
  if (pagination.total === 0) {
    return (
      <div className="mt-8 pt-6 border-t border-slate-200 text-center">
        <p className="text-slate-500">هیچ عملیاتی یافت نشد</p>
        <p className="text-xs text-slate-400 mt-1">صفحه‌بندی: {pagination.page} از {pagination.pages}</p>
      </div>
    );
  }

  // Always show pagination info, even if only one page
  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mt-8 pt-6 border-t border-slate-200">
      <div className="text-sm text-slate-600 text-center sm:text-right">
        نمایش {((pagination.page - 1) * pagination.per_page) + 1} تا {Math.min(pagination.page * pagination.per_page, pagination.total)} از {pagination.total} عملیات
      </div>
      
      {pagination.pages > 1 ? (
        <div className="flex flex-col sm:flex-row items-center gap-3 sm:gap-2">
          <div className="flex items-center gap-2 order-2 sm:order-1">
            <Button
              onClick={onPrevPage}
              disabled={!pagination.has_prev}
              variant="outline"
              size="sm"
              className="flex items-center gap-1 bg-white hover:bg-slate-50 border-slate-300 hover:border-slate-400 text-slate-700 hover:text-slate-800 shadow-sm hover:shadow-md transition-all duration-200 text-xs sm:text-sm"
            >
              <ChevronLeftIcon className="w-3 h-3 sm:w-4 sm:w-4 rotate-180" />
              <span className="hidden sm:inline">قبلی</span>
            </Button>
            
            <div className="flex items-center gap-1">
              {Array.from({ length: Math.min(5, pagination.pages) }, (_, i) => {
                let pageNum;
                if (pagination.pages <= 5) {
                  pageNum = i + 1;
                } else if (pagination.page <= 3) {
                  pageNum = i + 1;
                } else if (pagination.page >= pagination.pages - 2) {
                  pageNum = pagination.pages - 4 + i;
                } else {
                  pageNum = pagination.page - 2 + i;
                }
                
                return (
                  <Button
                    key={pageNum}
                    onClick={() => onPageChange(pageNum)}
                    variant={pagination.page === pageNum ? "default" : "outline"}
                    size="sm"
                    className={`w-7 h-7 sm:w-8 sm:h-8 p-0 text-xs sm:text-sm ${
                      pagination.page === pageNum 
                        ? "bg-blue-600 hover:bg-blue-700 text-white" 
                        : "bg-white hover:bg-slate-50 border-slate-300 hover:border-slate-400 text-slate-700 hover:text-slate-800 shadow-sm hover:shadow-md transition-all duration-200"
                    }`}
                  >
                    {pageNum}
                  </Button>
                );
              })}
            </div>
            
            <Button
              onClick={onNextPage}
              disabled={!pagination.has_next}
              variant="outline"
              size="sm"
              className="flex items-center gap-1 bg-white hover:bg-slate-50 border-slate-300 hover:border-slate-400 text-slate-700 hover:text-slate-800 shadow-sm hover:shadow-md transition-all duration-200 text-xs sm:text-sm"
            >
              <span className="hidden sm:inline">بعدی</span>
              <ChevronRightIcon className="w-3 h-3 sm:w-4 sm:h-4 rotate-180" />
            </Button>
          </div>
        </div>
      ) : (
        <div className="text-sm text-slate-500 text-center">
          صفحه {pagination.page} از {pagination.pages} {pagination.pages === 1 ? '(تک صفحه)' : ''}
        </div>
      )}
    </div>
  );
};

export default UnloadingPagination; 