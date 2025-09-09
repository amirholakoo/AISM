import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Search, Warehouse } from 'lucide-react';
import { API_ENDPOINTS } from '@/config';

const VisionServerSearchCard = ({ searchTerm, onSearchChange, selectedWarehouse, onWarehouseChange }) => {
  const [warehouses, setWarehouses] = useState([]);
  const [loading, setLoading] = useState(false);

  // بارگذاری لیست انبارها
  useEffect(() => {
    const loadWarehouses = async () => {
      try {
        setLoading(true);
        const response = await fetch(API_ENDPOINTS.WAREHOUSES);
        const data = await response.json();
        
        if (data.warehouses) {
          setWarehouses(data.warehouses);
        }
      } catch (error) {
        console.error('خطا در بارگذاری انبارها:', error);
      } finally {
        setLoading(false);
      }
    };

    loadWarehouses();
  }, []);

  return (
    <Card className="bg-white shadow-sm border border-slate-200">
      <CardContent className="px-4 py-0">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* انتخاب انبار */}
          <div className="relative">
            <Warehouse className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
            <Select value={selectedWarehouse} onValueChange={onWarehouseChange}>
              <SelectTrigger className="pr-10 w-full">
                <SelectValue placeholder="انتخاب انبار..." />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">همه انبارها</SelectItem>
                <SelectItem value="mobile">دوربین‌های متحرک</SelectItem>
                {warehouses.map((warehouse) => (
                  <SelectItem key={warehouse.id} value={warehouse.id.toString()}>
                    {warehouse.persian_name || warehouse.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* جستجو */}
          <div className="relative">
            <Search className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
            <Input
              type="text"
              placeholder="جستجو در سرورهای بینایی..."
              value={searchTerm}
              onChange={(e) => onSearchChange(e.target.value)}
              className="pr-10"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default VisionServerSearchCard; 