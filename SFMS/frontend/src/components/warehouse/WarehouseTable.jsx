import React from 'react';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Edit, X } from 'lucide-react';
import Spinner from '@/components/Spinner';

const WarehouseTable = ({ 
  warehouses, 
  loading, 
  searchTerm, 
  onEditClick, 
  onDeleteClick 
}) => {
  if (loading) {
    return (
      <TableRow>
        <TableCell colSpan={6} className="text-center py-8">
          <div className="flex items-center justify-center gap-2">
            <Spinner className="w-4 h-4 text-blue-600" />
            در حال بارگذاری...
          </div>
        </TableCell>
      </TableRow>
    );
  }

  if (warehouses.length === 0) {
    return (
      <TableRow>
        <TableCell colSpan={6} className="text-center py-8 text-gray-500">
          {searchTerm ? 'هیچ انباری با این جستجو یافت نشد.' : 'هیچ انباری یافت نشد.'}
        </TableCell>
      </TableRow>
    );
  }

  return warehouses.map((warehouse) => (
    <TableRow key={warehouse.id} className="group hover:bg-gray-50 transition-colors duration-200">
      <TableCell>
        <div className="font-medium">
          {warehouse.persian_name || warehouse.name}
        </div>
      </TableCell>
      <TableCell className="text-sm text-gray-500">
        {warehouse.name}
      </TableCell>
      <TableCell className="text-sm text-gray-500">
        {warehouse.id}
      </TableCell>

      <TableCell>
        <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs ${
          warehouse.is_active 
            ? 'bg-green-100 text-green-800' 
            : 'bg-red-100 text-red-800'
        }`}>
          {warehouse.is_active ? 'فعال' : 'غیرفعال'}
        </div>
      </TableCell>
      <TableCell>
        <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => onEditClick(warehouse)}
            className="h-8 w-8 p-0 border-gray-300 hover:border-blue-500 hover:bg-blue-50 transition-all duration-200"
          >
            <Edit className="h-4 w-4 text-gray-600 hover:text-blue-600" />
          </Button>
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => onDeleteClick(warehouse)}
            className="h-8 w-8 p-0 border-gray-300 hover:border-red-500 hover:bg-red-50 transition-all duration-200"
          >
            <X className="h-4 w-4 text-gray-600 hover:text-red-600" />
          </Button>
        </div>
      </TableCell>
    </TableRow>
  ));
};

export default WarehouseTable; 