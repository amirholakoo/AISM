import React from 'react';
import { Input } from '@/components/ui/input';
import { Search } from 'lucide-react';

const ProductSearch = ({ searchTerm, onSearchChange }) => {
  return (
    <div className="relative">
      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
      <Input
        placeholder="جستجو در محصولات..."
        value={searchTerm}
        onChange={(e) => onSearchChange(e.target.value)}
        className="pl-10 bg-white border-slate-300 focus:border-blue-500 focus:ring-blue-500/20"
      />
    </div>
  );
};

export default ProductSearch; 