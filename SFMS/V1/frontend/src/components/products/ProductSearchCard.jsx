import React from 'react';
import { Card } from '@/components/ui/card';
import ProductSearch from './ProductSearch';

const ProductSearchCard = ({ searchTerm, onSearchChange }) => {
  return (
    <Card className="mb-6 p-6">
      <ProductSearch 
        searchTerm={searchTerm}
        onSearchChange={onSearchChange}
      />
    </Card>
  );
};

export default ProductSearchCard; 