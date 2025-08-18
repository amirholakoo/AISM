import React from 'react';
import { Card } from '@/components/ui/card';
import WarehouseSearch from './WarehouseSearch';

const WarehouseSearchCard = ({ searchTerm, onSearchChange }) => {
  return (
    <Card className="mb-6 p-6">
      <WarehouseSearch 
        searchTerm={searchTerm}
        onSearchChange={onSearchChange}
      />
    </Card>
  );
};

export default WarehouseSearchCard; 