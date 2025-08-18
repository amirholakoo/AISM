import React from "react";
import { Button } from "@/components/ui/button";
import { PlusCircleIcon } from "lucide-react";

const ProductAdder = ({ 
  type, 
  products, 
  items, 
  onAddItem,
  disabled = false
}) => {
  // فیلتر کردن محصولاتی که هنوز در این نوع اضافه نشده‌اند
  const availableProducts = products.filter(product => 
    !items.some(item => item.name === product.name && item.type === type)
  );

  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {availableProducts.map(product => (
        <Button
          key={product.id}
          type="button"
          variant="secondary"
          className="text-xs flex justify-center items-center gap-2"
          onClick={() => onAddItem(type, product.name)}
          disabled={disabled}
        >
          <PlusCircleIcon className="w-4 h-4" />
          {product.persian_name || product.name}
        </Button>
      ))}
    </div>
  );
};

export default ProductAdder;