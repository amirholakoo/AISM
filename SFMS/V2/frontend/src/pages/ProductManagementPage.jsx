import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { API_ENDPOINTS } from "@/config";
import { toast } from "sonner";
import {
  ProductHeader,
  ProductSearchCard,
  ProductCard,
  ProductAddDialog,
  ProductEditDialog,
  ProductDeleteDialog
} from "@/components/products";

export default function ProductManagementPage() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [formData, setFormData] = useState({
    name: "",
    persian_name: "",
    vision_name: "",
    width: "",
    gsm: "",
    length: ""
  });

  // بارگذاری محصولات
  useEffect(() => {
    loadProducts();
  }, []);

  const loadProducts = async () => {
    try {
      setLoading(true);
      const response = await fetch(API_ENDPOINTS.PRODUCTS);
      const data = await response.json();
      
      if (data.success) {
        setProducts(data.data || []);
      } else {
        toast.error("خطا در بارگذاری محصولات");
      }
    } catch (error) {
      console.error("خطا در بارگذاری محصولات:", error);
      toast.error("خطا در بارگذاری محصولات");
    } finally {
      setLoading(false);
    }
  };

  // فیلتر کردن محصولات بر اساس جستجو
  const filteredProducts = products.filter(product =>
    product.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    product.persian_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    product.vision_name?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // باز کردن دیالوگ افزودن
  const openAddDialog = () => {
    setFormData({
      name: "",
      persian_name: "",
      vision_name: "",
      width: "",
      gsm: "",
      length: ""
    });
    setIsAddDialogOpen(true);
  };

  // باز کردن دیالوگ ویرایش
  const openEditDialog = (product) => {
    setSelectedProduct(product);
    setFormData({
      name: product.name || "",
      persian_name: product.persian_name || "",
      vision_name: product.vision_name || "",
      width: product.width?.toString() || "",
      gsm: product.gsm?.toString() || "",
      length: product.length?.toString() || ""
    });
    setIsEditDialogOpen(true);
  };

  // باز کردن دیالوگ حذف
  const openDeleteDialog = (product) => {
    setSelectedProduct(product);
    setIsDeleteDialogOpen(true);
  };

  // تغییر فرم
  const handleFormChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  // ذخیره محصول (افزودن یا ویرایش)
  const handleSaveProduct = async () => {
    try {
      const productData = {
        ...formData,
        width: formData.width ? parseInt(formData.width) : null,
        gsm: formData.gsm ? parseInt(formData.gsm) : null,
        length: formData.length ? parseInt(formData.length) : null
      };

      const url = selectedProduct 
        ? API_ENDPOINTS.PRODUCT_UPDATE(selectedProduct.id)
        : API_ENDPOINTS.PRODUCT_CREATE;
      
      const method = selectedProduct ? "PUT" : "POST";

      const response = await fetch(url, {
        method,
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(productData)
      });

      const data = await response.json();

      if (data.success) {
        toast.success(selectedProduct ? "محصول با موفقیت ویرایش شد" : "محصول با موفقیت افزوده شد");
        loadProducts();
        setIsAddDialogOpen(false);
        setIsEditDialogOpen(false);
        setSelectedProduct(null);
      } else {
        toast.error(data.error || "خطا در ذخیره محصول");
      }
    } catch (error) {
      console.error("خطا در ذخیره محصول:", error);
      toast.error("خطا در ذخیره محصول");
    }
  };

  // حذف محصول
  const handleDeleteProduct = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.PRODUCT_DELETE(selectedProduct.id), {
        method: "DELETE"
      });

      const data = await response.json();

      if (data.success) {
        toast.success("محصول با موفقیت حذف شد");
        loadProducts();
        setIsDeleteDialogOpen(false);
        setSelectedProduct(null);
      } else {
        toast.error(data.error || "خطا در حذف محصول");
      }
    } catch (error) {
      console.error("خطا در حذف محصول:", error);
      toast.error("خطا در حذف محصول");
    }
  };

  return (
    <>
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <ProductHeader 
            onAddClick={openAddDialog}
            productsCount={products.length}
          />
        </div>
      </div>

      {/* Main content */}
      <div className="min-h-screen bg-slate-50">
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Search Card */}
        <ProductSearchCard 
          searchTerm={searchTerm}
          onSearchChange={setSearchTerm}
        />

        {/* Products List */}
        <Card>
          <CardContent>
            {/* Product Cards - Both Desktop and Mobile */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <ProductCard 
                products={filteredProducts}
                loading={loading}
                searchTerm={searchTerm}
                onEditClick={openEditDialog}
                onDeleteClick={openDeleteDialog}
              />
            </div>
          </CardContent>
        </Card>
      </main>

      {/* Add Product Dialog */}
      <ProductAddDialog 
        open={isAddDialogOpen}
        onOpenChange={setIsAddDialogOpen}
        formData={formData}
        onFormChange={handleFormChange}
        onSave={handleSaveProduct}
        onCancel={() => setIsAddDialogOpen(false)}
      />

      {/* Edit Product Dialog */}
      <ProductEditDialog 
        open={isEditDialogOpen}
        onOpenChange={setIsEditDialogOpen}
        formData={formData}
        onFormChange={handleFormChange}
        onSave={handleSaveProduct}
        onCancel={() => setIsEditDialogOpen(false)}
      />

      {/* Delete Product Dialog */}
      <ProductDeleteDialog 
        open={isDeleteDialogOpen}
        onOpenChange={setIsDeleteDialogOpen}
        selectedProduct={selectedProduct}
        onDelete={handleDeleteProduct}
        onCancel={() => setIsDeleteDialogOpen(false)}
      />
        </div>
      </>
    );
  } 