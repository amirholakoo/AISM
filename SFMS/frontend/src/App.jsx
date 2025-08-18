import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Toaster } from "@/components/ui/sonner";
import { UnloadingProvider } from "./contexts/LoadingContext";
import HomePage from "./pages/HomePage";
import WarehouseSelectionPage from "./pages/WarehouseSelectionPage";
import UnloadingPage from "./pages/UnloadingPage";
import EditPage from "./pages/EditPage";
import UnloadingEditPage from "./pages/UnloadingEditPage";
import LoadingEditPage from "./pages/LoadingEditPage";

import ShipmentSelectionInitialPage from "./pages/ShipmentSelectionInitialPage";
import ShipmentSelectionUnloadingPage from "./pages/ShipmentSelectionUnloadingPage";
import ShipmentSelectionLoadingPage from "./pages/ShipmentSelectionLoadingPage";
import LoadingPage from "./pages/LoadingPage";
import CameraSelectionPage from "./pages/CameraSelectionPage";
import WarehouseManagementPage from "./pages/WarehouseManagementPage";
import ProductManagementPage from "./pages/ProductManagementPage";
import OperationsListPage from "./pages/OperationsListPage";
import AdminPanelPage from "./pages/AdminPanelPage";
import OperationTypesPage from "./pages/admin/OperationTypesPage";
import VisionServersPage from "./pages/admin/VisionServersPage";
import ServerAssignmentPage from "./pages/admin/ServerAssignmentPage";
import Footer from "./components/ui/footer";

export default function App() {




  return (
    <UnloadingProvider>
      <Router>
        <div className="min-h-screen flex flex-col">
          <div className="flex-1">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/warehouse-select" element={<WarehouseSelectionPage />} />
              <Route path="/camera-select/:warehouseId" element={<CameraSelectionPage />} />
              <Route path="/shipment-select" element={<ShipmentSelectionInitialPage />} />
              <Route path="/shipment-select-unloading" element={<ShipmentSelectionUnloadingPage />} />
              <Route path="/shipment-select-loading" element={<ShipmentSelectionLoadingPage />} />

              <Route path="/unloading/:warehouseId" element={<UnloadingPage />} />
              <Route path="/loading/:warehouseId" element={<LoadingPage />} />
              <Route path="/edit/:unloadingToken" element={<EditPage />} />
              <Route path="/unloading-edit/:unloadingToken" element={<UnloadingEditPage />} />
              <Route path="/loading-edit/:loadingToken" element={<LoadingEditPage />} />
              <Route path="/warehouses" element={<WarehouseManagementPage />} />
              <Route path="/products" element={<ProductManagementPage />} />
              <Route path="/operations" element={<OperationsListPage />} />
              
              {/* Admin Routes */}
              <Route path="/admin" element={<AdminPanelPage />} />
              <Route path="/admin/operation-types" element={<OperationTypesPage />} />
              <Route path="/admin/vision-servers" element={<VisionServersPage />} />
              <Route path="/admin/warehouse-assignments" element={<ServerAssignmentPage />} />
            </Routes>
          </div>
          <Footer />
        </div>
        <Toaster 
          position="top-center"
          duration={8000}
          richColors
          closeButton
          expand
          toastOptions={{
            style: {
              fontFamily: "'Shabnam', sans-serif",
              direction: "rtl",
              textAlign: "right"
            }
          }}
        />
      </Router>
    </UnloadingProvider>
  );
}