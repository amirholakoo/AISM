import React, { createContext, useContext, useState, useEffect } from 'react';

const UnloadingContext = createContext();

export const useUnloadingContext = () => {
  const context = useContext(UnloadingContext);
  if (!context) {
    throw new Error('useUnloadingContext must be used within an UnloadingProvider');
  }
  return context;
};

export const UnloadingProvider = ({ children }) => {
  // Initialize state from localStorage if available
  const [selectedShipment, setSelectedShipment] = useState(() => {
    const saved = localStorage.getItem('selectedShipment');
    return saved ? JSON.parse(saved) : null;
  });
  
  const [selectedWarehouse, setSelectedWarehouse] = useState(() => {
    const saved = localStorage.getItem('selectedWarehouse');
    return saved ? JSON.parse(saved) : null;
  });
  
  const [operationType, setOperationType] = useState(() => {
    const saved = localStorage.getItem('operationType');
    return saved || 'unloading';
  });
  
  const [selectedCameraId, setSelectedCameraId] = useState(() => {
    const saved = localStorage.getItem('selectedCameraId');
    return saved || null;
  });

  // Save to localStorage whenever state changes
  useEffect(() => {
    if (selectedShipment) {
      localStorage.setItem('selectedShipment', JSON.stringify(selectedShipment));
    } else {
      localStorage.removeItem('selectedShipment');
    }
  }, [selectedShipment]);

  useEffect(() => {
    if (selectedWarehouse) {
      localStorage.setItem('selectedWarehouse', JSON.stringify(selectedWarehouse));
    } else {
      localStorage.removeItem('selectedWarehouse');
    }
  }, [selectedWarehouse]);

  useEffect(() => {
    localStorage.setItem('operationType', operationType);
  }, [operationType]);

  useEffect(() => {
    if (selectedCameraId) {
      localStorage.setItem('selectedCameraId', selectedCameraId);
    } else {
      localStorage.removeItem('selectedCameraId');
    }
  }, [selectedCameraId]);

  const value = {
    selectedShipment,
    setSelectedShipment,
    selectedWarehouse,
    setSelectedWarehouse,
    operationType,
    setOperationType,
    selectedCameraId,
    setSelectedCameraId,
    // Helper function to clear all data
    clearData: () => {
      setSelectedShipment(null);
      setSelectedWarehouse(null);
      setOperationType('unloading');
      setSelectedCameraId(null);
      localStorage.removeItem('selectedShipment');
      localStorage.removeItem('selectedWarehouse');
      localStorage.removeItem('selectedCameraId');
      localStorage.setItem('operationType', 'unloading');
    }
  };

  return (
    <UnloadingContext.Provider value={value}>
      {children}
    </UnloadingContext.Provider>
  );
};