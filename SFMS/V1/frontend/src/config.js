// API Configuration
// export const API_BASE_URL = 'http://192.168.2.46:18888';
export const API_BASE_URL = 'http://127.0.0.1:18888';

// API Endpoints
export const API_ENDPOINTS = {
  // Health Check
  HEALTH: `${API_BASE_URL}/api/health`,
  
  // Warehouses
  WAREHOUSES: `${API_BASE_URL}/api/warehouses`,
  WAREHOUSES_SYNC: `${API_BASE_URL}/api/warehouses/sync`,
  WAREHOUSE_UPDATE: (id) => `${API_BASE_URL}/api/warehouses/${id}`,
  WAREHOUSE_DELETE: (id) => `${API_BASE_URL}/api/warehouses/${id}`,
  
  // Operation Types
  OPERATION_TYPES: `${API_BASE_URL}/api/operation-types`,
  OPERATION_TYPE_CREATE: `${API_BASE_URL}/api/operation-types`,
  OPERATION_TYPE_UPDATE: (id) => `${API_BASE_URL}/api/operation-types/${id}`,
  OPERATION_TYPE_DELETE: (id) => `${API_BASE_URL}/api/operation-types/${id}`,
  
  // Vision Servers
  VISION_SERVERS: `${API_BASE_URL}/api/vision-servers`,
  VISION_SERVER_CREATE: `${API_BASE_URL}/api/vision-servers`,
  VISION_SERVER_UPDATE: (id) => `${API_BASE_URL}/api/vision-servers/${id}`,
  VISION_SERVER_DELETE: (id) => `${API_BASE_URL}/api/vision-servers/${id}`,
  VISION_SERVERS_ASSIGNMENTS: `${API_BASE_URL}/api/vision-servers/assignments`,
  VISION_SERVERS_GET_ASSIGNMENTS: `${API_BASE_URL}/api/vision-servers/assignments`,
  VISION_SERVERS_BY_WAREHOUSE: (warehouseId) => `${API_BASE_URL}/api/vision-servers/warehouse/${warehouseId}`,
  
  // Unloadings
  UNLOADINGS_LAST_COMPLETED: `${API_BASE_URL}/api/unloadings/last-completed`,
  UNLOADINGS_ACTIVE: `${API_BASE_URL}/api/unloadings/active`,
  UNLOADINGS_ACTIVE_ANY: `${API_BASE_URL}/api/unloadings/active-any`,
  UNLOADINGS_EDIT: `${API_BASE_URL}/api/unloadings/edit`,
  UNLOADINGS_SAVE: `${API_BASE_URL}/api/unloadings/save`,
  UNLOADING_BY_TOKEN: (token) => `${API_BASE_URL}/api/unloadings/${token}`,
  UNLOADING_SHIPMENT_BY_TOKEN: (token) => `${API_BASE_URL}/api/unloadings/${token}/shipment`,
  UNLOADINGS_ALL: `${API_BASE_URL}/api/unloadings/all`,
  UNLOADING_ITEMS: (unloadingId) => `${API_BASE_URL}/api/unloadings/${unloadingId}/items`,
  
  // Operations (combined loadings and unloadings)
  OPERATIONS_ALL: `${API_BASE_URL}/api/operations/all`,
  OPERATIONS_LAST_COMPLETED: `${API_BASE_URL}/api/operations/last-completed`,
  
  // Loadings
  LOADINGS_LAST_COMPLETED: `${API_BASE_URL}/api/loadings/last-completed`,
  LOADINGS_ACTIVE: `${API_BASE_URL}/api/loadings/active`,
  LOADINGS_ACTIVE_ANY: `${API_BASE_URL}/api/loadings/active-any`,
  LOADINGS_EDIT: `${API_BASE_URL}/api/loadings/edit`,
  LOADINGS_SAVE: `${API_BASE_URL}/api/loadings/save`,
  LOADING_BY_TOKEN: (token) => `${API_BASE_URL}/api/loadings/${token}`,
  LOADING_SHIPMENT_BY_TOKEN: (token) => `${API_BASE_URL}/api/loadings/${token}/shipment`,
  LOADINGS_ALL: `${API_BASE_URL}/api/loadings/all`,
  LOADING_ITEMS: (loadingId) => `${API_BASE_URL}/api/loadings/${loadingId}/items`,
  LOADING_ITEM_UPDATE: (itemId) => `${API_BASE_URL}/api/loadings/items/${itemId}`,
  LOADING_ITEM_DELETE: (itemId) => `${API_BASE_URL}/api/loadings/items/${itemId}`,
  
  // Vision
  VISION_START: `${API_BASE_URL}/api/vision/start`,
  VISION_STOP: `${API_BASE_URL}/api/vision/stop`,
  VISION_STATUS: `${API_BASE_URL}/api/vision/status`,
  
  // Products
  PRODUCTS: `${API_BASE_URL}/api/products`,
  PRODUCT_CREATE: `${API_BASE_URL}/api/products`,
  PRODUCT_UPDATE: (id) => `${API_BASE_URL}/api/products/${id}`,
  PRODUCT_DELETE: (id) => `${API_BASE_URL}/api/products/${id}`,
  
  // Shipments
  SHIPMENTS_LATEST: `${API_BASE_URL}/api/shipments/latest`,
  SHIPMENTS_FOR_UNLOADING: `${API_BASE_URL}/api/shipments/for-unloading`,
  SHIPMENTS_FOR_LOADING: `${API_BASE_URL}/api/shipments/for-loading`,
  SHIPMENT_DETAIL: (id) => `${API_BASE_URL}/api/shipments/${id}`,
  
  // Database Management
  DATABASE_CLOSE: `${API_BASE_URL}/api/database/close`,
  DATABASE_STATUS: `${API_BASE_URL}/api/database/status`,
  
  // SSH Operations
  SSH_TEST: `${API_BASE_URL}/api/ssh/test`,
  SSH_COPY_DATABASE: `${API_BASE_URL}/api/ssh/copy-database`,
  SSH_LIST_FILES: `${API_BASE_URL}/api/ssh/list-files`,
  SSH_HEALTH: `${API_BASE_URL}/api/ssh/health`,
  
  // Debug
  DEBUG_OPERATIONS: `${API_BASE_URL}/api/debug/operations`,
  DEBUG_VISION_SERVERS: `${API_BASE_URL}/api/debug/vision-servers`,
  

}; 