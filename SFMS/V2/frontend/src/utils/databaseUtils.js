import { API_ENDPOINTS } from "@/config";

// تابع کمکی برای مدیریت اتصال دیتابیس
export const manageDatabaseConnection = async () => {
  try {
    console.log("🔒 Closing database connections...");
    const closeRes = await fetch(API_ENDPOINTS.DATABASE_CLOSE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(3000) // 3 second timeout
    });
    
    if (closeRes.ok) {
      const closeData = await closeRes.json();
      console.log("✅ Database connections closed:", closeData.message);
      return true;
    } else {
      console.warn("⚠️ Could not close database connections");
      return false;
    }
  } catch (error) {
    console.error("❌ Error closing database connections:", error);
    return false;
  }
};

// تابع کمکی برای کپی دیتابیس از طریق SSH
export const copyDatabaseViaSSH = async () => {
  try {
    console.log("🔗 Copying database via SSH...");
    const res = await fetch(API_ENDPOINTS.SSH_COPY_DATABASE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: 'localnew.sqlite3' }),
      signal: AbortSignal.timeout(1000) // 1 second timeout
    });
    
    if (res.ok) {
      const data = await res.json();
      if (data.success) {
        console.log("✅ Database copied successfully:", data.message);
        return true;
      } else {
        console.error("❌ Failed to copy database:", data.message);
        return false;
      }
    } else {
      const errorData = await res.json();
      console.error("❌ Failed to copy database:", errorData.message);
      return false;
    }
  } catch (error) {
    console.error("❌ Error copying database via SSH:", error);
    return false;
  }
};

// تابع کمکی برای تست اتصال SSH
export const testSSHConnection = async () => {
  try {
    console.log("🔗 Testing SSH connection...");
    const res = await fetch(API_ENDPOINTS.SSH_TEST, {
      signal: AbortSignal.timeout(1000) // 1 second timeout
    });
    
    if (res.ok) {
      const data = await res.json();
      if (data.success) {
        console.log("✅ SSH connection test successful:", data.message);
        return true;
      } else {
        console.error("❌ SSH connection test failed:", data.message);
        return false;
      }
    } else {
      const errorData = await res.json();
      console.error("❌ SSH connection test failed:", errorData.message);
      return false;
    }
  } catch (error) {
    console.error("❌ Error testing SSH connection:", error);
    return false;
  }
};

// تابع کمکی برای بررسی سریع وضعیت دیتابیس (بدون تلاش مکرر)
export const quickDatabaseCheck = async () => {
  try {
    console.log("🔍 Quick database status check...");
    const statusRes = await fetch(API_ENDPOINTS.DATABASE_STATUS, {
      signal: AbortSignal.timeout(2000) // 2 second timeout
    });
    
    if (statusRes.ok) {
      const statusData = await statusRes.json();
      if (statusData.success) {
        console.log("✅ Database is accessible (quick check)");
        return true;
      } else {
        console.warn("⚠️ Database not accessible (quick check):", statusData.message);
        return false;
      }
    } else {
      console.warn("⚠️ Could not check database status (quick check): HTTP", statusRes.status);
      return false;
    }
  } catch (error) {
    console.error("❌ Error in quick database check:", error.message);
    return false;
  }
};

// تابع کمکی برای بررسی وضعیت دیتابیس با تلاش چندباره
export const checkDatabaseStatus = async (maxRetries = 2, delayMs = 200) => {
  console.log(`🔄 Starting database status check with ${maxRetries} attempts...`);
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`🔍 Checking database status (attempt ${attempt}/${maxRetries})...`);
      const statusRes = await fetch(API_ENDPOINTS.DATABASE_STATUS, {
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      
      if (statusRes.ok) {
        const statusData = await statusRes.json();
        if (statusData.success) {
          console.log(`✅ Database is accessible (attempt ${attempt})`);
          console.log(`📊 Database info: ${statusData.data?.shipments_count || 'N/A'} shipments available`);
          return true;
        } else {
          console.warn(`⚠️ Database not accessible (attempt ${attempt}):`, statusData.message);
        }
      } else {
        console.warn(`⚠️ Could not check database status (attempt ${attempt}): HTTP ${statusRes.status}`);
      }
    } catch (error) {
      console.error(`❌ Error checking database status (attempt ${attempt}):`, error.message);
    }
    
    // اگر آخرین تلاش نیست، کمی صبر کن
    if (attempt < maxRetries) {
      console.log(`⏳ Waiting ${delayMs}ms before next attempt...`);
      await new Promise(resolve => setTimeout(resolve, delayMs));
    }
  }
  
  console.warn(`⚠️ Database might not be accessible after ${maxRetries} attempts, but continuing...`);
  return false;
};

// تابع کمکی برای بررسی وضعیت دیتابیس با تلاش‌های بیشتر (برای مواقع بحرانی)
export const checkDatabaseStatusExtended = async (maxRetries = 3, initialDelayMs = 150, maxDelayMs = 500) => {
  console.log(`🔄 Starting extended database status check with ${maxRetries} attempts...`);
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`🔍 Extended database status check (attempt ${attempt}/${maxRetries})...`);
      const statusRes = await fetch(API_ENDPOINTS.DATABASE_STATUS, {
        signal: AbortSignal.timeout(3000) // 3 second timeout
      });
      
      if (statusRes.ok) {
        const statusData = await statusRes.json();
        if (statusData.success) {
          console.log(`✅ Database is accessible (extended attempt ${attempt})`);
          console.log(`📊 Database info: ${statusData.data?.shipments_count || 'N/A'} shipments available`);
          return true;
        } else {
          console.warn(`⚠️ Database not accessible (extended attempt ${attempt}):`, statusData.message);
        }
      } else {
        console.warn(`⚠️ Could not check database status (extended attempt ${attempt}): HTTP ${statusRes.status}`);
      }
    } catch (error) {
      console.error(`❌ Error checking database status (extended attempt ${attempt}):`, error.message);
    }
    
    // اگر آخرین تلاش نیست، با تاخیر افزایشی صبر کن
    if (attempt < maxRetries) {
      const delayMs = Math.min(initialDelayMs * attempt, maxDelayMs);
      console.log(`⏳ Waiting ${delayMs}ms before next extended attempt...`);
      await new Promise(resolve => setTimeout(resolve, delayMs));
    }
  }
  
  console.warn(`⚠️ Database might not be accessible after ${maxRetries} extended attempts, but continuing...`);
  return false;
};

// تابع کمکی برای بررسی وضعیت سرور خارجی
export const checkRemoteServerHealth = async () => {
  try {
    console.log("🔍 Checking remote server health...");
    const res = await fetch(API_ENDPOINTS.SSH_HEALTH, {
      signal: AbortSignal.timeout(3000) // 3 second timeout
    });
    
    if (res.ok) {
      const data = await res.json();
      if (data.success) {
        console.log("✅ Remote server is healthy:", data.message);
        return true;
      } else {
        console.warn("⚠️ Remote server health check failed:", data.message);
        return false;
      }
    } else {
      console.warn("⚠️ Remote server health check failed: HTTP", res.status);
      return false;
    }
  } catch (error) {
    console.error("❌ Error checking remote server health:", error);
    return false;
  }
}; 