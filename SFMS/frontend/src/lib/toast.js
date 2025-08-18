import { toast } from "sonner";

// Toast configurations
const TOAST_CONFIG = {
  success: {
    duration: 8000,
    position: "top-center",
    style: {
      background: "#f0fdf4",
      border: "1px solid #bbf7d0",
      color: "#166534",
      fontSize: "14px",
      fontWeight: "500",
      fontFamily: "'Shabnam', sans-serif",
      direction: "rtl",
      textAlign: "right"
    }
  },
  error: {
    duration: 10000,
    position: "top-center",
    style: {
      background: "#fef2f2",
      border: "1px solid #fecaca",
      color: "#dc2626",
      fontSize: "14px",
      fontWeight: "500",
      fontFamily: "'Shabnam', sans-serif",
      direction: "rtl",
      textAlign: "right"
    }
  },
  warning: {
    duration: 6000,
    position: "top-center",
    style: {
      background: "#fffbeb",
      border: "1px solid #fed7aa",
      color: "#d97706",
      fontSize: "14px",
      fontWeight: "500",
      fontFamily: "'Shabnam', sans-serif",
      direction: "rtl",
      textAlign: "right"
    }
  },
  info: {
    duration: 6000,
    position: "top-center",
    style: {
      background: "#eff6ff",
      border: "1px solid #bfdbfe",
      color: "#1d4ed8",
      fontSize: "14px",
      fontWeight: "500",
      fontFamily: "'Shabnam', sans-serif",
      direction: "rtl",
      textAlign: "right"
    }
  }
};

// Toast helper functions
export const showSuccess = (message, id = null) => {
  toast.success(message, {
    ...TOAST_CONFIG.success,
    id: id || `success-${Date.now()}`
  });
};

export const showError = (message, id = null) => {
  toast.error(message, {
    ...TOAST_CONFIG.error,
    id: id || `error-${Date.now()}`
  });
};

export const showWarning = (message, id = null) => {
  toast.warning(message, {
    ...TOAST_CONFIG.warning,
    id: id || `warning-${Date.now()}`
  });
};

export const showInfo = (message, id = null) => {
  toast.info(message, {
    ...TOAST_CONFIG.info,
    id: id || `info-${Date.now()}`
  });
};

// Specific toast messages for the app
export const showLoadingStarted = (warehouseName) => {
  showSuccess(`بارگیری در انبار ${warehouseName} شروع شد`);
};

export const showLoadingEnded = () => {
  showSuccess("بارگیری با موفقیت پایان یافت");
};

export const showEditSaved = () => {
  showSuccess("ویرایش با موفقیت ذخیره شد");
};

export const showConnectedToExisting = () => {
  showInfo("بارگیری در انبار انتخاب شده در حال اجرا است. به آن متصل شدید.");
};

export const showEditTimeWarning = (remainingMinutes) => {
  showWarning(`⚠️ زمان ویرایش: ${remainingMinutes} دقیقه باقی مانده`);
};

export const showEditTimeExpired = () => {
  showError("⏰ زمان ویرایش به پایان رسیده است!");
};

export const showNetworkError = () => {
  showError("خطا در ارتباط با سرور. لطفاً دوباره تلاش کنید.");
};

export const showValidationError = (message) => {
  showError(message || "اطلاعات وارد شده صحیح نیست");
};

export const showOfflineServerWarning = () => {
  showWarning("⚠️ سرور خارجی (SSH) در دسترس نیست. از دیتابیس قبلی استفاده می‌شود.");
}; 