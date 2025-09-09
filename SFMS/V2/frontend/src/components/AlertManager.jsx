import React, { useEffect, useRef } from "react";
import { toast } from "sonner";
import { 
  showSuccess, 
  showError, 
  showInfo, 
  showWarning,
  showConnectedToExisting,
  showEditTimeWarning,
  showEditTimeExpired
} from "@/lib/toast";

const AlertManager = ({
  showEditExpiredAlert,
  setShowEditExpiredAlert,
  editingLoading,
  canEdit,
  remainingMinutes,
  showRemainingTimeAlert,
  setShowRemainingTimeAlert,
  connectedToExisting,
  setConnectedToExisting,
  started,
  message,
  error,
  setMessage,
  setError
}) => {
  const lastMessageRef = useRef("");

  // نمایش toast برای پیام‌های عادی
  useEffect(() => {
    if (message && !error && message !== lastMessageRef.current) {
      // پاک کردن toast های قبلی
      toast.dismiss();
      lastMessageRef.current = message;
      showSuccess(message);
      // پاک کردن پیام بعد از نمایش
      setTimeout(() => {
        setMessage("");
        lastMessageRef.current = "";
      }, 100);
    }
  }, [message, error, setMessage]);

  // نمایش toast برای خطاها
  useEffect(() => {
    if (message && error && message !== lastMessageRef.current) {
      // پاک کردن toast های قبلی
      toast.dismiss();
      lastMessageRef.current = message;
      showError(message);
      // پاک کردن پیام بعد از نمایش
      setTimeout(() => {
              setMessage("");
              setError(false);
        lastMessageRef.current = "";
      }, 100);
    }
  }, [message, error, setMessage, setError]);

  // نمایش toast برای اتصال به بارگیری موجود
  useEffect(() => {
    if (connectedToExisting && started) {
      showConnectedToExisting();
      setConnectedToExisting(false);
    }
  }, [connectedToExisting, started, setConnectedToExisting]);

  // نمایش toast برای هشدار زمان ویرایش
  useEffect(() => {
    if (showRemainingTimeAlert && editingLoading && canEdit && remainingMinutes > 0 && remainingMinutes <= 5) {
      showEditTimeWarning(remainingMinutes);
      setShowRemainingTimeAlert(false);
    }
  }, [showRemainingTimeAlert, editingLoading, canEdit, remainingMinutes, setShowRemainingTimeAlert]);

  // نمایش toast برای انقضای زمان ویرایش - فقط وقتی که واقعاً زمان تمام شده باشد
  useEffect(() => {
    if (showEditExpiredAlert && editingLoading && !canEdit && remainingMinutes === 0) {
      showEditTimeExpired();
      setShowEditExpiredAlert(false);
    }
  }, [showEditExpiredAlert, editingLoading, canEdit, remainingMinutes, setShowEditExpiredAlert]);

  // این کامپوننت دیگر چیزی render نمی‌کنه، فقط toast ها رو مدیریت می‌کنه
  return null;
};

export default AlertManager;