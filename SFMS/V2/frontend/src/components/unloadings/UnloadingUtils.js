// دریافت رنگ badge بر اساس وضعیت
export const getStatusColor = (status) => {
  switch (status) {
    case 'completed':
      return 'bg-green-100 text-green-800 border-green-200';
    case 'vision':
      return 'bg-blue-100 text-blue-800 border-blue-200';
    case 'edited':
      return 'bg-purple-100 text-purple-800 border-purple-200';
    case 'active':
      return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    default:
      return 'bg-gray-100 text-gray-800 border-gray-200';
  }
};

// دریافت متن وضعیت
export const getStatusText = (status) => {
  switch (status) {
    case 'completed':
      return 'تکمیل شده';
    case 'vision':
      return 'نسخه بینایی';
    case 'edited':
      return 'ویرایش شده';
    case 'active':
      return 'فعال';
    default:
      return status;
  }
};

// دریافت آیکون نوع
export const getTypeIcon = (type) => {
  switch (type) {
    case 'vision':
      return 'BotIcon';
    case 'user':
      return 'UserIcon';
    case 'history':
      return 'HistoryIcon';
    default:
      return 'PackageIcon';
  }
};

// دریافت متن نوع عملیات
export const getOperationTypeText = (type) => {
  switch (type) {
    case 'loading':
      return 'بارگیری';
    case 'unloading':
      return 'تخلیه';
    default:
      return 'عملیات';
  }
};

// استخراج ID از token
export const extractIdFromToken = (token) => {
  if (!token) return '';
  
  // token format: loading_61_1755506102 or unloading_15_1755506055
  const parts = token.split('_');
  if (parts.length >= 2) {
    return parts[1]; // Return the ID part (second part)
  }
  
  return token; // Return original token if format is unexpected
};

// دریافت آیکون وضعیت
export const getStatusIcon = (status) => {
  switch (status) {
    case 'completed':
      return 'CheckCircleIcon';
    case 'active':
      return 'AlertCircleIcon';
    case 'edited':
      return 'EditIcon';
    default:
      return 'PackageIcon';
  }
};

// گروه‌بندی آیتم‌ها بر اساس نسخه و منبع
export const groupItemsByVersionAndSource = (items) => {
  if (!items || items.length === 0) return [];
  
  const groups = {};
  
  items.forEach(item => {
    const version = item.version || 1;
    const source = item.source || 'unknown';
    const key = `${version}-${source}`;
    
    if (!groups[key]) {
      groups[key] = {
        version: version,
        source: source,
        items: []
      };
    }
    groups[key].items.push(item);
  });
  
  // مرتب‌سازی بر اساس نسخه (نزولی) و منبع
  return Object.values(groups).sort((a, b) => {
    if (a.version !== b.version) {
      return b.version - a.version; // نسخه جدیدتر اول
    }
    // اگر نسخه یکسان است، منبع user اول
    if (a.source === 'user' && b.source !== 'user') return -1;
    if (b.source === 'user' && a.source !== 'user') return 1;
    return a.source.localeCompare(b.source);
  });
};

// دریافت عنوان گروه
export const getGroupTitle = (group) => {
  const sourceText = group.source === 'vision' ? 'بینایی' : 
                    group.source === 'user' ? 'کاربر' : group.source;
  const versionText = getVersionText(group.version);
  return `نسخه ${versionText} - ${sourceText}`;
};

// دریافت رنگ badge برای منبع
export const getSourceBadgeColor = (source) => {
  switch (source) {
    case 'vision':
      return 'bg-blue-100 text-blue-800 border-blue-200';
    case 'user':
      return 'bg-purple-100 text-purple-800 border-purple-200';
    default:
      return 'bg-gray-100 text-gray-800 border-gray-200';
  }
};

// تبدیل عدد نسخه به متن فارسی
export const getVersionText = (version) => {
  const versionMap = {
    1: 'اول',
    2: 'دوم', 
    3: 'سوم',
    4: 'چهارم',
    5: 'پنجم',
    6: 'ششم',
    7: 'هفتم',
    8: 'هشتم',
    9: 'نهم'
  };
  
  return versionMap[version] || version;
};

// محاسبه درصد پیشرفت
export const getProgressPercentage = (unloading) => {
  const total = unloading.items_count || 0;
  const loaded = unloading.loaded_count || 0;
  return total > 0 ? Math.round((loaded / total) * 100) : 0;
};

// فرمت تاریخ
export const formatDate = (dateString) => {
  if (!dateString) return "نامشخص";
  return new Date(dateString).toLocaleDateString('fa-IR');
};

// فرمت زمان
export const formatTime = (dateString) => {
  if (!dateString) return "نامشخص";
  return new Date(dateString).toLocaleTimeString('fa-IR', { 
    hour: '2-digit', 
    minute: '2-digit' 
  });
}; 