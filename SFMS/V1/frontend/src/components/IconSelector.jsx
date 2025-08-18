import React from 'react';
import { 
  Plus, 
  Edit, 
  X, 
  Settings,
  Home,
  BarChart3,
  Truck,
  Upload,
  ShoppingCart,
  ArrowRightLeft,
  RotateCcw,
  Package,
  Warehouse,
  List,
  Eye,
  Users,
  FileText,
  Calendar,
  Mail,
  Phone,
  MapPin,
  Clock,
  Star,
  Heart,
  ThumbsUp,
  MessageCircle,
  Bell,
  Search,
  Filter,
  Download,
  Printer,
  Camera,
  Video,
  Music,
  Book,
  PenTool,
  Scissors,
  Tag,
  Link,
  Share,
  Copy,
  Save,
  Trash,
  Minus,
  Check,
  AlertCircle,
  Info,
  HelpCircle,
  Shield,
  Lock,
  Unlock,
  Key,
  CreditCard,
  DollarSign,
  Percent,
  TrendingUp,
  TrendingDown,
  Activity,
  Zap,
  Target,
  Award,
  Gift,
  Smile,
  Frown,
  Meh
} from 'lucide-react';

export const iconOptions = [
  { value: 'Truck', label: 'کامیون', icon: Truck },
  { value: 'Upload', label: 'آپلود', icon: Upload },
  { value: 'ShoppingCart', label: 'سبد خرید', icon: ShoppingCart },
  { value: 'ArrowRightLeft', label: 'انتقال', icon: ArrowRightLeft },
  { value: 'RotateCcw', label: 'بازگشت', icon: RotateCcw },
  { value: 'Package', label: 'بسته', icon: Package },
  { value: 'Warehouse', label: 'انبار', icon: Warehouse },
  { value: 'List', label: 'لیست', icon: List },
  { value: 'Settings', label: 'تنظیمات', icon: Settings },
  { value: 'Eye', label: 'چشم', icon: Eye },
  { value: 'Home', label: 'خانه', icon: Home },
  { value: 'BarChart3', label: 'نمودار', icon: BarChart3 },
  { value: 'Users', label: 'کاربران', icon: Users },
  { value: 'FileText', label: 'فایل', icon: FileText },
  { value: 'Calendar', label: 'تقویم', icon: Calendar },
  { value: 'Mail', label: 'ایمیل', icon: Mail },
  { value: 'Phone', label: 'تلفن', icon: Phone },
  { value: 'MapPin', label: 'موقعیت', icon: MapPin },
  { value: 'Clock', label: 'ساعت', icon: Clock },
  { value: 'Star', label: 'ستاره', icon: Star },
  { value: 'Heart', label: 'قلب', icon: Heart },
  { value: 'ThumbsUp', label: 'لایک', icon: ThumbsUp },
  { value: 'MessageCircle', label: 'پیام', icon: MessageCircle },
  { value: 'Bell', label: 'زنگ', icon: Bell },
  { value: 'Search', label: 'جستجو', icon: Search },
  { value: 'Filter', label: 'فیلتر', icon: Filter },
  { value: 'Download', label: 'دانلود', icon: Download },
  { value: 'Printer', label: 'پرینتر', icon: Printer },
  { value: 'Camera', label: 'دوربین', icon: Camera },
  { value: 'Video', label: 'ویدیو', icon: Video },
  { value: 'Music', label: 'موسیقی', icon: Music },
  { value: 'Book', label: 'کتاب', icon: Book },
  { value: 'PenTool', label: 'قلم', icon: PenTool },
  { value: 'Scissors', label: 'قیچی', icon: Scissors },
  { value: 'Tag', label: 'برچسب', icon: Tag },
  { value: 'Link', label: 'لینک', icon: Link },
  { value: 'Share', label: 'اشتراک', icon: Share },
  { value: 'Copy', label: 'کپی', icon: Copy },
  { value: 'Save', label: 'ذخیره', icon: Save },
  { value: 'Edit', label: 'ویرایش', icon: Edit },
  { value: 'Trash', label: 'سطل زباله', icon: Trash },
  { value: 'Plus', label: 'افزودن', icon: Plus },
  { value: 'Minus', label: 'حذف', icon: Minus },
  { value: 'Check', label: 'تایید', icon: Check },
  { value: 'X', label: 'انصراف', icon: X },
  { value: 'AlertCircle', label: 'هشدار', icon: AlertCircle },
  { value: 'Info', label: 'اطلاعات', icon: Info },
  { value: 'HelpCircle', label: 'راهنما', icon: HelpCircle },
  { value: 'Shield', label: 'محافظت', icon: Shield },
  { value: 'Lock', label: 'قفل', icon: Lock },
  { value: 'Unlock', label: 'باز کردن', icon: Unlock },
  { value: 'Key', label: 'کلید', icon: Key },
  { value: 'CreditCard', label: 'کارت اعتباری', icon: CreditCard },
  { value: 'DollarSign', label: 'پول', icon: DollarSign },
  { value: 'Percent', label: 'درصد', icon: Percent },
  { value: 'TrendingUp', label: 'صعودی', icon: TrendingUp },
  { value: 'TrendingDown', label: 'نزولی', icon: TrendingDown },
  { value: 'Activity', label: 'فعالیت', icon: Activity },
  { value: 'Zap', label: 'برق', icon: Zap },
  { value: 'Target', label: 'هدف', icon: Target },
  { value: 'Award', label: 'جایزه', icon: Award },
  { value: 'Gift', label: 'هدیه', icon: Gift },
  { value: 'Smile', label: 'لبخند', icon: Smile },
  { value: 'Frown', label: 'ناراحت', icon: Frown },
  { value: 'Meh', label: 'متوسط', icon: Meh }
];

// Function to get icon component by name
export const getIconComponent = (iconName) => {
  const iconOption = iconOptions.find(option => option.value === iconName);
  return iconOption ? iconOption.icon : Settings; // Default to Settings if not found
};

// IconSelector component for use in forms
export const IconSelector = ({ value, onChange, label, id, className = "" }) => {
  return (
    <div className={className}>
      {label && <label htmlFor={id} className="block text-sm font-medium text-gray-700 mb-2">{label}</label>}
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
      >
        <option value="">انتخاب آیکون</option>
        {iconOptions.map(icon => (
          <option key={icon.value} value={icon.value}>
            {icon.label}
          </option>
        ))}
      </select>
    </div>
  );
};

export default IconSelector; 