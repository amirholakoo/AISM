import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { BarChart3 } from 'lucide-react';

const AdminButton = ({ 
  className = "",
  size = "default",
  variant = "outline",
  onClick,
  disabled = false
}) => {
  const navigate = useNavigate();

  const handleClick = () => {
    if (onClick) {
      onClick();
    } else {
      navigate("/admin");
    }
  };

  const getSizeClasses = () => {
    switch (size) {
      case "sm":
        return "h-8 w-8 p-0";
      case "lg":
        return "h-12 w-12 p-0";
      default:
        return "h-10 w-10 p-0";
    }
  };

  const getIconSize = () => {
    switch (size) {
      case "sm":
        return "h-4 w-4";
      case "lg":
        return "h-6 w-6";
      default:
        return "h-5 w-5";
    }
  };

  return (
    <Button
      variant={variant}
      size={size}
      onClick={handleClick}
      disabled={disabled}
      className={`${getSizeClasses()} bg-white hover:bg-slate-50 border-slate-300 hover:border-slate-400 transition-all duration-200 ${className}`}
    >
      <BarChart3 className={getIconSize()} />
    </Button>
  );
};

export default AdminButton; 