import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
    AlertDialogTrigger,
  } from "@/components/ui/alert-dialog";
  import { Button } from "@/components/ui/button";
  import { XIcon } from "lucide-react";
  
  export default function DeleteConfirmDialog({ itemName, itemType, onConfirm, children, disabled = false }) {
      const getTitle = () => {
    if (itemType === "loaded") return "حذف بارگیری";
    if (itemType === "vision-server") return "حذف سرور بینایی";
    return "حذف تخلیه";
  };
  
    return (
      <AlertDialog>
        <AlertDialogTrigger asChild disabled={disabled}>
          {children || (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="text-red-500 hover:bg-red-100"
              disabled={disabled}
            >
              <XIcon className="w-4 h-4" />
            </Button>
          )}
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{getTitle()}</AlertDialogTitle>
            <AlertDialogDescription>
              آیا از حذف "{itemName}" اطمینان دارید؟ این عملیات قابل بازگشت نیست.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>انصراف</AlertDialogCancel>
            <AlertDialogAction onClick={onConfirm}>
              حذف
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    );
  }