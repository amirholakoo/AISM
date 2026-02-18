from ultralytics import YOLO

# Load the YOLO11n model
model = YOLO('weights/best.pt')  # e.g., 'runs/detect/train/weights/best.pt'

# Export to ONNX with dynamic axes (for multi-image sizes)
# - dynamic=True: Allows variable H/W (e.g., imgsz=640 or 1280)
# - imgsz: Default export size; runtime can vary it
# - opset: Use 12+ for better compatibility (e.g., with OpenCV DNN or TensorRT)
success = model.export(
    format="onnx",
    dynamic=True,      # Enables dynamic input shapes: [batch, 3, H, W] where H/W vary
    imgsz=640,         # Base export size; change to 320/1280 as needed
    opset=12,          # ONNX opset version (higher for newer features)
    simplify=True      # Optional: Simplify graph for faster inference
)

print("Export successful!" if success else "Export failed.")




