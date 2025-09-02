from ultralytics import YOLO

# --- Configuration ---
# Path to your new fine-tuned PyTorch model.
# <<<<<<< IMPORTANT: UPDATE THIS FILENAME IF NEEDED >>>>>>>>>
PT_MODEL_PATH = "100_epoch_650_yolov11n_v3.pt"

# --- Main Export Logic ---
def main():
    """
    This script exports a trained YOLO .pt model to the ONNX format.
    It is simplified for direct conversion without quantization.
    """
    print(f"Loading PyTorch model from: {PT_MODEL_PATH}")
    try:
        # Load your fine-tuned PyTorch model
        model = YOLO(PT_MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load the model file at '{PT_MODEL_PATH}'. Please ensure the path is correct.")
        print(f"   Details: {e}")
        return

    # --- Export to Standard FP32 ONNX ---
    print("\nExporting to ONNX model...")
    try:
        model.export(
            format='onnx',
            imgsz=640,          # The base image size the model was trained on
            dynamic=True,       # CRITICAL: Allows for variable input sizes
            simplify=True,      # Optimizes the graph for faster inference
            opset=12,           # A stable ONNX opset version
            nms=True            # RECOMMENDED: Bakes post-processing into the model
        )
        output_filename = PT_MODEL_PATH.replace('.pt', '.onnx')
        print(f"✅ ONNX model exported successfully!")
        print(f"   -> Your new model is saved as: '{output_filename}'")
    except Exception as e:
        print(f"❌ ERROR during ONNX export: {e}")

if __name__ == "__main__":
    main()
    print("\n--- How to Run ---")
    print(f"1. Make sure the model '{PT_MODEL_PATH}' is in the same directory.")
    print("2. Run the script from your terminal: python export_model.py")
    print("3. After it succeeds, update 'config/warehouse_config.py' to use the new .onnx file name.")
    print("------------------")
