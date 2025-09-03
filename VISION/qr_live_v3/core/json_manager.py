import os
import json
from datetime import datetime

class JsonLogManager:
    """Handles the creation and management of a timestamped JSON log file for each run."""
    def __init__(self, config):
        self.config = config
        self.run_start_time = datetime.now(self.config.TIMEZONE).isoformat()
        self.records = []
        
        # Ensure output directory exists
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def add_record(self, qr_data):
        """Adds a QR code record to the list of detections for this run."""
        self.records.append(qr_data)
        print(f"💾 Logged new unique QR code: {qr_data['content']}")

    def finalize_log(self):
        """Finalizes the log with a run finish time and saves it to a unique file."""
        run_finish_time = datetime.now(self.config.TIMEZONE).isoformat()
        
        # Create a unique filename based on the start time
        start_time_obj = datetime.fromisoformat(self.run_start_time)
        filename = f"qrcodes_{start_time_obj.strftime('%Y%m%d_%H%M%S')}.json"
        output_file_path = os.path.join(self.config.OUTPUT_DIR, filename)

        output_data = {
            "run_start_time": self.run_start_time,
            "run_finish_time": run_finish_time,
            "qrcodes": self.records
        }

        try:
            with open(output_file_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"📝 Log file saved to {output_file_path}")
        except IOError as e:
            print(f"❌ Error saving JSON log file: {e}") 