import requests
import csv
import ast
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import interpolate
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class TemperatureDataPreprocessor:
    def __init__(self, api_url, raw_csv="DHT22_data_raw.csv", 
                 filled_csv="data_filled.csv", hourly_csv="Modified_Timeseries_Temperature_and_Humidity_1_Hour.csv"):
        self.api_url = api_url
        self.raw_csv = raw_csv
        self.filled_csv = filled_csv
        self.hourly_csv = hourly_csv
        
    def extract_raw_data(self):
        """Step 1: Extract data from API with validation"""
        print("=== Step 1: Extracting raw data from API ===")
        try:
            response = requests.get(self.api_url, timeout=30)
            response.raise_for_status()
            sensor_data = response.json()
            
            processed_data = []
            
            for entry in sensor_data:
                try:
                    raw_data_str = entry.get("data", "{}")
                    data_dict = ast.literal_eval(raw_data_str)
                    
                    temperature = data_dict.get('temperature')
                    humidity = data_dict.get('humidity')
                    last_update_unix = entry.get('LastUpdate')
                    
                    # Validate data
                    if temperature is None or humidity is None:
                        continue
                    if not isinstance(temperature, (int, float)) or not isinstance(humidity, (int, float)):
                        continue
                    if temperature < -50 or temperature > 80 or humidity < 0 or humidity > 100:
                        print(f"Skipping invalid values: T={temperature}, H={humidity}")
                        continue
                    
                    last_update_readable = datetime.fromtimestamp(last_update_unix).strftime('%Y-%m-%d %H:%M:%S')
                    processed_data.append([last_update_readable, float(temperature), float(humidity)])
                    
                except (ValueError, SyntaxError, KeyError) as e:
                    continue
            
            # Sort by timestamp
            processed_data.sort(key=lambda x: x[0])
            
            # Save raw data
            with open(self.raw_csv, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time steps", "Temperature", "Humidity"])
                writer.writerows(processed_data)
            
            print(f"✓ Extracted and saved {len(processed_data)} valid records to {self.raw_csv}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return []
    
    def analyze_gaps(self, df):
        """Analyze missing data patterns and prepare properly indexed DataFrame"""
        print("\n=== Step 2: Analyzing data gaps ===")
        
        # Ensure datetime index is set properly
        df['Time steps'] = pd.to_datetime(df['Time steps'])
        df = df.sort_values('Time steps').drop_duplicates('Time steps').set_index('Time steps')
        
        # Create expected minute-level timeline
        start_time = df.index.min()
        end_time = df.index.max()
        expected_times = pd.date_range(start=start_time, end=end_time, freq='T')  # Minute-level
        
        print(f"Data range: {start_time} to {end_time}")
        print(f"Original records: {len(df)}")
        
        # Create complete timeline with NaNs
        df_complete = pd.DataFrame(index=expected_times)
        df_complete = df_complete.join(df[['Temperature', 'Humidity']], how='left')
        
        # Gap analysis
        temp_missing = df_complete['Temperature'].isna().sum()
        missing_pct = temp_missing / len(df_complete) * 100
        
        print(f"Missing temperature points: {temp_missing} ({missing_pct:.1f}%)")
        print(f"Missing humidity points: {df_complete['Humidity'].isna().sum()}")
        
        # Find gap lengths
        gaps = df_complete['Temperature'].isna()
        gap_lengths = []
        current_gap = 0
        for g in gaps:
            if g:
                current_gap += 1
            else:
                if current_gap > 0:
                    gap_lengths.append(current_gap)
                    current_gap = 0
        if current_gap > 0:
            gap_lengths.append(current_gap)
        
        if gap_lengths:
            print(f"Gap statistics:")
            print(f"  Average gap: {np.mean(gap_lengths):.1f} minutes")
            print(f"  Largest gap: {np.max(gap_lengths)} minutes ({np.max(gap_lengths)/60:.1f} hours)")
        
        return df_complete, df  # Return both complete and original indexed
    
    def intelligent_imputation(self, df_complete):
        """Step 3: Intelligent gap filling with tiered approach"""
        print("\n=== Step 3: Intelligent gap imputation ===")
        df_imputed = df_complete.copy()
        
        # Method 1: Linear interpolation for short gaps (<6 hours = 360 minutes)
        df_imputed['Temperature'] = df_complete['Temperature'].interpolate(method='linear', limit=360)
        df_imputed['Humidity'] = df_complete['Humidity'].interpolate(method='linear', limit=360)
        
        # Method 2: Handle remaining gaps with boundary interpolation
        remaining_temp_na = df_imputed['Temperature'].isna()
        if remaining_temp_na.any():
            print("Handling remaining gaps with boundary interpolation...")
            for idx in df_imputed.index[remaining_temp_na]:
                # Find nearest valid points
                before_mask = df_imputed.index < idx
                after_mask = df_imputed.index > idx
                
                before_idx = df_imputed.loc[before_mask, 'Temperature'].dropna().index[-1] if before_mask.any() else None
                after_idx = df_imputed.loc[after_mask, 'Temperature'].dropna().index[0] if after_mask.any() else None
                
                if before_idx is not None and after_idx is not None:
                    # Linear interpolation between boundaries
                    time_before = df_imputed.index.get_loc(before_idx)
                    time_after = df_imputed.index.get_loc(after_idx)
                    time_current = df_imputed.index.get_loc(idx)
                    
                    fraction = (time_current - time_before) / (time_after - time_before)
                    temp_before = df_imputed.at[before_idx, 'Temperature']
                    temp_after = df_imputed.at[after_idx, 'Temperature']
                    df_imputed.at[idx, 'Temperature'] = temp_before + fraction * (temp_after - temp_before)
        
        # Method 3: Large gaps - Seasonal pattern
        large_gaps_remaining = df_imputed['Temperature'].isna()
        if large_gaps_remaining.any():
            print("Handling large gaps with seasonal pattern...")
            
            # Extract daily temperature pattern
            daily_pattern = []
            for hour in range(24):
                hourly_data = df_imputed[df_imputed.index.hour == hour]
                valid_temps = hourly_data['Temperature'].dropna()
                if len(valid_temps) > 5:
                    daily_pattern.append(valid_temps.mean())
                else:
                    daily_pattern.append(np.nan)
            
            daily_pattern = pd.Series(daily_pattern).interpolate().fillna(method='ffill').fillna(method='bfill').values
            
            for idx in df_imputed.index[large_gaps_remaining]:
                hour_of_day = idx.hour
                if 0 <= hour_of_day < len(daily_pattern):
                    df_imputed.at[idx, 'Temperature'] = daily_pattern[hour_of_day]
        
        # Final cleanup
        df_imputed['Temperature'] = df_imputed['Temperature'].fillna(method='ffill').fillna(method='bfill')
        df_imputed['Humidity'] = df_imputed['Humidity'].fillna(method='ffill').fillna(method='bfill')
        
        if df_imputed.isnull().any().any():
            print("Warning: Some NaNs remain after imputation")
        else:
            print("All gaps successfully filled")
        
        return df_imputed
    
    def resample_hourly(self, df_imputed):
        """Step 4: Resample to hourly data"""
        print("\n=== Step 4: Resampling to hourly ===")
        
        df_hourly = df_imputed.resample('H').agg({
            'Temperature': 'mean',
            'Humidity': 'mean'
        }).dropna()
        
        df_hourly = df_hourly.reset_index()
        df_hourly.rename(columns={'index': 'Time steps'}, inplace=True)
        
        df_hourly.to_csv(self.hourly_csv, index=False)
        print(f"✓ Created {len(df_hourly)} hourly records")
        print(f"Hourly range: {df_hourly['Time steps'].min()} to {df_hourly['Time steps'].max()}")
        
        return df_hourly
    
    def visualize_results(self, df_original_indexed, df_imputed, df_hourly):
        """Step 5: Visualize with properly indexed data"""
        print("\n=== Step 5: Visualizing results ===")
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Original vs Filled (last 7 days)
        sample_end = df_imputed.index.max()
        sample_start = sample_end - timedelta(days=7)
        
        # Filter imputed data
        sample_imputed = df_imputed.loc[sample_start:sample_end]
        
        # Filter original data (now properly indexed)
        original_sample = df_original_indexed.loc[sample_start:sample_end]['Temperature'].dropna()
        
        axes[0].plot(sample_imputed.index, sample_imputed['Temperature'], 'b-', label='Filled', alpha=0.7)
        if len(original_sample) > 0:
            axes[0].scatter(original_sample.index, original_sample.values, color='red', s=10, label='Original', alpha=0.7)
        axes[0].set_title('Temperature: Original vs Filled (Last 7 Days)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Daily pattern
        daily_pattern = []
        for hour in range(24):
            hourly_data = df_imputed[df_imputed.index.hour == hour]
            valid_temps = hourly_data['Temperature'].dropna()
            if len(valid_temps) > 0:
                daily_pattern.append(valid_temps.mean())
            else:
                daily_pattern.append(np.nan)
        
        hours = np.arange(24)
        axes[1].plot(hours, daily_pattern, 'go-', linewidth=2, markersize=8)
        axes[1].set_title('Extracted Daily Temperature Pattern')
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Average Temperature (°C)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(hours)
        
        # Plot 3: Hourly data (last 48 hours)
        if len(df_hourly) >= 48:
            hourly_sample = df_hourly.tail(48)
            axes[2].plot(pd.to_datetime(hourly_sample['Time steps']), hourly_sample['Temperature'], 
                        'purple', linewidth=2)
        else:
            axes[2].plot(pd.to_datetime(df_hourly['Time steps']), df_hourly['Temperature'], 'purple', linewidth=2)
        
        axes[2].set_title('Final Hourly Data (Last 48 Hours)')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Temperature (°C)')
        axes[2].grid(True, alpha=0.3)
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('complete_preprocessing_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_pipeline(self, start_date=None, end_date=None):
        """Run the entire preprocessing pipeline"""
        print("Starting Complete Temperature Data Preprocessing Pipeline")
        print("=" * 60)
        
        # Step 1: Extract raw data
        raw_data = self.extract_raw_data()
        if not raw_data:
            print("Failed to extract data. Exiting.")
            return None
        
        # Step 2: Load, analyze, and index raw data properly
        df_raw = pd.read_csv(self.raw_csv)
        df_complete, df_original_indexed = self.analyze_gaps(df_raw)
        
        # Step 3: Intelligent imputation
        df_imputed = self.intelligent_imputation(df_complete)
        
        # Step 4: Save filled data
        df_imputed.to_csv(self.filled_csv)
        print(f"✓ Filled minute-level data saved to {self.filled_csv}")
        
        # Step 5: Resample to hourly
        df_hourly = self.resample_hourly(df_imputed)
        
        # Step 6: Visualize with properly indexed data
        self.visualize_results(df_original_indexed, df_imputed, df_hourly)
        
        print("Pipeline completed successfully!")
        print(f"Final dataset: {self.hourly_csv} (ready for ML training)")
        print(f"Total hourly records: {len(df_hourly)}")
        
        return df_hourly

# Usage
if __name__ == "__main__":
    # Configuration
    API_URL = "http://192.168.2.20:7500/dashboard/device_info_json/4/"
    
    # Initialize preprocessor
    preprocessor = TemperatureDataPreprocessor(
        api_url=API_URL,
        raw_csv="DHT22_data_raw.csv",
        filled_csv="data_filled_minute_level.csv",
        hourly_csv="Modified_Timeseries_Temperature_and_Humidity_1_Hour.csv"
    )
    
    # Run complete pipeline
    df_final = preprocessor.run_complete_pipeline()
    
    if df_final is not None:
        print(f"Ready to train ML models with {len(df_final)} clean hourly records!")
        print(f"Files generated:")
        print(f"  - {preprocessor.raw_csv} (raw API data)")
        print(f"  - {preprocessor.filled_csv} (filled minute-level)")
        print(f"  - {preprocessor.hourly_csv} (final hourly dataset)")