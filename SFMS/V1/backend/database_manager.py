#!/usr/bin/env python3
"""
Database Connection Manager for External SQLite Database
This script helps manage the external database connections for file replacement
"""

import requests
import time
import os
from pathlib import Path

# API base URL
API_BASE_URL = "http://192.168.2.46:18888"

def close_database():
    """Close external database connections via API"""
    try:
        print("🔒 Closing database connections...")
        response = requests.post(f"{API_BASE_URL}/api/database/close")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Database connections closed successfully")
                print(f"📝 Message: {result.get('message')}")
                return True
            else:
                print(f"❌ Failed to close database: {result.get('message')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure the Flask app is running.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_database_status():
    """Check if external database is accessible"""
    try:
        print("🔍 Checking database status...")
        response = requests.get(f"{API_BASE_URL}/api/database/status")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                data = result.get('data', {})
                print("✅ Database is accessible")
                print(f"📊 Shipments count: {data.get('shipments_count', 'N/A')}")
                print(f"📈 Status: {data.get('status', 'N/A')}")
                return True
            else:
                print(f"❌ Database not accessible: {result.get('message')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure the Flask app is running.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def replace_database_file():
    """Replace the external database file via SSH"""
    try:
        print("🔄 Replacing database file...")
        
        # Check if the external database file exists
        db_path = Path(__file__).parent / "external_db" / "localnew.sqlite3"
        
        if not db_path.exists():
            print(f"❌ Database file not found at: {db_path}")
            return False
        
        print(f"📁 Current database file: {db_path}")
        print(f"📏 File size: {db_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Here you would add your SSH command to download the new database
        # For example:
        # os.system("scp user@remote-server:/path/to/localnew.sqlite3 external_db/")
        
        print("⚠️  Please manually replace the database file via SSH")
        print("   Example SSH command:")
        print("   scp user@remote-server:/path/to/localnew.sqlite3 external_db/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error replacing database file: {e}")
        return False

def main():
    """Main function to manage database operations"""
    print("🗄️  External Database Manager")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Close database connections")
        print("2. Check database status")
        print("3. Replace database file (manual)")
        print("4. Full workflow (close → replace → check)")
        print("5. Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == "1":
            close_database()
            
        elif choice == "2":
            check_database_status()
            
        elif choice == "3":
            replace_database_file()
            
        elif choice == "4":
            print("\n🔄 Starting full workflow...")
            
            # Step 1: Close database
            if close_database():
                print("\n⏳ Waiting 2 seconds...")
                time.sleep(2)
                
                # Step 2: Replace file
                if replace_database_file():
                    print("\n⏳ Waiting for file replacement...")
                    input("Press Enter when file replacement is complete...")
                    
                    # Step 3: Check status
                    print("\n🔍 Checking new database...")
                    check_database_status()
                else:
                    print("❌ Failed to replace database file")
            else:
                print("❌ Failed to close database connections")
                
        elif choice == "5":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option. Please select 1-5.")

if __name__ == "__main__":
    main() 