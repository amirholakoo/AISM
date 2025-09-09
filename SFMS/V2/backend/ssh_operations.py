import paramiko
import os
import shutil
from datetime import datetime
import logging
from config import SSH_CONFIG, SSH_FILE_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SSHOperations:
    def __init__(self, hostname=None, username=None, password=None, timeout=None, port=None):
        # Use config values if not provided
        self.hostname = hostname or SSH_CONFIG['hostname']
        self.username = username or SSH_CONFIG['username']
        self.password = password or SSH_CONFIG['password']
        self.timeout = timeout or SSH_CONFIG['timeout']
        self.port = port or SSH_CONFIG['port']
        self.ssh_client = None
        self.sftp_client = None
    
    def connect(self):
        """Establish SSH connection"""
        try:
            logger.info(f"🔗 Connecting to {self.hostname}:{self.port}...")
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(
                hostname=self.hostname,
                port=self.port,
                username=self.username,
                password=self.password,
                timeout=self.timeout
            )
            logger.info("✅ SSH connection established successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ SSH connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close SSH connection"""
        try:
            if self.sftp_client:
                self.sftp_client.close()
            if self.ssh_client:
                self.ssh_client.close()
            logger.info("🔌 SSH connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing SSH connection: {e}")
    
    def test_connection(self):
        """Test SSH connection and list home directory"""
        try:
            if not self.connect():
                return False
            
            logger.info("🔍 Testing connection and listing home directory...")
            stdin, stdout, stderr = self.ssh_client.exec_command('ls -la ~')
            result = stdout.read().decode()
            error = stderr.read().decode()
            
            if error:
                logger.error(f"❌ Command error: {error}")
                return False
            
            logger.info("✅ Connection test successful!")
            logger.info("📁 Home directory contents:")
            logger.info(result)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
        finally:
            self.disconnect()
    
    def copy_database_file(self, remote_filename=None, local_path=None):
        """
        Copy database file from remote server to local machine
        
        Args:
            remote_filename: Name of the file in remote home directory (optional)
            local_path: Local directory to save the file (optional)
        """
        try:
            # Use config values if not provided
            remote_filename = remote_filename or SSH_FILE_CONFIG['remote_filename']
            local_path = local_path or SSH_FILE_CONFIG['local_path']
            
            if not self.connect():
                return False
            
            # Create local directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)
            
            # Setup file paths
            remote_path = SSH_FILE_CONFIG['remote_path']
            remote_file = os.path.join(remote_path, remote_filename).replace('\\', '/')
            local_file = os.path.join(local_path, remote_filename)
            
            # Create backup of existing file if enabled
            if SSH_FILE_CONFIG['backup_enabled'] and os.path.exists(local_file):
                backup_directory = SSH_FILE_CONFIG['backup_directory']
                os.makedirs(backup_directory, exist_ok=True)
                
                backup_suffix = SSH_FILE_CONFIG['backup_suffix']
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_filename = f"{remote_filename}{backup_suffix}.{timestamp}"
                backup_file = os.path.join(backup_directory, backup_filename)
                
                shutil.copy2(local_file, backup_file)
                logger.info(f"📦 Backup created: {backup_file}")
                
                # Clean old backups if max_backups is set
                self._cleanup_old_backups(backup_directory, remote_filename)
            
            # Create SFTP client
            self.sftp_client = self.ssh_client.open_sftp()
            
            logger.info(f"📁 Copying file from {remote_file} to {local_file}...")
            
            # Copy file
            self.sftp_client.get(remote_file, local_file)
            
            # Verify file was copied
            if os.path.exists(local_file):
                file_size = os.path.getsize(local_file)
                logger.info(f"✅ Database file copied successfully! Size: {file_size} bytes")
                return True
            else:
                logger.error("❌ File copy failed - local file not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error copying database file: {e}")
            return False
        finally:
            self.disconnect()
    
    def list_remote_files(self, directory="~"):
        """List files in remote directory"""
        try:
            if not self.connect():
                return False
            
            logger.info(f"📁 Listing files in {directory}...")
            stdin, stdout, stderr = self.ssh_client.exec_command(f'ls -la {directory}')
            result = stdout.read().decode()
            error = stderr.read().decode()
            
            if error:
                logger.error(f"❌ Command error: {error}")
                return False
            
            logger.info("📋 Remote directory contents:")
            logger.info(result)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error listing remote files: {e}")
            return False
        finally:
            self.disconnect()
    
    def _cleanup_old_backups(self, backup_directory, filename_prefix):
        """Clean up old backup files, keeping only the most recent ones"""
        try:
            from config import SSH_FILE_CONFIG
            
            max_backups = SSH_FILE_CONFIG.get('max_backups', 10)
            if max_backups <= 0:
                return
            
            # Find all backup files for this database
            backup_pattern = f"{filename_prefix}{SSH_FILE_CONFIG['backup_suffix']}.*"
            import glob
            backup_files = glob.glob(os.path.join(backup_directory, backup_pattern))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Remove old backups
            if len(backup_files) > max_backups:
                for old_backup in backup_files[max_backups:]:
                    try:
                        os.remove(old_backup)
                        logger.info(f"🗑️ Removed old backup: {old_backup}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not remove old backup {old_backup}: {e}")
                        
        except Exception as e:
            logger.warning(f"⚠️ Error during backup cleanup: {e}")

# Convenience functions
def test_ssh_connection():
    """Test SSH connection to remote server"""
    ssh_ops = SSHOperations()
    return ssh_ops.test_connection()

def copy_database_via_ssh(remote_filename=None):
    """Copy database file from remote server"""
    ssh_ops = SSHOperations()
    return ssh_ops.copy_database_file(remote_filename)

def list_remote_database_files():
    """List database files in remote home directory"""
    ssh_ops = SSHOperations()
    return ssh_ops.list_remote_files()

if __name__ == "__main__":
    # Test the functions
    print("🧪 Testing SSH operations...")
    print(f"📋 SSH Config: {SSH_CONFIG}")
    print(f"📁 File Config: {SSH_FILE_CONFIG}")
    
    # Test connection
    print("\n1. Testing SSH connection...")
    if test_ssh_connection():
        print("✅ SSH connection test passed!")
    else:
        print("❌ SSH connection test failed!")
    
    # List remote files
    print("\n2. Listing remote files...")
    if list_remote_database_files():
        print("✅ File listing successful!")
    else:
        print("❌ File listing failed!")
    
    # Copy database file
    print("\n3. Copying database file...")
    if copy_database_via_ssh():
        print("✅ Database file copy successful!")
    else:
        print("❌ Database file copy failed!")