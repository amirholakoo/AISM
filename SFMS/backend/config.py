# config.py

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Main application database
DATABASE_CONFIG = {
    'uri': 'sqlite:///db.sqlite3',
    'track_modifications': False
}

# External database configuration (SFMS database)
EXTERNAL_DATABASE_CONFIG = {
    'url': 'sqlite:///external_db/localnew.sqlite3',
    'directory': 'external_db/',
    'filename': 'localnew.sqlite3'
}

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

SECRET_KEY = "your_super_secret_key_here_change_in_production"

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

# Edit window configuration (minutes)
EDIT_WINDOW_MINUTES = 20

# =============================================================================
# SSH CONFIGURATION
# =============================================================================

# SSH Connection Configuration
SSH_CONFIG = {
    'hostname': '172.20.97.9',
    'username': 'root',
    'password': 'ahmadi',
    'timeout': 30,
    'port': 22,
    'key_filename': None,  # Path to SSH key file (optional)
    'allow_agent': False,  # Allow SSH agent authentication
    'look_for_keys': False  # Look for keys in ~/.ssh/
}

# SSH File Operations Configuration
SSH_FILE_CONFIG = {
    'remote_filename': 'localnew.sqlite3',
    'remote_path': '/root/',  # Remote directory path
    'local_path': 'external_db/',
    'backup_enabled': True,
    'backup_suffix': '.backup',
    'backup_directory': 'external_db/backups/',  # Separate backup directory
    'max_backups': 10,  # Maximum number of backups to keep
    'compression_enabled': False,  # Enable file compression during transfer
    'verify_checksum': True  # Verify file integrity after transfer
}

# SSH Commands Configuration
SSH_COMMANDS = {
    'list_files': 'ls -la',
    'check_file_exists': 'test -f',
    'get_file_size': 'stat -c %s',
    'get_file_checksum': 'md5sum'
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/ssh_operations.log',
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

RETRY_CONFIG = {
    'max_attempts': 3,
    'initial_delay': 1,  # seconds
    'max_delay': 30,  # seconds
    'backoff_factor': 2
}

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

PERFORMANCE_CONFIG = {
    'chunk_size': 8192,  # File transfer chunk size
    'buffer_size': 32768,  # SFTP buffer size
    'connection_pool_size': 5
}

# =============================================================================
# COMPATIBILITY VARIABLES (for backward compatibility)
# =============================================================================

# Legacy database configuration (for existing code)
SQLALCHEMY_DATABASE_URI = DATABASE_CONFIG['uri']
SQLALCHEMY_TRACK_MODIFICATIONS = DATABASE_CONFIG['track_modifications']

# Legacy external database configuration
EXTERNAL_DATABASE_URL = EXTERNAL_DATABASE_CONFIG['url']
