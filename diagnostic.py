import sys
import os
import importlib
import logging
from pathlib import Path

def check_python_version():
    required_version = (3, 7)
    current_version = sys.version_info
    return current_version >= required_version

def check_dependencies():
    required_packages = [
        'python-binance',
        'pandas',
        'numpy',
        'python-dotenv'
    ]
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    return missing_packages

def check_config_files():
    required_files = [
        '.env',
        'config.py',
        'main.py'
    ]
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    return missing_files

def run_diagnostics():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('diagnostics')
    
    # Check Python version
    if not check_python_version():
        logger.error("Python version must be 3.7 or higher")
        return False
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install using: pip install " + ' '.join(missing_packages))
        return False
    
    # Check config files
    missing_files = check_config_files()
    if missing_files:
        logger.error(f"Missing configuration files: {', '.join(missing_files)}")
        return False
    
    logger.info("All diagnostic checks passed successfully")
    return True

if __name__ == "__main__":
    run_diagnostics()
