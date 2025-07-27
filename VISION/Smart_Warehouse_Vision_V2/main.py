"""
Smart Warehouse CV System - Main Entry Point
This is the main application file that orchestrates all components.
"""

import torch
# Suppress Streamlit watcher error on torch.classes
torch.classes.__path__ = []

from ui.streamlit_ui import StreamlitUI
from core.logging_config import setup_logging


def main():
    """Main application entry point."""
    # Set up logging
    setup_logging()
    
    # Initialize and render the Streamlit UI
    ui = StreamlitUI()
    ui.render()


if __name__ == "__main__":
    main()