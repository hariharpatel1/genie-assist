"""
Main application entry point for the Onboarding Agent.
"""
import logging
import os
from datetime import datetime

import streamlit as st

from cfg.settings import settings
from ui.main_page import render_main_page

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"onboarding_agent_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    try:
        logger.info("Starting onboarding agent application")
        render_main_page()
    except Exception as e:
        logger.error(f"Error running application: {e}", exc_info=True)
        st.error("An error occurred. Please check the logs or contact support.")

if __name__ == "__main__":
    main()