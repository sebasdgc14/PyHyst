import logging

# Set up logging configuration
logging.basicConfig(
    filename='pyhyst.log',
    filemode='a',  # Append mode
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Optional: Get a logger object for use in modules
logger = logging.getLogger(__name__)