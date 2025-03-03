""" Utilities for setting up and configuring the LLM. """
import logging
import os
from typing import Optional

from langchain_core.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import AzureChatOpenAI
from langchain_core.language_models import BaseChatModel

from cfg.settings import settings

logger = logging.getLogger(__name__)

def get_azure_openai_llm(streaming: bool = False) -> BaseChatModel:
    """
    Get an instance of the Azure OpenAI chat model.
    
    Args:
        streaming (bool): Whether to enable streaming for the model.
    
    Returns:
        An instance of the Azure OpenAI chat model
    """
    try:
        # Debug log the connection details (without the API key)
        logger.info(f"Connecting to Azure OpenAI with endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
        logger.info(f"Using deployment: {settings.AZURE_OPENAI_DEPLOYMENT_NAME}")
        logger.info(f"API version: {settings.AZURE_OPENAI_API_VERSION}")
        
        # Check if API key is set and not empty
        if not settings.AZURE_OPENAI_API_KEY or settings.AZURE_OPENAI_API_KEY.strip() == "":
            logger.error("Azure OpenAI API key is empty or not set")
            raise ValueError("Azure OpenAI API key is empty or not set")
            
        # Check if endpoint is properly formatted
        if not settings.AZURE_OPENAI_ENDPOINT.startswith('https://'):
            logger.warning(f"Azure endpoint doesn't start with https:// - current value: {settings.AZURE_OPENAI_ENDPOINT}")
            
        callbacks = None
        if streaming:
            callbacks = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Set more explicit parameters for troubleshooting
        llm = AzureChatOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.5,
            request_timeout=60,  # Increase timeout for troubleshooting
            max_retries=3
        )
        
        # Test the connection with a simple request
        test_result = llm.invoke("Hello, please respond with 'Connection test successful'")
        logger.info(f"Connection test result: {test_result.content[:50]}...")
        
        logger.info("Successfully initialized Azure OpenAI chat model")
        return llm
    except Exception as e:
        logger.error(f"Error initializing Azure OpenAI: {str(e)}", exc_info=True)
        
        # Additional debugging information
        logger.error("Environment and configuration details:")
        try:
            # Log redacted versions of credentials for debugging
            api_key_redacted = settings.AZURE_OPENAI_API_KEY[:4] + "..." if settings.AZURE_OPENAI_API_KEY else "Not set"
            logger.error(f"  API Key (redacted): {api_key_redacted}")
            logger.error(f"  Endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
            logger.error(f"  Deployment: {settings.AZURE_OPENAI_DEPLOYMENT_NAME}")
            logger.error(f"  API Version: {settings.AZURE_OPENAI_API_VERSION}")
        except Exception as config_error:
            logger.error(f"Error while logging configuration details: {config_error}")
            
        raise RuntimeError(f"Failed to initialize Azure OpenAI: {str(e)}")

# Initialize the Azure OpenAI model
try:
    azure_llm = get_azure_openai_llm()
except Exception as e:
    logger.critical(f"Critical failure initializing Azure OpenAI: {e}")
    raise e