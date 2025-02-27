"""
Utilities for setting up and configuring the LLM.
"""
import logging
from typing import Optional

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_azure import AzureChatOpenAI
from langchain_core.language_models import BaseChatModel

from config.settings import settings

logger = logging.getLogger(__name__)

def get_azure_openai_chat_model(
    temperature: float = 0.1,
    streaming: bool = True,
    callback_manager: Optional[CallbackManager] = None,
) -> BaseChatModel:
    """
    Get an instance of the Azure OpenAI chat model.
    
    Args:
        temperature: The temperature parameter for the model
        streaming: Whether to enable streaming of responses
        callback_manager: Optional callback manager for the model
    
    Returns:
        An instance of the Azure OpenAI chat model
    """
    try:
        callbacks = []
        if streaming:
            callbacks.append(StreamingStdOutCallbackHandler())
        
        if callback_manager:
            chat_model = AzureChatOpenAI(
                openai_api_key=settings.AZURE_OPENAI_API_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                temperature=temperature,
                callback_manager=callback_manager,
            )
        else:
            chat_model = AzureChatOpenAI(
                openai_api_key=settings.AZURE_OPENAI_API_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                temperature=temperature,
                callbacks=callbacks,
            )
        
        logger.info("Successfully initialized Azure OpenAI chat model")
        return chat_model
    
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI chat model: {e}")
        raise