"""
Configuration settings for the onboarding agent.
"""
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseSettings, Field

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Settings for the onboarding agent."""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
    
    # Azure OpenAI Settings
    AZURE_OPENAI_API_KEY: str = Field(default=os.getenv("AZURE_OPENAI_API_KEY", ""))
    AZURE_OPENAI_ENDPOINT: str = Field(default=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    AZURE_OPENAI_DEPLOYMENT_NAME: str = Field(default=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""))
    AZURE_OPENAI_API_VERSION: str = Field(default=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"))
    
    # GitHub Settings
    GITHUB_ACCESS_TOKEN: str = Field(default=os.getenv("GITHUB_ACCESS_TOKEN", ""))
    GITHUB_REPOS: List[str] = Field(default_factory=lambda: 
        os.getenv("GITHUB_REPOS", "").split(",") if os.getenv("GITHUB_REPOS") else [])
    
    # Google Cloud Settings
    GOOGLE_APPLICATION_CREDENTIALS: str = Field(default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))
    GOOGLE_DOCS_IDS: List[str] = Field(default_factory=lambda: 
        os.getenv("GOOGLE_DOCS_IDS", "").split(",") if os.getenv("GOOGLE_DOCS_IDS") else [])
    
    # Vector Database Settings
    CHROMA_PERSIST_DIRECTORY: str = Field(default=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"))
    
    # Human Escalation Settings
    SLACK_BOT_TOKEN: Optional[str] = Field(default=os.getenv("SLACK_BOT_TOKEN", ""))
    SLACK_CHANNEL_ID: Optional[str] = Field(default=os.getenv("SLACK_CHANNEL_ID", ""))
    
    # Logging Settings
    LOG_LEVEL: str = Field(default=os.getenv("LOG_LEVEL", "INFO"))

    class Config:
        """Pydantic config"""
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()