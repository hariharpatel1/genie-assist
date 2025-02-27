"""
Human escalation tool for the onboarding agent.
"""
import json
import logging
from typing import Dict, List, Optional

from langchain_core.tools import Tool
from langgraph.types import Command, interrupt

from cfg.settings import settings

logger = logging.getLogger(__name__)

class HumanEscalationTools:
    """
    Provides tools for escalating queries to human experts.
    """
    
    def __init__(self, slack_token: Optional[str] = None, slack_channel: Optional[str] = None):
        """
        Initialize the human escalation tools.
        
        Args:
            slack_token: Optional Slack bot token for notifications
            slack_channel: Optional Slack channel ID for notifications
        """
        self.slack_token = slack_token or settings.SLACK_BOT_TOKEN
        self.slack_channel = slack_channel or settings.SLACK_CHANNEL_ID
        self.has_slack = bool(self.slack_token and self.slack_channel)
        
        logger.info(f"Initialized human escalation tools (Slack integration: {self.has_slack})")
    
    def _send_slack_notification(self, message: str) -> bool:
        """
        Send a notification to Slack.
        
        Args:
            message: Message to send
        
        Returns:
            True if successful, False otherwise
        """
        if not self.has_slack:
            logger.warning("Slack integration not configured, skipping notification")
            return False
        
        try:
            # We're using a simple approach here, but you could use a proper Slack client library
            import requests
            
            response = requests.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {self.slack_token}"},
                json={"channel": self.slack_channel, "text": message}
            )
            
            if response.status_code == 200 and response.json().get("ok"):
                logger.info(f"Successfully sent Slack notification to channel {self.slack_channel}")
                return True
            else:
                logger.error(f"Failed to send Slack notification: {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    def escalate_to_human(self, query: str, context: str) -> Dict:
        """
        Escalate a query to a human expert.
        Uses LangGraph's interrupt to pause execution and wait for human input.
        
        Args:
            query: The query to escalate
            context: Additional context about the query
        
        Returns:
            Human response
        """
        try:
            # Notify via Slack if configured
            if self.has_slack:
                notification = (
                    f"*New question requiring human assistance:*\n"
                    f">Query: {query}\n"
                    f">Context: {context}\n"
                    f"Please check the onboarding assistant to provide your answer."
                )
                self._send_slack_notification(notification)
            
            # Use LangGraph's interrupt to pause execution and wait for human input
            human_response = interrupt({
                "query": query,
                "context": context,
                "require_human": True,
            })
            
            # Process and return the human response
            return {
                "response": human_response.get("response", "No response provided"),
                "expert": human_response.get("expert", "Unknown expert"),
                "escalated": True,
            }
        
        except Exception as e:
            logger.error(f"Error during human escalation: {e}")
            return {
                "response": f"There was an error processing this escalation: {str(e)}",
                "expert": "System",
                "escalated": False,
            }
    
    def get_tools(self) -> List[Tool]:
        """
        Get the list of human escalation tools.
        
        Returns:
            List of LangChain tools
        """
        return [
            Tool.from_function(
                func=self.escalate_to_human,
                name="escalate_to_human",
                description=(
                    "Escalate a query to a human expert when you cannot confidently answer it. "
                    "Provide the original query and any relevant context you have gathered."
                ),
                return_direct=False,
            ),
        ]