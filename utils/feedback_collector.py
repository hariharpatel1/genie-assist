"""
Feedback collection and processing utilities.
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

class FeedbackCollector:
    """
    Collects and processes user feedback about the onboarding agent.
    """
    
    def __init__(self, feedback_dir: Optional[str] = None):
        """
        Initialize the feedback collector.
        
        Args:
            feedback_dir: Directory to store feedback data. If None, uses a default path.
        """
        self.feedback_dir = feedback_dir or os.path.join(settings.PROJECT_ROOT, "feedback_data")
        os.makedirs(self.feedback_dir, exist_ok=True)
        
        logger.info(f"Initialized feedback collector with storage in {self.feedback_dir}")
    
    def record_feedback(
        self, 
        thread_id: str, 
        user_id: str, 
        rating: int, 
        comments: Optional[str] = None,
        knowledge_gap: Optional[Dict] = None
    ) -> bool:
        """
        Record user feedback.
        
        Args:
            thread_id: Conversation thread ID
            user_id: User identifier
            rating: Numerical rating (1-5)
            comments: Optional comments from the user
            knowledge_gap: Optional knowledge gap information
        
        Returns:
            True if feedback was successfully recorded, False otherwise
        """
        try:
            timestamp = datetime.now().isoformat()
            feedback_id = f"feedback_{int(datetime.now().timestamp())}"
            
            feedback_data = {
                "feedback_id": feedback_id,
                "thread_id": thread_id,
                "user_id": user_id,
                "rating": rating,
                "comments": comments,
                "knowledge_gap": knowledge_gap,
                "timestamp": timestamp,
            }
            
            # Save to file
            feedback_file = os.path.join(self.feedback_dir, f"{feedback_id}.json")
            with open(feedback_file, "w") as f:
                json.dump(feedback_data, f, indent=2)
            
            logger.info(f"Recorded feedback {feedback_id} with rating {rating}")
            return True
        
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict]:
        """
        Get recent feedback entries.
        
        Args:
            limit: Maximum number of entries to return
        
        Returns:
            List of feedback entries, sorted by recency (newest first)
        """
        try:
            feedback_files = [
                os.path.join(self.feedback_dir, f)
                for f in os.listdir(self.feedback_dir)
                if f.endswith(".json")
            ]
            
            feedback_entries = []
            for file_path in feedback_files:
                try:
                    with open(file_path, "r") as f:
                        feedback_entries.append(json.load(f))
                except Exception as e:
                    logger.warning(f"Error reading feedback file {file_path}: {e}")
            
            # Sort by timestamp (newest first)
            sorted_entries = sorted(
                feedback_entries, 
                key=lambda x: x.get("timestamp", ""), 
                reverse=True
            )
            
            return sorted_entries[:limit]
        
        except Exception as e:
            logger.error(f"Error getting recent feedback: {e}")
            return []
    
    def get_knowledge_gaps(self) -> List[Dict]:
        """
        Get reported knowledge gaps from feedback.
        
        Returns:
            List of knowledge gap entries
        """
        try:
            feedback_entries = self.get_recent_feedback(limit=100)
            
            # Filter entries with knowledge gaps
            knowledge_gaps = [
                entry.get("knowledge_gap")
                for entry in feedback_entries
                if entry.get("knowledge_gap") is not None
            ]
            
            return knowledge_gaps
        
        except Exception as e:
            logger.error(f"Error getting knowledge gaps: {e}")
            return []