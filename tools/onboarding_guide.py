"""
Onboarding guide tool for structured onboarding processes.
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.tools import Tool

from agents.state import OnboardingProgress

logger = logging.getLogger(__name__)

class OnboardingGuideTools:
    """
    Provides tools for guiding users through the onboarding process.
    """
    
    def __init__(self):
        """Initialize the onboarding guide tools."""
        # Default onboarding steps - could be loaded from a config file or database
        self.default_steps = [
            "Introduction to the company and team",
            "Setup development environment",
            "Access to systems and tools",
            "Code review process",
            "Project overview",
            "Team workflow and processes",
            "First task assignment",
        ]
        
        logger.info("Initialized onboarding guide tools")
    
    def create_onboarding_plan(self, user_id: str, user_name: str, role: str) -> OnboardingProgress:
        """
        Create a personalized onboarding plan for a new team member.
        
        Args:
            user_id: Unique identifier for the user
            user_name: Name of the user
            role: Role of the user in the team
        
        Returns:
            Onboarding progress object
        """
        try:
            # Create a new onboarding progress object
            progress = OnboardingProgress(
                user_id=user_id,
                user_name=user_name,
                role=role,
                start_date=datetime.now().isoformat(),
                completed_steps=[],
                remaining_steps=self.default_steps.copy(),
                current_step=self.default_steps[0] if self.default_steps else None,
            )
            return progress
            
        except Exception as e:
            logger.error(f"Error creating onboarding plan: {e}")
            return {
                "error": f"Failed to create onboarding plan: {str(e)}",
            }
    
    def get_current_step(self, onboarding_progress: Dict) -> Dict:
        """
        Get the current onboarding step with details.
        
        Args:
            onboarding_progress: Onboarding progress dictionary
        
        Returns:
            Current step details
        """
        try:
            progress = OnboardingProgress.model_validate(onboarding_progress)
            
            if not progress.current_step:
                return {
                    "status": "completed",
                    "message": "All onboarding steps have been completed.",
                    "next_steps": "Consider setting up regular check-ins with team members."
                }
            
            # Get step index and calculate progress percentage
            current_index = progress.remaining_steps.index(progress.current_step) if progress.current_step in progress.remaining_steps else 0
            total_steps = len(progress.completed_steps) + len(progress.remaining_steps)
            progress_percentage = (len(progress.completed_steps) / total_steps) * 100 if total_steps > 0 else 0
            
            return {
                "current_step": progress.current_step,
                "description": self._get_step_description(progress.current_step),
                "progress_percentage": round(progress_percentage, 1),
                "completed_steps": progress.completed_steps,
                "remaining_steps": progress.remaining_steps,
            }
        
        except Exception as e:
            logger.error(f"Error getting current step: {e}")
            return {
                "error": f"Failed to get current step: {str(e)}",
            }
    
    def complete_step(self, onboarding_progress: Dict) -> Dict:
        """
        Mark the current step as completed and move to the next step.
        
        Args:
            onboarding_progress: Onboarding progress dictionary
        
        Returns:
            Updated onboarding progress
        """
        try:
            progress = OnboardingProgress.model_validate(onboarding_progress)
            
            if not progress.current_step:
                return {
                    "status": "already_completed",
                    "message": "All onboarding steps have already been completed.",
                    "onboarding_progress": progress.model_dump()
                }
            
            # Move current step from remaining to completed
            if progress.current_step in progress.remaining_steps:
                progress.completed_steps.append(progress.current_step)
                progress.remaining_steps.remove(progress.current_step)
            
            # Set new current step
            if progress.remaining_steps:
                progress.current_step = progress.remaining_steps[0]
            else:
                progress.current_step = None
            
            return {
                "status": "success",
                "message": "Step completed successfully.",
                "next_step": progress.current_step,
                "onboarding_progress": progress.model_dump()
            }
        
        except Exception as e:
            logger.error(f"Error completing step: {e}")
            return {
                "error": f"Failed to complete step: {str(e)}",
            }
    
    def _get_step_description(self, step: str) -> str:
        """
        Get a detailed description for an onboarding step.
        
        Args:
            step: The step name
        
        Returns:
            Detailed description of the step
        """
        # This would typically come from a database or configuration file
        descriptions = {
            "Introduction to the company and team": (
                "Learn about the company's history, mission, values, and team structure. "
                "Schedule introductory meetings with key team members."
            ),
            "Setup development environment": (
                "Set up your local development environment including required software, "
                "tools, and access credentials. Follow the setup guide in the documentation."
            ),
            "Access to systems and tools": (
                "Gain access to all necessary systems, tools, and resources required for your role. "
                "This includes source code repositories, project management tools, and communication platforms."
            ),
            "Code review process": (
                "Learn about the team's code review process and standards. "
                "Understand how to submit code for review and how to review others' code."
            ),
            "Project overview": (
                "Get a comprehensive overview of the project, its architecture, "
                "and how different components work together."
            ),
            "Team workflow and processes": (
                "Understand the team's workflow, methodologies, meeting cadence, "
                "and communication protocols."
            ),
            "First task assignment": (
                "Receive your first task assignment. This will typically be a small, "
                "well-defined task to help you get familiar with the codebase and processes."
            ),
        }
        
        return descriptions.get(step, "No detailed description available for this step.")
    
    def get_tools(self) -> List[Tool]:
        """
        Get the list of onboarding guide tools.
        
        Returns:
            List of LangChain tools
        """
        return [
            Tool.from_function(
                func=self.create_onboarding_plan,
                name="create_onboarding_plan",
                description=(
                    "Create a personalized onboarding plan for a new team member. "
                    "Provide the user ID, name, and role."
                ),
                return_direct=False,
            ),
            Tool.from_function(
                func=self.get_current_step,
                name="get_current_step",
                description=(
                    "Get details about the current onboarding step. "
                    "Provide the onboarding progress object."
                ),
                return_direct=False,
            ),
            Tool.from_function(
                func=self.complete_step,
                name="complete_step",
                description=(
                    "Mark the current onboarding step as completed and move to the next step. "
                    "Provide the onboarding progress object."
                ),
                return_direct=False,
            ),
        ]