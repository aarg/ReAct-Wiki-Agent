"""
Tools for the multi-tool agent.

This module defines various tools that the agent can use to answer questions.
Each tool has a name, description, and execute method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json
import re
from datetime import datetime, timedelta
import math
import os
from dateutil import parser
from .google_search import GoogleSearchTool

# Optional: Import langsmith for detailed tool tracking
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    # Create a no-op decorator if langsmith is not available
    def traceable(**kwargs):
        def decorator(func):
            return func
        return decorator
    LANGSMITH_AVAILABLE = False


class Tool(ABC):
    """Base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does and when to use it."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters."""
        pass


class CalculatorTool(Tool):
    """Tool for performing mathematical calculations."""
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Performs mathematical calculations. Use for arithmetic, algebra, and basic math operations. Input should be a valid mathematical expression."
    
    @traceable(name="calculator_tool", tags=["tool", "calculator"])
    def execute(self, expression: str) -> str:
        """
        Execute a mathematical expression safely.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result of the calculation as a string
        """
        try:
            # Remove any dangerous functions
            if any(bad in expression.lower() for bad in ['import', 'exec', 'eval', '__']):
                return "Error: Invalid expression"
            
            # Allow only safe math operations
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round})
            
            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: Could not calculate. {str(e)}"


class DateTimeTool(Tool):
    """Tool for getting current date and time information."""
    
    @property
    def name(self) -> str:
        return "datetime"
    
    @property
    def description(self) -> str:
        return "Gets current date, time, day of week, and other temporal information. Use when questions ask about 'today', 'current time', 'what day', etc."
    
    @traceable(name="datetime_tool", tags=["tool", "datetime"])
    def execute(self, format: str = None) -> str:
        """
        Get current date and time.
        
        Args:
            format: Optional format string for the output
            
        Returns:
            Current date and time information
        """
        now = datetime.now()
        
        if format:
            try:
                return now.strftime(format)
            except Exception as e:
                return f"Error: Invalid format string '{format}'. {str(e)}"
        
        # Return comprehensive time info
        return f"""Current date and time:
- Date: {now.strftime('%Y-%m-%d')}
- Time: {now.strftime('%H:%M:%S')}
- Day: {now.strftime('%A')}
- Month: {now.strftime('%B')}
- Year: {now.year}
- Timezone: Local system time"""


class DateCalculatorTool(Tool):
    """Tool for calculating future or past dates."""
    
    @property
    def name(self) -> str:
        return "datecalculator"
    
    @property
    def description(self) -> str:
        return "Calculates dates and days between dates. Use for: 1) 'what day will it be in X days' (provide days=X), 2) 'how many days until [date]' (provide target_date='date string'). Accepts either days (integer) OR target_date (date string like 'November 27, 2025' or '2025-11-27')."
    
    @traceable(name="datecalculator_tool", tags=["tool", "datecalculator"])
    def execute(self, days: int = None, target_date: str = None) -> str:
        """
        Calculate a date relative to today or days until a target date.
        
        Args:
            days: Number of days to add (positive) or subtract (negative) from today
            target_date: Target date string to calculate days until
            
        Returns:
            Formatted string with date calculation results
        """
        try:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Mode 1: Calculate days until a target date
            if target_date is not None:
                try:
                    # Parse the target date string
                    target = parser.parse(target_date)
                    target = target.replace(hour=0, minute=0, second=0, microsecond=0)
                    
                    # Calculate days difference
                    delta = target - today
                    days_diff = delta.days
                    
                    if days_diff == 0:
                        return f"The target date {target.strftime('%B %d, %Y')} is today!"
                    elif days_diff > 0:
                        return f"""Days until {target.strftime('%B %d, %Y')}:
- Today: {today.strftime('%A, %B %d, %Y')}
- Target: {target.strftime('%A, %B %d, %Y')}
- Days until: {days_diff} days"""
                    else:
                        return f"""Days since {target.strftime('%B %d, %Y')}:
- Today: {today.strftime('%A, %B %d, %Y')}
- Target: {target.strftime('%A, %B %d, %Y')}
- Days since: {abs(days_diff)} days ago"""
                except Exception as e:
                    return f"Error parsing date '{target_date}'. Please use formats like 'November 27, 2025' or '2025-11-27'. Error: {str(e)}"
            
            # Mode 2: Calculate date X days from today (original behavior)
            elif days is not None:
                days = int(days)
                target_date = today + timedelta(days=days)
                
                if days == 0:
                    return f"Today is {today.strftime('%A, %B %d, %Y')}"
                elif days > 0:
                    return f"""Date calculation:
- Today: {today.strftime('%A, %B %d, %Y')}
- {days} days from today: {target_date.strftime('%A, %B %d, %Y')}
- That will be a {target_date.strftime('%A')}"""
                else:
                    return f"""Date calculation:
- Today: {today.strftime('%A, %B %d, %Y')}
- {abs(days)} days ago: {target_date.strftime('%A, %B %d, %Y')}
- That was a {target_date.strftime('%A')}"""
            else:
                return f"Today is {today.strftime('%A, %B %d, %Y')}"
                
        except (ValueError, TypeError) as e:
            return f"Error: Invalid input. {str(e)}"
        except Exception as e:
            return f"Error calculating date: {str(e)}"


class WebSearchTool(Tool):
    """Tool for searching the web using Google."""
    
    def __init__(self):
        self.search_tool = GoogleSearchTool()
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Searches the web for current information. Use for facts, news, general knowledge, and any questions requiring up-to-date information from the internet."
    
    @traceable(name="web_search_tool", tags=["tool", "web_search"])
    def execute(self, query: str) -> str:
        """
        Search the web for information.
        
        Args:
            query: Search query
            
        Returns:
            Search results and extracted content
        """
        try:
            result = self.search_tool.search_and_extract(query, max_length=3000)
            return f"Found information from {result['title']}:\n\n{result['content']}"
        except Exception as e:
            return f"Error searching web: {str(e)}"


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register the default set of tools."""
        self.register(CalculatorTool())
        self.register(DateTimeTool())
        self.register(DateCalculatorTool())
        self.register(WebSearchTool())
    
    def register(self, tool: Tool):
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Tool:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools with their descriptions."""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools.values()
        ]
    
    def get_tools_description(self) -> str:
        """Get a formatted description of all tools for the LLM."""
        tools_desc = []
        for tool in self.tools.values():
            tools_desc.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tools_desc)