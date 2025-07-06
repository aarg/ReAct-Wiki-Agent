"""
State definition for the Multi-Tool Agent.

This module defines the AgentState TypedDict that represents the data
flowing through the LangGraph workflow.
"""

from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict):
    """
    State object for the multi-tool agent workflow.
    
    This state flows through the LangGraph nodes and maintains
    all necessary information for the ReAct pattern.
    
    Attributes:
        question: The user's input question
        thought: The agent's reasoning about what to do next
        tool_name: The name of the tool to use
        tool_input: The input parameters for the tool
        observation: The output from the tool execution
        final_answer: The final synthesized answer
        steps: History of all reasoning steps taken
        error: Any error message encountered
    """
    # User input
    question: str
    
    # ReAct pattern fields
    thought: Optional[str]
    tool_name: Optional[str]
    tool_input: Optional[Dict[str, Any]]
    observation: Optional[str]
    
    # Results
    final_answer: Optional[str]
    
    # History and debugging
    steps: Optional[List[Dict[str, str]]]
    error: Optional[str]