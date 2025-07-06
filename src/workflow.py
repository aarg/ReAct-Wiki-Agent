"""
LangGraph workflow for the Multi-Tool Agent.

This module implements a ReAct (Reasoning and Acting) pattern where the agent:
1. Thinks about what tool to use
2. Calls the appropriate tool
3. Observes the result
4. Decides if more tools are needed or if it can provide a final answer
"""

import re
import os
from datetime import datetime
from typing import Dict, Any, Literal, Optional
from langgraph.graph import StateGraph, END

from .state import AgentState
from .claude_client import ClaudeClient
from .tools import ToolRegistry

# Optional: Import langsmith for better tracing
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


# Helper functions
def extract_between(text: str, start_marker: str, end_marker: Optional[str] = None) -> str:
    """Extract text between markers."""
    if start_marker not in text:
        return ""
    
    start = text.find(start_marker) + len(start_marker)
    if end_marker:
        end = text.find(end_marker, start)
        if end == -1:
            end = len(text)
    else:
        # When no end marker, extract all remaining text
        end = len(text)
    
    return text[start:end].strip()


def clean_input_text(input_text: str) -> str:
    """Remove quotes and clean up input text."""
    # Remove outer quotes
    if (input_text.startswith("'") and input_text.endswith("'")) or \
       (input_text.startswith('"') and input_text.endswith('"')):
        input_text = input_text[1:-1]
    return input_text.strip()


def parse_tool_input(tool_name: str, raw_input: str) -> Dict[str, Any]:
    """Parse tool input based on tool type."""
    input_text = clean_input_text(raw_input)
    
    # Tool-specific parsing
    if tool_name == 'calculator':
        return {'expression': input_text}
    
    elif tool_name == 'datetime':
        if input_text and input_text.lower() != 'none':
            return {'format': input_text}
        return {}
    
    elif tool_name == 'web_search':
        # Handle nested dict patterns like {'query': 'actual query'}
        dict_pattern = r"\{['\"]query['\"]:\s*['\"](.+?)['\"]\}"
        match = re.search(dict_pattern, input_text)
        if match:
            input_text = match.group(1)
        return {'query': input_text}
    
    elif tool_name == 'datecalculator':
        # Handle nested dict patterns like {'target_date': 'date string'}
        dict_pattern = r"\{['\"]target_date['\"]:\s*['\"](.+?)['\"]\}"
        match = re.search(dict_pattern, input_text)
        if match:
            input_text = match.group(1)
        
        # Check if it's a number (days) or a date string
        if re.match(r'^-?\d+$', input_text):
            return {'days': int(input_text)}
        else:
            return {'target_date': input_text}
    
    return {}


def parse_tool_input_from_question(tool_name: str, question: str) -> Dict[str, Any]:
    """Extract tool input from the original question when no explicit input provided."""
    if tool_name == 'datecalculator':
        numbers = re.findall(r'\d+', question)
        # Check if question is asking about days until something
        if 'until' in question.lower() or 'days to' in question.lower():
            return {}  # Let the tool handle empty input for target dates
        elif numbers:
            return {'days': int(numbers[0])}
        return {'days': 0}
    return {}


def should_force_final_answer(steps: list) -> tuple[bool, str]:
    """Check if we should force a final answer based on execution history."""
    if not steps:
        return False, ""
    
    # Check if last tool provided complete answer
    if len(steps) >= 1:
        last_step = steps[-1]
        last_observation = last_step.get('observation', '')
        
        complete_answer_patterns = [
            ('Date calculation:' in last_observation and 'will be a' in last_observation),
            ('Date calculation:' in last_observation and 'was a' in last_observation),
            ('Days until' in last_observation or 'Days since' in last_observation),
            ('Result:' in last_observation and last_step.get('tool') == 'calculator'),
            ('Current date and time:' in last_observation)
        ]
        
        if any(complete_answer_patterns):
            return True, "complete_answer"
    
    # Check for loop pattern
    if len(steps) >= 3:
        recent_tools = [s.get('tool') for s in steps[-3:]]
        recent_inputs = [str(s.get('tool_input')) for s in steps[-3:]]
        if len(set(recent_tools)) == 1 and len(set(recent_inputs)) == 1:
            return True, "loop_detected"
    
    # Check for max steps
    if len(steps) >= 10:
        return True, "max_steps"
    
    return False, ""


def print_tool_output(observation: str, max_lines: int = 5) -> None:
    """Print tool output with proper formatting."""
    lines = observation.split('\n')
    if len(lines) > 1:
        print("   Output:")
        for line in lines[:max_lines]:
            if line.strip():
                print(f"     {line}")
        if len(lines) > max_lines:
            print(f"     ... ({len(lines) - max_lines} more lines)")
    else:
        print(f"   Output: {observation[:100]}{'...' if len(observation) > 100 else ''}")


def reasoning_node(state: AgentState) -> Dict[str, Any]:
    """
    Reasoning node that decides what tool to use next.
    
    This node uses Claude to analyze the question and current observations
    to decide which tool to call next, or if enough information has been
    gathered to provide a final answer.
    """
    claude = ClaudeClient()
    registry = ToolRegistry()
    
    # Print header for first step only
    if not state.get('steps'):
        print("\n" + "="*60)
        print("🤔 STARTING REASONING")
        print("="*60)
        print(f"\nQuestion: {state['question']}")
        print("\nAvailable tools:")
        for tool_info in registry.list_tools():
            print(f"  - {tool_info['name']}: {tool_info['description'][:60]}...")
    
    # Build context from previous steps
    context = f"Question: {state['question']}\n\n"
    
    if state.get('steps'):
        context += "Previous steps:\n"
        for step in state['steps']:
            if step.get('thought'):
                context += f"Thought: {step['thought']}\n"
            if step.get('tool'):
                context += f"Tool used: {step['tool']} with input: {step.get('tool_input', 'N/A')}\n"
            if step.get('observation'):
                context += f"Observation: {step['observation']}\n"
            context += "\n"
        
        # Check if we should force final answer
        force_answer, reason = should_force_final_answer(state['steps'])
        if force_answer:
            if reason == "complete_answer":
                print("\n✨ Previous tool provided complete answer")
                context += "\n\nIMPORTANT: The previous tool call has already provided the complete answer to the question. You should now provide your FINAL ANSWER based on this information. Do NOT call the same tool again.\n\n"
            elif reason == "loop_detected":
                print("\n⚠️  Detected repeated tool usage pattern")
                context += "\n\nCRITICAL: STOP USING TOOLS! You are stuck in a loop. The same tool with the same input has been used multiple times. You MUST provide a FINAL ANSWER NOW. Do NOT use any more tools.\n\n"
            elif reason == "max_steps":
                print("\n⚠️  Maximum steps reached")
                context += "\n\nCRITICAL: MAXIMUM STEPS REACHED! You MUST provide a FINAL ANSWER NOW. Do NOT use any more tools.\n\n"
    
    # Create reasoning prompt
    tools_desc = registry.get_tools_description()
    
    reasoning_prompt = f"""{context}

Available tools:
{tools_desc}

Based on the question and any previous observations, reason about what to do next.

IMPORTANT: Check if you already have the complete answer from previous tool calls!
- If a tool has already provided the exact answer needed, provide your FINAL ANSWER immediately
- Do NOT call the same tool again with the same input
- Only use another tool if you need different or additional information

You can either:
1. Use a tool to gather more information (ONLY if needed)
2. Provide a final answer if you have enough information

Think step by step about what information is needed to answer the question.

NOTE: For weather questions without a specific location, you should acknowledge that you cannot provide current weather without knowing the location and provide a final answer explaining this limitation.

IMPORTANT: You MUST use tools to gather information. Do NOT try to answer without using tools first.

If you need to use a tool, respond in this format:
THOUGHT: [Your reasoning about what to do next]
TOOL: [tool_name]
INPUT: [For calculator: the expression, for web_search: the query, for datecalculator: just the number of days OR the target date string, for datetime: leave empty or specify format]

Only after you have used tools and gathered information, you can provide a final answer:
THOUGHT: [Your reasoning about why you can now answer]
FINAL ANSWER: [Your complete answer to the question based on the tool observations]

Rules:
- For ANY math calculation, you MUST use the calculator tool
- For current date/time questions (e.g., "what day is today"), use the datetime tool
- For date calculations (e.g., "what day will it be in X days" or "how many days until [date]"), use the datecalculator tool
- For ANY factual question, you MUST use the web_search tool
- NEVER provide an answer without first using the appropriate tool
- If a tool returns an error or unhelpful result after 2-3 attempts, provide the best answer you can based on available information
- If you see "Error searching web" multiple times, acknowledge the limitation and provide a final answer
- Weather questions require location - if no location is specified, explain this in your final answer
"""
    
    # Get Claude's reasoning
    if not state.get('steps'):
        print("\nThinking...")
    
    reasoning_text = claude.generate_reasoning(
        prompt=reasoning_prompt,
        max_tokens=500
    )
    
    # Initialize updates
    updates = {
        'thought': "",
        'tool_name': None,
        'tool_input': None,
        'final_answer': None
    }
    
    # Extract thought
    thought = extract_between(reasoning_text, "THOUGHT:")
    # Extract only the first line for thought display
    if thought and '\n' in thought:
        thought_display = thought.split('\n')[0].strip()
    else:
        thought_display = thought
    if thought:
        updates['thought'] = thought  # Store full thought
        print(f"\n💭 {thought_display}")  # Display only first line
    
    # Check for tool usage (prioritize over final answer)
    if "TOOL:" in reasoning_text and "FINAL ANSWER:" not in reasoning_text[:reasoning_text.find("TOOL:")]:
        tool_name = extract_between(reasoning_text, "TOOL:")
        # Extract only the first line for tool name
        if tool_name and '\n' in tool_name:
            tool_name = tool_name.split('\n')[0].strip()
        if tool_name:
            updates['tool_name'] = tool_name
            
            # Extract input if provided
            if "INPUT:" in reasoning_text:
                # Find INPUT section bounds
                input_start = reasoning_text.find("INPUT:") + 6
                input_end = len(reasoning_text)
                
                # Check for other markers after INPUT
                for marker in ["THOUGHT:", "FINAL ANSWER:", "TOOL:"]:
                    marker_pos = reasoning_text.find(marker, input_start)
                    if marker_pos != -1 and marker_pos < input_end:
                        input_end = marker_pos
                
                raw_input = reasoning_text[input_start:input_end].strip()
                updates['tool_input'] = parse_tool_input(tool_name, raw_input)
            else:
                # No explicit input - try to extract from question
                updates['tool_input'] = parse_tool_input_from_question(tool_name, state.get('question', ''))
    
    # Check for final answer
    elif "FINAL ANSWER:" in reasoning_text:
        answer = extract_between(reasoning_text, "FINAL ANSWER:", None)
        if answer:
            updates['final_answer'] = answer
    
    return updates


def tool_execution_node(state: AgentState) -> Dict[str, Any]:
    """Execute the selected tool with the given input."""
    registry = ToolRegistry()
    
    if not state.get('tool_name'):
        return {'error': 'No tool selected'}
    
    tool = registry.get_tool(state['tool_name'])
    if not tool:
        return {'error': f"Tool '{state['tool_name']}' not found"}
    
    try:
        tool_input = state.get('tool_input', {})
        
        # Print tool execution info
        print(f"\n🔧 Tool: {state['tool_name']}")
        print(f"   Input: {tool_input}")
        
        # Execute tool
        observation = tool.execute(**tool_input)
        
        # Print formatted output
        print_tool_output(observation)
        
        # Update steps history
        current_steps = state.get('steps', [])
        current_steps.append({
            'thought': state.get('thought', ''),
            'tool': state['tool_name'],
            'tool_input': str(tool_input),
            'observation': observation
        })
        
        return {
            'observation': observation,
            'steps': current_steps
        }
    except Exception as e:
        print(f"\n❌ Tool error: {str(e)}")
        return {'error': f"Tool execution error: {str(e)}"}


@traceable(
    name="route_after_reasoning",
    tags=["routing", "decision", "react"],
    metadata={"description": "Decides whether to execute a tool or provide final answer"}
)
def route_after_reasoning(state: AgentState) -> Literal["execute_tool", "provide_answer"]:
    """
    Route the workflow after reasoning step.
    
    Examines the agent state to determine the next action:
    - If a final answer is ready → route to end node
    - If a tool was selected → route to tool execution
    - Otherwise → end with no answer
    
    This creates the ReAct loop: Reason → Tool → Reason → Answer
    
    Args:
        state: Current agent state containing tool selection or final answer
        
    Returns:
        "execute_tool": Route to tool execution node
        "provide_answer": Route to end node to provide final answer
    """
    if state.get('final_answer'):
        return "provide_answer"
    elif state.get('tool_name'):
        return "execute_tool"
    else:
        return "provide_answer"


def create_workflow() -> StateGraph:
    """Create and compile the ReAct workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("reason", reasoning_node)
    workflow.add_node("tool", tool_execution_node)
    
    # Define the flow
    workflow.set_entry_point("reason")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "reason",
        route_after_reasoning,
        {
            "execute_tool": "tool",
            "provide_answer": END
        }
    )
    
    # After tool execution, go back to reasoning
    workflow.add_edge("tool", "reason")
    
    return workflow.compile()


def run_qa_workflow(question: str) -> Dict[str, Any]:
    """
    Run the multi-tool agent workflow.
    
    Args:
        question: The user's question
        
    Returns:
        Dictionary containing the answer and execution history
    """
    print("\n" + "="*60)
    print("🤖 MULTI-TOOL AGENT")
    print("="*60)
    
    # Check if LangSmith tracing is enabled
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        langsmith_project = os.getenv("LANGCHAIN_PROJECT", "default")
        print(f"\n📊 LangSmith tracing enabled (project: {langsmith_project})")
        print("   View traces at: https://smith.langchain.com")
    
    # Create initial state
    initial_state = {
        'question': question,
        'steps': []
    }
    
    # Create and run workflow
    workflow = create_workflow()
    
    # Execute the workflow with LangSmith tracing config
    try:
        # Enhanced config for LangSmith tracing
        config = {
            "recursion_limit": 25,
            "run_name": f"wiki_qa: {question[:50]}{'...' if len(question) > 50 else ''}",
            "tags": ["wiki-agent", "multi-tool", "react-pattern"],
            "metadata": {
                "question_length": len(question),
                "timestamp": datetime.now().isoformat(),
                "has_langsmith": os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
            }
        }
        result = workflow.invoke(initial_state, config=config)
    except Exception as e:
        result = initial_state.copy()
        result['error'] = str(e)
        if "recursion limit" in str(e).lower():
            result['final_answer'] = "I apologize, but I wasn't able to complete the request as it required too many steps. Please try rephrasing your question or being more specific."
        else:
            result['final_answer'] = f"I encountered an error while processing your request: {str(e)}"
    
    # Show final answer section if successful
    if result.get('final_answer') and not result.get('error'):
        print("\n" + "="*60)
        print("🎆 FINAL ANSWER")
        print("="*60)
    
    # Extract final result
    return {
        'question': question,
        'answer': result.get('final_answer', 'Unable to generate an answer'),
        'steps': result.get('steps', []),
        'error': result.get('error')
    }