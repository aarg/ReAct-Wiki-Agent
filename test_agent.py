#!/usr/bin/env python3
"""
Test script to demonstrate the multi-tool agent capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.workflow import run_qa_workflow

# Test questions that require different tools
test_questions = [
    "What is 15 * 23 + 47?",
    "What day of the week is today?",
    "Who won the 2024 Super Bowl?",
    "Calculate the square root of 144 plus 10% of 200",
    "What time is it right now?"
]

for question in test_questions:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    
    result = run_qa_workflow(question)
    
    # Show steps
    if result.get('steps'):
        print("\nSteps taken:")
        for i, step in enumerate(result['steps'], 1):
            print(f"\n{i}. Thought: {step.get('thought', 'N/A')}")
            print(f"   Tool: {step.get('tool', 'N/A')}")
            print(f"   Input: {step.get('tool_input', 'N/A')}")
            obs = step.get('observation', 'N/A')
            if len(obs) > 100:
                print(f"   Result: {obs[:100]}...")
            else:
                print(f"   Result: {obs}")
    
    print(f"\nFinal Answer: {result['answer']}")
    
    if result.get('error'):
        print(f"Error: {result['error']}")