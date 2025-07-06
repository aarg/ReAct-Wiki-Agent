#!/usr/bin/env python3
"""
Enhanced test script to demonstrate all multi-tool agent capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.workflow import run_qa_workflow

# Test questions that demonstrate each tool
test_questions = [
    # Calculator tests
    ("What is 15 * 23 + 47?", "calculator"),
    ("Calculate the square root of 144 plus 10% of 200", "calculator"),
    
    # DateTime tests
    ("What day of the week is today?", "datetime"),
    ("What time is it right now?", "datetime"),
    
    # DateCalculator tests
    ("What day will it be in 10 days?", "datecalculator"),
    ("How many days until Christmas?", "datecalculator"),
    ("What day was it 30 days ago?", "datecalculator"),
    
    # Web Search tests
    ("Who won the 2024 Super Bowl?", "web_search"),
    ("What is the capital of France?", "web_search"),
    
    # Multi-tool tests
    ("How many days until Thanksgiving?", "web_search + datecalculator"),
]

def test_agent():
    """Run all test cases and display results."""
    print("\n" + "="*60)
    print("MULTI-TOOL AGENT TEST SUITE")
    print("="*60)
    
    for question, expected_tool in test_questions:
        print(f"\n{'='*60}")
        print(f"Test: {expected_tool}")
        print(f"Question: {question}")
        print('='*60)
        
        try:
            result = run_qa_workflow(question)
            
            # Show tools used
            if result.get('steps'):
                tools_used = [step.get('tool', 'N/A') for step in result['steps'] if step.get('tool')]
                print(f"\nTools used: {' → '.join(tools_used)}")
            
            # Show final answer
            print(f"\nFinal Answer: {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
            else:
                print("✅ Success")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    print(f"\n{'='*60}")
    print("TEST SUITE COMPLETE")
    print('='*60)

if __name__ == "__main__":
    test_agent()