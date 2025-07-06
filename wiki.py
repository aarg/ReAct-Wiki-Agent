#!/usr/bin/env python3
"""
Multi-Tool AI Agent - Ask questions and get answers using various tools

Usage: python wiki.py "Your question here"
"""

import sys
import os
import argparse

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.workflow import run_qa_workflow


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Tool AI Agent - Ask questions and get answers using various tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wiki.py "What is 25 * 4 + 10?"
  python wiki.py "What day of the week is it today?"
  python wiki.py "Who is the current president of France?"
  python wiki.py "What is the weather like in Tokyo?"
        """
    )
    
    parser.add_argument(
        'question',
        help='The question to answer'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed reasoning steps'
    )
    
    args = parser.parse_args()
    
    # Process the question
    print(f"\n🤔 Question: {args.question}\n")
    
    try:
        result = run_qa_workflow(args.question)
        
        # Show reasoning steps if verbose
        if args.verbose and result.get('steps'):
            print("🔍 Reasoning Steps:")
            print("-" * 50)
            for i, step in enumerate(result['steps'], 1):
                print(f"\nStep {i}:")
                if step.get('thought'):
                    print(f"💭 Thought: {step['thought']}")
                if step.get('tool'):
                    print(f"🔧 Tool: {step['tool']}")
                    print(f"📥 Input: {step.get('tool_input', 'None')}")
                if step.get('observation'):
                    print(f"👁️ Observation: {step['observation'][:200]}..." 
                          if len(step['observation']) > 200 else f"👁️ Observation: {step['observation']}")
            print("\n" + "-" * 50)
        
        # Display the final answer
        print("\n✅ Answer:")
        print("-" * 50)
        print(result['answer'])
            
        if result.get('error'):
            print(f"\n⚠️ Note: {result['error']}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()