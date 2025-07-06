"""
Claude LLM client for the Web QnA Agent.

This module handles all interactions with the Anthropic Claude API,
including prompt engineering, response generation, and token optimization.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
load_dotenv()

class ClaudeClient:
    """
    Client for interacting with Anthropic's Claude API.
    
    This class encapsulates prompt engineering strategies and
    provides a clean interface for generating answers based on
    web content.
    """
    
    # System prompt template for the QnA agent
    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided content.
Your role is to provide accurate, informative answers using ONLY the information provided in the context.

Guidelines:
1. Base your answers strictly on the provided content
2. If the context doesn't contain enough information to answer the question, clearly state this
3. Be concise but comprehensive in your responses
4. Cite specific information from the context when relevant
5. If the context contains conflicting information, acknowledge this
6. Do not make up or infer information not present in the context
7. Structure your answers clearly with paragraphs when appropriate"""

    # User prompt template
    USER_PROMPT_TEMPLATE = """Question: {question}

Context:
{context}

Please provide a comprehensive answer based solely on the above context."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the Claude client.
        
        Args:
            api_key: Anthropic API key (defaults to env variable ANTHROPIC_API_KEY)
            model: Claude model to use (default: claude-3-haiku for cost efficiency)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables or provided")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def generate_answer(
        self, 
        question: str, 
        context: str,
        max_tokens: int = 1000,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate an answer to the question based on the provided context.
        
        Args:
            question: The user's question
            context: Source content (Wikipedia article, web page, etc.)
            max_tokens: Maximum tokens for the response
            temperature: Sampling temperature (lower = more focused)
            
        Returns:
            Dictionary containing:
                - answer: The generated answer
                - usage: Token usage statistics
                - model: Model used for generation
                
        Raises:
            Exception: If API call fails after retries
        """
        try:
            # Prepare the user prompt
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                question=question,
                context=context[:10000]  # Limit context to avoid token limits
            )
            
            # Classify question and get appropriate prompt
            question_type = self.classify_question(question)
            system_prompt = self.create_prompt_variant(question_type)
            
            # Make API call
            response = self.client.messages.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract the answer
            answer = response.content[0].text if response.content else "No response generated"
            
            # Prepare response with metadata
            result = {
                "answer": answer,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                "model": self.model
            }
            
            return result
            
        except Exception:
            raise
    
    def classify_question(self, question: str) -> str:
        """
        Classify question type based on keywords.
        
        Args:
            question: The user's question
            
        Returns:
            Question type: 'historical', 'comparative', 'analytical', or 'factual'
        """
        q_lower = question.lower()
        
        # Historical questions - focus on time and chronology
        if any(word in q_lower for word in ['when', 'date', 'year', 'who invented', 'founded', 'discovered', 'history']):
            return 'historical'
        
        # Comparative questions - focus on similarities and differences
        elif any(word in q_lower for word in ['compare', 'difference', 'similar', 'versus', 'vs', 'better', 'worse']):
            return 'comparative'
        
        # Analytical questions - focus on reasoning and relationships
        elif any(word in q_lower for word in ['why', 'how does', 'explain', 'analyze', 'cause', 'effect', 'impact']):
            return 'analytical'
        
        # Default to factual - focus on specific information
        else:
            return 'factual'
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate the generated response meets quality criteria.
        
        Args:
            response: The response dictionary from generate_answer
            
        Returns:
            True if response is valid, False otherwise
        """
        # Check if answer exists and is not empty
        if not response.get("answer") or len(response["answer"].strip()) < 10:
            return False
        
        # Check for common failure patterns
        failure_phrases = [
            "i cannot answer",
            "no information provided",
            "context does not contain"
        ]
        
        answer_lower = response["answer"].lower()
        for phrase in failure_phrases:
            if phrase in answer_lower and len(response["answer"]) < 100:
                return True  # This is actually valid - admitting lack of info
        
        return True
    
    def create_prompt_variant(self, question_type: str) -> str:
        """
        Create specialized prompt variants based on question type.
        
        Args:
            question_type: Type of question (e.g., 'factual', 'analytical', 'comparative')
            
        Returns:
            Specialized system prompt for the question type
        """
        variants = {
            "factual": self.SYSTEM_PROMPT + "\n\nFor factual questions, focus on providing specific dates, names, and figures from the context.",
            
            "analytical": self.SYSTEM_PROMPT + "\n\nFor analytical questions, explain the relationships and implications found in the context.",
            
            "comparative": self.SYSTEM_PROMPT + "\n\nFor comparative questions, clearly outline similarities and differences found in the context.",
            
            "historical": self.SYSTEM_PROMPT + "\n\nFor historical questions, pay special attention to chronology and cause-effect relationships in the context."
        }
        
        return variants.get(question_type, self.SYSTEM_PROMPT)
    
    def generate_reasoning(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate reasoning for tool selection without the QA system prompt.
        
        Args:
            prompt: The reasoning prompt with instructions
            max_tokens: Maximum tokens for the response
            
        Returns:
            The raw reasoning text
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                # No system prompt to avoid conflicting instructions
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            return response.content[0].text if response.content else ""
            
        except Exception:
            raise