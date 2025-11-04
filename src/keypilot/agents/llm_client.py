"""
LLM client interfaces for data generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import os
from loguru import logger

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import requests
except ImportError:
    requests = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text
        """
        pass


class ChatGPTClient(LLMClient):
    """OpenAI ChatGPT client for data generation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize ChatGPT client.
        
        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: Model name to use
            
        Raises:
            ValueError: If OpenAI package is not installed or API key is missing
        """
        if OpenAI is None:
            raise ValueError("openai package is not installed. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"ChatGPT client initialized with model: {model}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text using ChatGPT.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for generating keyboard typing data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            generated_text = response.choices[0].message.content
            logger.debug(f"Generated {len(generated_text)} characters from ChatGPT")
            return generated_text
            
        except Exception as e:
            logger.error(f"ChatGPT API call failed: {e}")
            raise RuntimeError(f"ChatGPT API call failed: {e}")


class DeepSeekClient(LLMClient):
    """DeepSeek API client for data generation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.deepseek.com/v1/chat/completions",
        model: str = "deepseek-chat"
    ):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key (if None, reads from DEEPSEEK_API_KEY env var)
            api_url: API endpoint URL
            model: Model name to use
            
        Raises:
            ValueError: If requests package is not installed or API key is missing
        """
        if requests is None:
            raise ValueError("requests package is not installed. Install with: pip install requests")
        
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter")
        
        self.api_url = api_url
        self.model = model
        logger.info(f"DeepSeek client initialized with model: {model}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text using DeepSeek.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If API call fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for generating keyboard typing data."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result['choices'][0]['message']['content']
            logger.debug(f"Generated {len(generated_text)} characters from DeepSeek")
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek API call failed: {e}")
            raise RuntimeError(f"DeepSeek API call failed: {e}")
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse DeepSeek response: {e}")
            raise RuntimeError(f"Failed to parse DeepSeek response: {e}")

