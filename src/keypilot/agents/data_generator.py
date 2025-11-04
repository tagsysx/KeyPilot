"""
Multi-agent pipeline for generating diverse training data.
"""

import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from tqdm import tqdm

from .llm_client import LLMClient, ChatGPTClient, DeepSeekClient


class DataGenerationAgent:
    """
    Agent for generating keyboard typing training data.
    
    Each agent generates diverse scenarios with different:
    - Conversation contexts (chat, email, forms, etc.)
    - Language patterns (formal, casual, technical, etc.)
    - Input intents and keyboard layouts
    """
    
    def __init__(
        self,
        agent_id: int,
        llm_client: LLMClient,
        intent_classes: List[str],
        layout_types: List[str],
    ):
        """
        Initialize data generation agent.
        
        Args:
            agent_id: Unique identifier for this agent
            llm_client: LLM client for generating data
            intent_classes: List of input intent classes
            layout_types: List of keyboard layout types
            
        Raises:
            ValueError: If intent_classes or layout_types are empty
        """
        if not intent_classes:
            raise ValueError("intent_classes cannot be empty")
        if not layout_types:
            raise ValueError("layout_types cannot be empty")
        
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.intent_classes = intent_classes
        self.layout_types = layout_types
        
        logger.info(f"Agent {agent_id} initialized")
    
    def generate_sample(self, scenario: str) -> Dict[str, Any]:
        """
        Generate a single training sample.
        
        Args:
            scenario: Description of the typing scenario
            
        Returns:
            Dictionary containing:
                - conversation_text: Text conversation history
                - screen_description: Description of screen context
                - next_intent: Next input intent
                - optimal_layout: Optimal keyboard layout
                
        Raises:
            RuntimeError: If sample generation fails
        """
        prompt = f"""
Generate a realistic keyboard typing scenario for mobile device input prediction.

Scenario: {scenario}

Please provide:
1. Conversation text (what has been typed so far)
2. Screen description (what's visible on screen: app, UI elements, context)
3. Next input intent (choose from: {', '.join(self.intent_classes)})
4. Optimal keyboard layout (choose from: {', '.join(self.layout_types)})

Format your response as JSON:
{{
    "conversation_text": "...",
    "screen_description": "...",
    "next_intent": "...",
    "optimal_layout": "..."
}}
"""
        
        try:
            response = self.llm_client.generate(prompt, temperature=0.8)
            
            # Parse JSON response
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                logger.error(f"No JSON found in response: {response[:200]}")
                raise RuntimeError("Failed to parse JSON from LLM response")
            
            json_str = response[start_idx:end_idx+1]
            sample = json.loads(json_str)
            
            # Validate sample
            required_keys = ["conversation_text", "screen_description", "next_intent", "optimal_layout"]
            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"Missing required key in generated sample: {key}")
            
            # Validate intent and layout values
            if sample["next_intent"] not in self.intent_classes:
                logger.warning(f"Invalid intent '{sample['next_intent']}', defaulting to 'text'")
                sample["next_intent"] = "text"
            
            if sample["optimal_layout"] not in self.layout_types:
                logger.warning(f"Invalid layout '{sample['optimal_layout']}', defaulting to 'qwerty'")
                sample["optimal_layout"] = "qwerty"
            
            sample["agent_id"] = self.agent_id
            sample["scenario"] = scenario
            
            return sample
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise RuntimeError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            logger.error(f"Sample generation failed: {e}")
            raise RuntimeError(f"Sample generation failed: {e}")
    
    def generate_batch(
        self,
        num_samples: int,
        scenarios: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of training samples.
        
        Args:
            num_samples: Number of samples to generate
            scenarios: List of scenarios to use (if None, uses default scenarios)
            
        Returns:
            List of generated samples
            
        Raises:
            ValueError: If num_samples <= 0
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        
        if scenarios is None:
            scenarios = self._get_default_scenarios()
        
        samples = []
        logger.info(f"Agent {self.agent_id} generating {num_samples} samples")
        
        for i in tqdm(range(num_samples), desc=f"Agent {self.agent_id}"):
            scenario = random.choice(scenarios)
            try:
                sample = self.generate_sample(scenario)
                samples.append(sample)
            except Exception as e:
                logger.error(f"Failed to generate sample {i}: {e}")
                # Continue generating other samples
        
        logger.info(f"Agent {self.agent_id} generated {len(samples)}/{num_samples} samples")
        return samples
    
    def _get_default_scenarios(self) -> List[str]:
        """Get default typing scenarios."""
        return [
            "Replying to a casual text message from a friend",
            "Composing a professional email to a colleague",
            "Filling out a registration form with personal information",
            "Chatting in a group conversation about weekend plans",
            "Writing a social media post about a recent event",
            "Entering a phone number in a contact form",
            "Typing a search query in a shopping app",
            "Responding to a customer support inquiry",
            "Writing code in a mobile code editor",
            "Composing a tweet with hashtags and emojis",
            "Entering payment information in a checkout form",
            "Replying to a dating app conversation",
            "Writing a review for a restaurant",
            "Entering an address in a delivery app",
            "Composing a message in a professional Slack channel",
        ]


class MultiAgentPipeline:
    """
    Multi-agent pipeline for parallel data generation.
    
    Coordinates multiple agents to generate diverse training data
    using different LLM providers (ChatGPT, DeepSeek, etc.)
    """
    
    def __init__(
        self,
        num_agents: int,
        intent_classes: List[str],
        layout_types: List[str],
        use_chatgpt: bool = True,
        use_deepseek: bool = True,
    ):
        """
        Initialize multi-agent pipeline.
        
        Args:
            num_agents: Number of agents to create
            intent_classes: List of input intent classes
            layout_types: List of keyboard layout types
            use_chatgpt: Whether to use ChatGPT
            use_deepseek: Whether to use DeepSeek
            
        Raises:
            ValueError: If num_agents <= 0 or no LLM providers are enabled
        """
        if num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {num_agents}")
        
        if not use_chatgpt and not use_deepseek:
            raise ValueError("At least one LLM provider must be enabled")
        
        self.num_agents = num_agents
        self.intent_classes = intent_classes
        self.layout_types = layout_types
        
        # Create agents with different LLM clients
        self.agents = []
        for i in range(num_agents):
            # Alternate between LLM providers
            if use_chatgpt and (i % 2 == 0 or not use_deepseek):
                try:
                    llm_client = ChatGPTClient()
                    self.agents.append(DataGenerationAgent(i, llm_client, intent_classes, layout_types))
                except ValueError as e:
                    logger.warning(f"Failed to initialize ChatGPT client: {e}")
            elif use_deepseek:
                try:
                    llm_client = DeepSeekClient()
                    self.agents.append(DataGenerationAgent(i, llm_client, intent_classes, layout_types))
                except ValueError as e:
                    logger.warning(f"Failed to initialize DeepSeek client: {e}")
        
        if not self.agents:
            raise ValueError("Failed to initialize any agents. Check API keys and configuration.")
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    def generate_dataset(
        self,
        total_samples: int,
        output_path: str,
        max_workers: int = 4,
    ) -> None:
        """
        Generate dataset using all agents in parallel.
        
        Args:
            total_samples: Total number of samples to generate
            output_path: Path to save generated dataset
            max_workers: Maximum number of parallel workers
            
        Raises:
            ValueError: If total_samples <= 0
            IOError: If dataset cannot be saved
        """
        if total_samples <= 0:
            raise ValueError(f"total_samples must be positive, got {total_samples}")
        
        samples_per_agent = total_samples // len(self.agents)
        logger.info(f"Generating {total_samples} samples across {len(self.agents)} agents ({samples_per_agent} each)")
        
        all_samples = []
        
        # Generate data in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(agent.generate_batch, samples_per_agent)
                for agent in self.agents
            ]
            
            for future in as_completed(futures):
                try:
                    samples = future.result()
                    all_samples.extend(samples)
                except Exception as e:
                    logger.error(f"Agent failed: {e}")
        
        logger.info(f"Generated total of {len(all_samples)} samples")
        
        # Save dataset
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(all_samples, f, indent=2)
            logger.info(f"Dataset saved to {output_path}")
        except IOError as e:
            logger.error(f"Failed to save dataset: {e}")
            raise IOError(f"Failed to save dataset: {e}")

