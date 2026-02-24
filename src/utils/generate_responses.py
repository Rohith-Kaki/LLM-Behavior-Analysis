"""
Module for generating responses from language models.
"""

import torch
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate responses from language models."""
    
    def __init__(self, model, tokenizer, model_config: Dict, model_name: str):
        """
        Initialize ResponseGenerator.
        
        Args:
            model: The language model
            tokenizer: The tokenizer for the model
            model_config: Configuration dictionary for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = model_config
        self.device = next(model.parameters()).device
        self.model_name = model_name

    
    def generate(self, prompt: str) -> str:
        """
        Generate a response for a given prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text response
        """        
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True,
        max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 50),
                temperature=self.config.get('temperature', 0.0),
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                do_sample=False,
                early_stopping = True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # if "gpt2" in self.model_name:
        #     response = response[len(prompt):].strip()

        response = response.strip()
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response
    
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            try:
                response = self.generate(prompt)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error generating response for prompt: {prompt}")
                responses.append(f"Error: {str(e)}")
        
        return responses
