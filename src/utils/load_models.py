"""
Module for loading and initializing language models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from typing import Dict, Tuple
import yaml
import logging
import os   


HF_TOKEN = os.getenv("HF_TOKEN")

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage language models."""
    
    def __init__(self, config_path: str):
        """
        Initialize ModelLoader with configuration.
        
        Args:
            config_path: Path to model_config.yaml
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.models = {}
        self.tokenizers = {}
    
    def load_model(self, model_name: str) -> Tuple:
        """
        Load a specific model and tokenizer.
        
        Args:
            model_name: Name of the model (e.g., 'gpt2', 'flan_t5', 'llama')
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_name not in self.config['models']:
            raise ValueError(f"Model {model_name} not found in config")
        
        model_config = self.config['models'][model_name]
        model_path = model_config['model_path']
        device = torch.device(model_config['device'] if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading {model_name} from {model_path}")
        print(f"Loading {model_name} from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
        
        if 'flan' in model_path.lower() or 't5' in model_path.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, token=HF_TOKEN)
        
        model = model.to(device)
        model.eval()
        
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        
        logger.info(f"Successfully loaded {model_name}")
        return model, tokenizer
    
    def load_all_models(self) -> Dict:
        """Load all configured models."""
        for model_name in self.config['models'].keys():
            self.load_model(model_name)
        return self.models
