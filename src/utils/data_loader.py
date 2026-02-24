"""
Data loading utilities.
"""

import json
import logging
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage test datasets."""
    
    def __init__(self):
        """Initialize DataLoader."""
        pass
    
    @staticmethod
    def load_json(file_path: str) -> List[Dict]:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of data dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} samples from {file_path}")
            return data if isinstance(data, list) else [data]
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in file: {file_path}")
            return []
    
    @staticmethod
    def save_json(data: List[Dict], file_path: str) -> bool:
        """
        Save data to JSON file.
        
        Args:
            data: List of dictionaries to save
            file_path: Path to output JSON file
            
        Returns:
            Boolean indicating success
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(data)} samples to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return False
    
    @staticmethod
    def load_dataset_category(category_name: str, data_dir: str = "data") -> List[Dict]:
        """
        Load dataset for a specific category.
        
        Args:
            category_name: Name of the category (e.g., 'factual', 'bias')
            data_dir: Base data directory
            
        Returns:
            List of data samples
        """
        file_path = f"{data_dir}/{category_name}/{category_name}.json"
        return DataLoader.load_json(file_path)
    
    @staticmethod
    def load_all_categories(data_dir: str = "data") -> Dict[str, List[Dict]]:
        """
        Load all available dataset categories.
        
        Args:
            data_dir: Base data directory
            
        Returns:
            Dictionary mapping category names to data samples
        """
        categories = ['factual', 'reasoning', 'ambiguous', 'bias', 'safety', 'context']
        all_data = {}
        
        for category in categories:
            data = DataLoader.load_dataset_category(category, data_dir)
            all_data[category] = data
        
        return all_data
    
# format of all_data
# {
#     'factual': [
#         {
#             "id": "ambig_0022",
#             "input": "What type of government system does the usa have?",
#             "expected_output": [
#             "federal republic",
#             "Republic"
#             ],
#             "task_type": "ambiguous",
#             "metadata": {
#             "ambiguity_type": "multiple_valid_answers"
#             }
#         },
#     ],
    # "reasoning": []
# }