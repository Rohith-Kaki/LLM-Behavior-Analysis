import os
import yaml
import pandas as pd
from tqdm import tqdm
import logging

from src.utils.load_models import ModelLoader
from src.utils.generate_responses import ResponseGenerator
from src.utils.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "results/raw_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_output(text):
    text = text.strip()
    
    # Remove A), B), etc.
    if text.startswith(("A)", "B)", "C)", "D)")):
        text = text[2:].strip()
    
    # Remove prompt echo
    if "Question:" in text and "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    
    return text

def run_generation(config_path="config/model_config.yaml"):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load datasets
    logger.info("Loading datasets...")
    datasets = DataLoader.load_all_categories()

    # Load models
    model_loader = ModelLoader(config_path)

    for model_name in config['models']:
        logger.info(f"\n=== Running model: {model_name} ===")

        model, tokenizer = model_loader.load_model(model_name)
        generator = ResponseGenerator(
            model,
            tokenizer,
            config['models'][model_name],
            model_name
        )

        rows = []
        for category, data in datasets.items():
            logger.info(f"Processing category: {category} ({len(data)} samples)")

            # Optional: test mode
            # data = data[:20]

            for sample in tqdm(data):
                prompt = sample["input"]

                if "phi-1_5" in model_name:
                    # Base LM → needs Q/A format
                    prompt = f"Question: {prompt}\nAnswer in one short phrase:"
                else:
                    # Instruction models (FLAN, LLaMA, Mistral, etc.)
                    prompt = f"Answer the following question concisely:\n{prompt}"

                try:
                    response = generator.generate(prompt)
                    response = clean_output(response)
                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    response = "GENERATION_ERROR"

                rows.append({
                    "model": model_name,
                    "dataset": category,
                    "id": sample["id"],
                    "input": sample["input"],
                    "expected_output": sample.get("expected_output", ""),
                    "generated_output": response
                })
                

        # Save CSV
        output_file = os.path.join(OUTPUT_DIR, f"{model_name}.csv")
        pd.DataFrame(rows).to_csv(output_file, index=False)
        logger.info(f"Saved outputs to {output_file}")


if __name__ == "__main__":
    run_generation()
