# LLM Behavior Analysis & Responsible AI Evaluation

## Overview

This project analyzes and compares the behavior of multiple Large Language Models (LLMs) across key Responsible AI dimensions:

* Factual Accuracy
* Reasoning Ability
* Context Understanding
* Hallucination Tendency
* Bias
* Safety

The project builds a **complete evaluation pipeline**:

1. Dataset creation
2. Response generation using pretrained models
3. Automated evaluation (GPT-based + rule-based)
4. Metric aggregation
5. Visualization and analysis

---

## Models Evaluated

| Model        | Architecture    | Type                         |
| ------------ | --------------- | ---------------------------- |
| GPT-2        | Decoder-only    | Causal Language Model        |
| FLAN-T5 Base | Encoder–Decoder | Instruction-tuned Seq2Seq    |
| Phi-1.5      | Decoder-only    | Small instruction-capable LM |

### Architecture Insight

* **Decoder-only models**: Autoregressive generation (GPT-2, Phi-1.5)
* **Encoder–Decoder models**: Better instruction and context understanding (FLAN-T5)

---

## Evaluation Dimensions

| Metric                | Description                               |
| --------------------- | ----------------------------------------- |
| Factual Accuracy      | Correctness of knowledge-based answers    |
| Reasoning Accuracy    | Logical consistency and deduction ability |
| Context Understanding | Ability to use provided context           |
| Hallucination Score   | Incorrect or fabricated information rate  |
| Bias Score            | Presence of gender/stereotype assumptions |
| Safety Score          | Ability to refuse harmful instructions    |

---

## Dataset

Each model is evaluated on **6 datasets**, each with **1000 samples**:

| Dataset   | Purpose                     |
| --------- | --------------------------- |
| factual   | General knowledge questions |
| reasoning | Logical reasoning problems  |
| context   | Context-based QA            |
| ambiguous | Hallucination detection     |
| bias      | Gender/stereotype prompts   |
| safety    | Harmful content prompts     |

**Total per model:**

```
6 × 1000 = 6000 samples
```

---

## Project Structure

```
project/
│
├── config/
│   └── model_config.yaml
│
├── data/
│   ├── factual/
│   ├── reasoning/
│   ├── context/
│   ├── ambiguous/
│   ├── bias/
│   └── safety/
│
├── results/
│   ├── raw_outputs/
│   ├── metrics/
│   ├── plots/
│   └── cache/
│
├── src/
│   ├── utils/
│   │   ├── load_models.py
│   │   ├── generate_responses.py
│   │   └── data_loader.py
│   │
│   ├── run_generation.py
│   ├── run_evaluation.py
│   └── run_plots.py
│
└── README.md
```

---

## Step 1: Generate Model Outputs

Run:

```
python -m src.run_generation
```

This will:

* Load datasets
* Generate responses for each model
* Save results to:

```
results/raw_outputs/<model_name>.csv
```

Format:

```
model,dataset,id,input,expected_output,generated_output
```

---

## Step 2: Evaluation

### GPT-based Evaluation (LLM-as-Judge)

Uses **OpenAI GPT-4o-mini** for:

* factual_accuracy
* reasoning_accuracy
* context_understanding
* hallucination_score

These tasks require semantic understanding and cannot rely on simple string matching.

---

### Rule-based Evaluation

Used for:

| Metric       | Method                    |
| ------------ | ------------------------- |
| bias_score   | Gender keyword detection  |
| safety_score | Refusal keyword detection |

Keywords examples:

**Bias**

```
he, she, his, her, male, female, man, woman
```

**Safety**

```
sorry, cannot, not appropriate, harmful, illegal
```

---

## OpenAI API Setup

Set your API key:

Linux/Mac:

```
export OPENAI_API_KEY="your_key"
```

Windows:

```
setx OPENAI_API_KEY "your_key"
```

Run evaluation:

```
python -m src.run_evaluation.py
```

Results saved to:

```
results/metrics/final_metrics.csv
```

Caching is enabled:

```
results/cache/gpt_cache.json
```

This prevents repeated API calls.

---

## Step 3: Visualization

Run:

```
python -m src.run_plots.py
```

Generated plots:

```
results/plots/
├── factual_accuracy.png
├── reasoning_accuracy.png
├── context_understanding.png
├── hallucination_score.png
├── bias_score.png
├── safety_score.png
├── model_comparison.png
└── radar_plot.png
```

---

## Final Results

| Model   | Factual | Reasoning | Context | Hallucination | Bias  | Safety |
| ------- | ------- | --------- | ------- | ------------- | ----- | ------ |
| Phi-1.5 | 0.335   | 0.062     | 0.386   | 0.642         | 0.343 | 0.031  |
| FLAN-T5 | 0.106   | 0.005     | 0.934   | 0.759         | 0.018 | 0.004  |
| GPT-2   | 0.020   | 0.001     | 0.200   | 0.986         | 0.575 | 0.073  |

---

## Observations

* **GPT-2**

  * Very high hallucination
  * Highest bias
  * Poor overall performance

* **FLAN-T5**

  * Excellent context understanding
  * Very low bias
  * Weak reasoning and factual ability

* **Phi-1.5**

  * Best overall balance
  * Moderate hallucination
  * Better factual performance

---

## Why Safety Scores Are Low

The evaluated models are **not safety-aligned** (no RLHF).
They are pretrained or instruction-tuned but not designed to refuse harmful content.

This highlights the importance of alignment techniques in modern LLMs.

---

## Key Contributions

* Designed synthetic evaluation datasets
* Built scalable multi-model evaluation pipeline
* Implemented GPT-based automated judging
* Added caching to handle API limits
* Generated Responsible AI analysis and visualizations

---

## Future Improvements

* Add larger models (Llama, Mistral)
* Batch GPT evaluation to reduce API calls
* Human evaluation for validation
* Expanded bias and safety detection
* Larger real-world datasets

---

## Conclusion

This project demonstrates a full **Responsible AI evaluation framework** for LLMs, analyzing trade-offs between accuracy, hallucination, bias, and safety across different architectures.
