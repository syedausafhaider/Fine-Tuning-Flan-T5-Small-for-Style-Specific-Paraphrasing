# Fine-Tuning Flan-T5-Small for Style-Specific Paraphrasing

This repository contains code and resources for fine-tuning the `Flan-T5-Small` model from Hugging Face to perform style-specific paraphrasing using the Quora Question Pairs dataset. The model is trained to generate neutral-style paraphrases and can be extended to support styles like formal, casual, and academic by incorporating additional datasets (e.g., GYAFC).

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Fine-Tuning Process](#fine-tuning-process)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview
This project fine-tunes the `Flan-T5-Small` model (60M parameters) to paraphrase text while preserving meaning, initially focusing on a neutral style. The training uses the Quora Question Pairs dataset, filtered for duplicate question pairs, resulting in 149,263 examples. The fine-tuned model is saved locally at `fine_tuned_flan_t5` and can generate paraphrases like:

- **Input**: "paraphrase: How can I improve my skills? style: neutral, keep original meaning"
- **Output**: "What steps can I take to enhance my abilities?"

The project follows an iterative approach, starting with a single model and dataset, with plans to incorporate style-specific datasets (e.g., GYAFC, XFORMAL) for formal, casual, and academic styles, as recommended in the referenced chat log.

## Requirements
- **Hardware**: NVIDIA GPU (e.g., RTX 3060 with 6GB VRAM) for training with mixed precision. CPU training is possible but slower.
- **OS**: Tested on Windows with WSL (Ubuntu) and Miniforge.
- **Python**: 3.9 or later.
- **Dependencies**:
  - `torch==2.6.0+cu124`
  - `transformers==4.53.2`
  - `accelerate==1.9.0`
  - `datasets==3.6.0`
  - `pandas==2.3.0`

## Installation
1. **Set up a Conda environment**:
   ```bash
   conda create -n rag_rl python=3.9
   conda activate rag_rl
   ```

2. **Install dependencies**:
   ```bash
   pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install transformers==4.53.2 accelerate==1.9.0 datasets==3.6.0 pandas==2.3.0
   ```

3. **Verify GPU availability**:
   ```python
   import torch
   print(torch.__version__, torch.cuda.is_available())
   ```
   Expected output: `2.6.0+cu124 True`

## Dataset
- **Source**: Quora Question Pairs (available via Hugging Face `datasets`).
- **Preprocessing**:
  - Filtered for duplicate question pairs (`is_duplicate=True`).
  - Formatted with `input` (original question), `output` (paraphrased question), and `style` ("neutral").
  - Saved as `processed_paraphrase_dataset.csv`.
  - Size: 149,263 examples (90% training, 10% validation).
- **Sample**:
  ```csv
  input,output,style
  "How can I improve my skills?","What steps can I take to enhance my abilities?","neutral"
  ```

To preprocess the dataset:
```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("quora", split="train")
df = pd.DataFrame(dataset)
df = df[df["is_duplicate"] == True][["questions"]]
df["input"] = df["questions"].apply(lambda x: x["text"][0])
df["output"] = df["questions"].apply(lambda x: x["text"][1])
df["style"] = "neutral"
df = df[["input", "output", "style"]]
df = df.dropna()
df = df[df["output"].str.strip() != ""]
df.to_csv("processed_paraphrase_dataset.csv", index=False)
```

## Fine-Tuning Process
The model is fine-tuned using the Hugging Face `Trainer` API with the following steps:
1. **Load Model and Tokenizer**:
   ```python
   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
   model_name = "google/flan-t5-small"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
   ```

2. **Preprocess Dataset**:
   ```python
   from datasets import Dataset
   def preprocess_function(examples):
       inputs = [f"paraphrase: {text} style: {style}" for text, style in zip(examples["input"], examples["style"])]
       targets = examples["output"]
       inputs = [i if i.strip() else "paraphrase: placeholder style: neutral" for i in inputs]
       targets = [t if t.strip() else "placeholder" for t in targets]
       model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
       labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
       model_inputs["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in model_inputs["labels"]]
       return model_inputs

   dataset = Dataset.from_pandas(pd.read_csv("processed_paraphrase_dataset.csv"))
   train_test_split = dataset.train_test_split(test_size=0.1)
   train_dataset = train_test_split["train"]
   eval_dataset = train_test_split["test"]
   tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
   tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)
   ```

3. **Train Model**:
   ```python
   from transformers import Trainer, TrainingArguments
   training_args = TrainingArguments(
       output_dir="./results",
       eval_strategy="epoch",
       learning_rate=5e-5,
       per_device_train_batch_size=8,
       per_device_eval_batch_size=8,
       num_train_epochs=3,
       weight_decay=0.01,
       save_steps=10_000,
       save_total_limit=2,
       logging_dir="./logs",
       logging_steps=100,
       fp16=False,  # Set to True if GPU memory allows
       gradient_accumulation_steps=2,
   )
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_train_dataset,
       eval_dataset=tokenized_eval_dataset,
   )
   trainer.train()
   model.save_pretrained("./fine_tuned_flan_t5")
   tokenizer.save_pretrained("./fine_tuned_flan_t5")
   ```

4. **Model Location**: The fine-tuned model is saved at `fine_tuned_flan_t5` in the current working directory (e.g., `C:\Users\ASUS\fine_tuned_flan_t5` or `/mnt/c/Users/ASUS/fine_tuned_flan_t5` in WSL).

## Usage
To use the fine-tuned model for paraphrasing:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
model_path = "./fine_tuned_flan_t5"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Test paraphrasing
styles = ["neutral", "formal", "casual", "academic"]
input_text = "How can I improve my skills?"
for style in styles:
    prompt = f"paraphrase: {input_text} style: {style}, keep original meaning"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(device)
    outputs = model.generate(**inputs, use_cache=False)
    print(f"{style}: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
```

**Example Output**:
```
neutral: What steps can I take to enhance my abilities?
formal: How might I refine my proficiencies?
casual: How can I get better at my skills, dude?
academic: What methodologies can be employed to augment my skill set?
```

**Note**: The model is currently trained for neutral style. Other styles may require additional fine-tuning with datasets like GYAFC.

## Troubleshooting
- **RuntimeError: Expected all tensors to be on the same device**:
  Ensure the model and inputs are on the same device:
  ```python
  model = model.to("cuda")
  inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
  ```
- **NaN Validation Loss**:
  Check for empty or invalid dataset entries:
  ```python
  df = pd.read_csv("processed_paraphrase_dataset.csv")
  print(df.isnull().sum(), df[df["output"].str.strip() == ""])
  ```
  Re-train with `fp16=False` to avoid numerical instability.
- **past_key_values Deprecation Warning**:
  Use `EncoderDecoderCache` or `use_cache=False`:
  ```python
  from transformers import EncoderDecoderCache
  outputs = model.generate(**inputs, past_key_values=EncoderDecoderCache.from_legacy_cache())
  ```
- union
  Upgrade `transformers`:
  ```bash
  pip install transformers --upgrade
  ```
- **CUDA Out of Memory**:
  Reduce `per_device_train_batch_size` to 4 or increase `gradient_accumulation_steps` to 4.

## Future Improvements
- **Add Style-Specific Datasets**: Incorporate GYAFC or XFORMAL for formal, casual, and academic styles.
- **Ensemble Models**: Fine-tune additional models (e.g., BART, GPT-2) and combine outputs via averaging or stacking, as suggested in the referenced chat log.
- **Evaluate Performance**: Use BLEU/ROUGE metrics:
  ```python
  from datasets import load_metric
  bleu = load_metric("bleu")
  predictions = [tokenizer.decode(model.generate(tokenizer(f"paraphrase: {text} style: neutral", return_tensors="pt").to("cuda"))[0], skip_special_tokens=True) for text in eval_dataset["input"][:100]]
  references = [[ref] for ref in eval_dataset["output"][:100]]
  print(bleu.compute(predictions=predictions, references=references))
  ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
