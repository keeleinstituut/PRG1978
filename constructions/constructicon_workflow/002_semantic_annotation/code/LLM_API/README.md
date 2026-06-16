# LLM API Semantic Annotation

This directory contains Python scripts for the automatic semantic classification of Estonian nouns and noun phrases using Large Language Models (LLMs) via either the **OpenRouter** or **Featherless.ai** APIs.

---

## 🔑 Setup and API Keys

To run the scripts, create a `.env` configuration file in this directory. 

Create a `.env` file with the following contents:
```env
# OpenRouter API key (for standard scripts using OpenRouter)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Featherless API key (for scripts using TartuNLP EstLLM)
FEATHERLESS_API_KEY=your_featherless_api_key_here
```

---

## 🛠️ General Command-Line Arguments

Most scripts support the following parameters:
- `model_name` (positional argument): The LLM identifier (e.g., `google/gemini-2.5-flash-lite` or `meta-llama/llama-3.3-70b-instruct`).
- `output_csv` (positional argument, optional): Path to the output CSV file.
- `--input_csv`: Path to the input CSV file.
- `--env_file`: Path to the `.env` file (defaults to `.env`).
- `--sleep`: Delay between API calls in seconds (defaults to `0.0`).
- `--timeout`: HTTP timeout in seconds (defaults to `60`).
- `--max_retries`: Maximum retry attempts for failed API calls (defaults to `4`).
- `--overwrite`: Overwrite the output CSV file instead of resuming progress.

---

## 📁 Directory Structure & Scripts

The repository is organized into three test folders based on the classification task and input data format:

### 1️⃣ `Test1` (Word-level Classification)
Classifies individual nouns into semantic categories:
- **Material/Substance (`MATERJALI`):** Decides whether a noun denotes a material/substance (1 = YES, 0 = NO).
- **Profession/Nationality/Role (`ELUKUTSET, RAHVUST või ROLLI`):** Decides whether a noun denotes a profession, nationality, or role (1 = YES, 0 = NO).

**Scripts:**
- `classify_material.py` / `classify_nouns.py`: Standard zero-shot classification using OpenRouter API.
- `classify_material_few.py` / `classify_nouns_few.py`: Few-shot classification using prompt examples.
- `classify_material_estllm.py` / `classify_nouns_estllm.py`: Classification using TartuNLP **EstLLM** (`tartuNLP/Llama-3.1-EstLLM-8B-Instruct-1125`) via the Featherless API.
- `classify_nouns_3081.py`: Variant reading a `cp1257`-encoded input file containing 3081 frequent nouns.

---

### 2️⃣ `Test2` (Phrase-level Classification - 3 Columns)
Classifies noun phrases where the input CSV has at least three columns: `instance_lemma` (phrase), `comp1_form` (form of the first word), and `comp2_lemma` (lemma of the head noun).
Decides whether the first word expresses a profession, nationality, or role of the head noun in the phrase.

**Scripts:**
- `classify_nouns_2_1.py` to `2_4.py`: Different prompt and model variations for profession/nationality/role classification.
- `classify_material_2_5.py` to `2_8.py`: Different prompt and model variations for material/substance classification.

---

### 3️⃣ `Test3` (Phrase-level Classification - 1 Column)
Classifies noun phrases analyzing only the complete phrase (`instance_lemma`).
- **Elative Appositive (`3_1.py` to `3_5.py`):** Decides whether the phrase matches the definition: *"the elative appositive names the same object as the head noun, highlighting some trait"* (Estonian: *"seestütlevas lisand nimetab mingit tunnust esile tõstes sama objekti kui põhisõnagi"*).
- **Elative Modifier (`3_6.py` to `3_10.py`):** Decides whether the phrase matches the definition: *"the elative modifier expresses the substance or material of the head noun's referent"* (Estonian: *"seestütlevas täiend väljendab põhisõna referendi ainet või materjali"*).

---

## 🚀 Usage Examples

Run zero-shot material classification:
```bash
python3 Test1/classify_material.py google/gemini-2.5-flash-lite output_material.csv --input_csv Test1/katse1_unikaalsed_lemmad_EI_llama.csv
```

Run TartuNLP EstLLM via the Featherless API:
```bash
python3 Test1/classify_material_estllm.py tartuNLP/Llama-3.1-EstLLM-8B-Instruct-1125 output_estllm.csv --input_csv Test1/katse1_unikaalsed_lemmad_EI_estllm.csv
```