# LLM API Semantic Annotation

This directory contains Python scripts for automatic semantic classification of Estonian nouns and noun phrases using Large Language Models (LLMs) through either the **OpenRouter** or **Featherless.ai** APIs.

The scripts are intentionally kept as separate files. Each file represents one experiment or experiment variant, preserving the exact prompt, input columns, API setup, and output shape used for that run.

## Run From This Directory

The examples below assume that commands are run from this folder:

```bash
cd code/LLM_API
```

Paths such as `.env`, `millest_mis_1000.csv`, and output CSV names are resolved relative to the current working directory unless absolute paths are provided.

## Setup and API Keys

Create a `.env` file in this directory:

```env
# OpenRouter API key (used by most scripts)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Featherless API key (used by TartuNLP EstLLM scripts)
FEATHERLESS_API_KEY=your_featherless_api_key_here
```

Most scripts use OpenRouter and require `OPENROUTER_API_KEY`. The `*_estllm.py` scripts use Featherless.ai and require `FEATHERLESS_API_KEY`.

## Dependencies

The LLM API scripts use only the Python standard library. No package installation is required for running the classifiers themselves.

The analysis notebooks in `../result_analysis` are separate from these API scripts and use additional packages such as `pandas` and `scikit-learn`.

## General Command-Line Arguments

Most scripts support the following parameters:

- `model_name` (positional): LLM identifier, e.g. `google/gemini-2.5-flash-lite` or `meta-llama/llama-3.3-70b-instruct`.
- `output_csv` (positional, optional): output CSV path. Defaults to `mudel_vastused.csv`.
- `--input_csv`: input CSV path.
- `--env_file`: path to the `.env` file. Defaults to `.env`.
- `--sleep`: delay between API calls in seconds. Defaults to `0.0`.
- `--timeout`: HTTP timeout in seconds. Defaults to `60`.
- `--max_retries`: maximum retry attempts for failed API calls. Defaults to `4`.
- `--overwrite`: overwrite the output CSV instead of resuming progress.

The Featherless/EstLLM scripts also support `--api_url`, which defaults to the Featherless chat completions endpoint.

## Labels

All classifier scripts write binary labels:

- `1` = yes, the item matches the target category or definition.
- `0` = no, the item does not match the target category or definition.

The main target categories are:

- **Profession/Nationality/Role** (`ELUKUTSET, RAHVUST või ROLLI`): the noun or phrase component expresses a profession, nationality, or role.
- **Material/Substance** (`MATERJALI`): the noun or phrase component expresses a material or substance.
- **Elative appositive**: the phrase matches the definition *"seestütlevas lisand nimetab mingit tunnust esile tõstes sama objekti kui põhisõnagi"*.
- **Elative modifier**: the phrase matches the definition *"seestütlevas täiend väljendab põhisõna referendi ainet või materjali"*.

## Input Data

The committed file `millest_mis_1000.csv` contains the main phrase-level reference data used by the Test2 and Test3 scripts. It includes columns such as:

- `sentence`
- `instance_form_long`
- `instance_form`
- `instance_lemma`
- `comp1_form`
- `comp1_lemma`
- `comp2_form`
- `comp2_lemma`
- `comp1_id`
- `comp2_id`

The Test1 scripts expect word-list inputs with one noun or lemma per row. Some historical/default filenames used by the scripts, such as `katse1_unikaalsed_lemmad_EI_llama.csv`, are experiment inputs and may be external to this directory.

Phrase-level scripts expect semicolon-delimited CSV files with named columns.

## Output Format and Resuming

Output files are semicolon-delimited CSV files. The final column is always `0/1`.

If an output file already exists, scripts read the existing valid rows and continue with the remaining input. This makes interrupted API runs resumable. Use `--overwrite` to start the output file from scratch.

Some later scripts validate the existing output header before resuming. If the header does not match the expected columns, choose a new output file or rerun with `--overwrite`.

## Directory Structure and Scripts

### Test1: Word-Level Classification

These scripts classify individual nouns or lemmas.

| Script | Task | Prompt/API variant | Input | Output |
| --- | --- | --- | --- | --- |
| `Test1/classify_nouns.py` | profession/nationality/role | OpenRouter, zero-shot | first column contains one lemma per row | `lemma;0/1` |
| `Test1/classify_material.py` | material/substance | OpenRouter, zero-shot | first column contains one lemma per row | `lemma;0/1` |
| `Test1/classify_nouns_few.py` | profession/nationality/role | OpenRouter, few-shot | first column contains one lemma per row | `lemma;0/1` |
| `Test1/classify_material_few.py` | material/substance | OpenRouter, few-shot | first column contains one lemma per row | `lemma;0/1` |
| `Test1/classify_nouns_estllm.py` | profession/nationality/role | TartuNLP EstLLM via Featherless | first column contains one lemma per row | `lemma;0/1` |
| `Test1/classify_material_estllm.py` | material/substance | TartuNLP EstLLM via Featherless | first column contains one lemma per row | `lemma;0/1` |
| `Test1/classify_nouns_3081.py` | profession/nationality/role | OpenRouter variant for `cp1257` input | first column contains one lemma per row | `lemma;0/1` |

### Test2: Phrase-Level Classification With Component Columns

These scripts classify the semantic relation between the first phrase component and the head word. The role/profession/nationality scripts are `2_1` to `2_4`; the material/substance scripts are `2_5` to `2_8`.

| Scripts | Task | Required input columns | Output columns |
| --- | --- | --- | --- |
| `Test2/classify_nouns_2_1.py`, `Test2/classify_material_2_5.py` | role or material | `instance_lemma`, `comp1_form`, `comp2_lemma` | input columns + `0/1` |
| `Test2/classify_nouns_2_2.py`, `Test2/classify_material_2_6.py` | role or material | `instance_form`, `comp1_form`, `comp2_form` | input columns + `0/1` |
| `Test2/classify_nouns_2_3.py`, `Test2/classify_material_2_7.py` | role or material | `instance_form_long`, `comp1_form`, `comp2_form` | input columns + `0/1` |
| `Test2/classify_nouns_2_4.py`, `Test2/classify_material_2_8.py` | role or material with sentence context | `sentence`, `comp1_form`, `comp1_id`, `comp2_form`, `comp2_id` | input columns + `0/1` |

### Test3: Phrase-Level Classification From Phrase/Context

These scripts classify whether a phrase matches a construction-level definition. The elative appositive scripts are `3_1` to `3_5`; the elative modifier/material scripts are `3_6` to `3_10`.

| Scripts | Task | Required input columns | Output columns |
| --- | --- | --- | --- |
| `Test3/classify_nouns_3_1.py`, `Test3/classify_material_3_6.py` | appositive or modifier | `instance_lemma` | `instance_lemma`, `0/1` |
| `Test3/classify_nouns_3_2.py`, `Test3/classify_material_3_7.py` | appositive or modifier | `instance_form` | `instance_form`, `0/1` |
| `Test3/classify_nouns_3_3.py`, `Test3/classify_material_3_8.py` | appositive or modifier with sentence context | `instance_form`, `sentence` | `instance_form`, `sentence`, `0/1` |
| `Test3/classify_nouns_3_4.py`, `Test3/classify_material_3_9.py` | appositive or modifier | `instance_form_long` | `instance_form_long`, `0/1` |
| `Test3/classify_nouns_3_5.py`, `Test3/classify_material_3_10.py` | appositive or modifier with sentence context | `instance_form_long`, `sentence` | `instance_form_long`, `sentence`, `0/1` |

## Usage Examples

Run a Test2 profession/nationality/role classifier on the committed reference CSV:

```bash
python Test2/classify_nouns_2_1.py google/gemini-2.5-flash-lite output_nouns_2_1.csv --input_csv millest_mis_1000.csv
```

Run a Test3 material/modifier classifier with sentence context:

```bash
python Test3/classify_material_3_8.py google/gemini-2.5-flash-lite output_material_3_8.csv --input_csv millest_mis_1000.csv
```

Run a Test1 zero-shot material classifier on an external word-list CSV:

```bash
python Test1/classify_material.py google/gemini-2.5-flash-lite output_material.csv --input_csv path/to/word_list.csv
```

Run TartuNLP EstLLM through Featherless.ai:

```bash
python Test1/classify_material_estllm.py tartuNLP/Llama-3.1-EstLLM-8B-Instruct-1125 output_estllm.csv --input_csv path/to/word_list.csv
```

Use `python3` instead of `python` if that is how Python is exposed in your environment.

## Troubleshooting

- **Missing API key**: check that `.env` exists in the directory you are running from, or pass `--env_file`.
- **Input CSV column error**: check the script table above and make sure the input file has the required named columns.
- **Existing output header mismatch**: choose a new output filename or use `--overwrite`.
- **Interrupted run**: rerun the same command with the same output file; completed valid rows will be skipped.
- **API rate limits or temporary failures**: scripts retry HTTP `429`, `500`, `502`, `503`, and `504` responses up to `--max_retries`.
