# LLM API Semantic Annotation

This directory contains Python scripts for automatic semantic classification of Estonian nouns and noun phrases using Large Language Models (LLMs) through either the **OpenRouter** or **Featherless.ai** APIs.

The scripts are intentionally kept as separate files. Each file represents one experiment or experiment variant, preserving the exact prompt, input columns, API setup, and output shape used for that run.

## Run From This Directory

The examples below assume that commands are run from this folder:

```bash
cd code/LLM_API
```

Paths such as `.env`, `test_data/millest_mis_1000.csv`, and output CSV names are resolved relative to the current working directory unless absolute paths are provided.

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
- `output_csv` (positional, optional): output CSV path. The default depends on the script.
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

The file `test_data/millest_mis_1000.csv` contains the main phrase-level reference data used by the Test2 and Test3 scripts. It includes columns such as:

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

The Test1 scripts expect word-list CSV files where the first row is a header. They use the first header as the input column name and copy that same header to the output file. In the current Test1 workflow, the noun scripts default to `test_data/lemma.csv`, whose header is `lemma`.

Phrase-level scripts expect CSV files with named columns. Input readers sniff comma, semicolon, and tab delimiters. If the delimiter cannot be detected, for example in a single-column file, scripts assume a semicolon-delimited CSV.

## Output Format and Resuming

Output files are semicolon-delimited CSV files. The final column is always `0/1`. Test1, Test2, and Test3 workflow scripts write the input header or headers plus `0/1`, for example `lemma;0/1`, `instance_form;sentence;0/1`, or `instance_form;comp1_form;comp2_form;0/1`.

If an output file already exists, scripts read the existing valid rows and continue with the remaining input. This makes interrupted API runs resumable. Use `--overwrite` to start the output file from scratch.

Some later scripts validate the existing output header before resuming. If the header does not match the expected columns, choose a new output file or rerun with `--overwrite`.

## Directory Structure and Scripts

### Test1: Word-Level Classification

These scripts classify individual nouns or lemmas.

| Script | Task | Prompt/API variant | Input | Output |
| --- | --- | --- | --- | --- |
| `test1/classify_nouns.py` | profession/nationality/role | OpenRouter, zero-shot | first-column word-list CSV; default `test_data/lemma.csv` | input header + `0/1`; default `classified_nouns_1_1.csv` |
| `test1/classify_material.py` | material/substance | OpenRouter, zero-shot | helper output from noun classifier; default `classified_nouns_1_1_zero_values.csv` | input header + `0/1`; default `classified_materials_1_2.csv` |
| `test1/classify_nouns_few.py` | profession/nationality/role | OpenRouter, few-shot | first-column word-list CSV; default `test_data/lemma.csv` | input header + `0/1`; default `classified_nouns_1_1.csv` |
| `test1/classify_material_few.py` | material/substance | OpenRouter, few-shot | helper output from noun classifier; default `classified_nouns_1_1_zero_values.csv` | input header + `0/1`; default `classified_materials_1_2.csv` |
| `test1/classify_nouns_estllm.py` | profession/nationality/role | TartuNLP EstLLM via Featherless | first-column word-list CSV; default `test_data/lemma.csv` | input header + `0/1`; default `classified_nouns_1_1.csv` |
| `test1/classify_material_estllm.py` | material/substance | TartuNLP EstLLM via Featherless | helper output from noun classifier; default `classified_nouns_1_1_zero_values.csv` | input header + `0/1`; default `classified_materials_1_2.csv` |
| `test1/classify_nouns_3081.py` | profession/nationality/role | OpenRouter historical 3081 variant | first-column word-list CSV; default `test_data/lemma.csv` | input header + `0/1`; default `classified_nouns_1_1.csv` |

#### Test1 Workflow

Test1 has two subtasks:

- **Subtask 1**: run a `classify_nouns*.py` script to identify profession, nationality, or role nouns.
- **Subtask 2**: run a matching `classify_material*.py` script only on rows that received `0` in subtask 1.

The default Test1 noun input is `test_data/lemma.csv`, generated from the unique `comp1_lemma` values in `test_data/millest_mis_1000.csv`. The noun output defaults to `classified_nouns_1_1.csv`.

Before running a material script, create the material input with:

```bash
python from_subtask1_extract_zero_values.py classified_nouns_1_1.csv classified_nouns_1_1_zero_values.csv
```

This helper keeps only rows where `0/1` is `0` and removes the `0/1` column. The remaining CSV keeps the original input header, usually `lemma`. The Test1 material scripts default to reading `classified_nouns_1_1_zero_values.csv` and writing `classified_materials_1_2.csv`.

If you run multiple Test1 variants in the same directory, pass explicit output filenames so one experiment does not resume from or append to another variant's output.

### Test2: Phrase-Level Classification With Component Columns

These scripts classify the semantic relation between the first phrase component and the head word. The role/profession/nationality scripts are `2_1` to `2_4`; the material/substance scripts are `2_5` to `2_8`.

Test2 also has two subtasks:

- **Subtask 1**: run a `classify_nouns_2_*.py` script to identify profession, nationality, or role relations.
- **Subtask 2**: run the paired `classify_material_2_*.py` script only on rows that received `0` in subtask 1.

The Test2 input CSVs in `test_data/` are derived from `test_data/millest_mis_1000.csv`. The scripts preserve the input headers in the output and append `0/1`.

| Case | Noun script | Default noun input | Default noun output | Helper output / material input | Material script | Default material output |
| --- | --- | --- | --- | --- | --- | --- |
| `2_1` -> `2_5` | `test2/classify_nouns_2_1.py` | `test_data/instance_lemma_comp1_form_comp2_lemma.csv` | `classified_nouns_2_1.csv` | `classified_nouns_2_1_zero_values.csv` | `test2/classify_material_2_5.py` | `classified_materials_2_5.csv` |
| `2_2` -> `2_6` | `test2/classify_nouns_2_2.py` | `test_data/instance_form_comp1_form_comp2_form.csv` | `classified_nouns_2_2.csv` | `classified_nouns_2_2_zero_values.csv` | `test2/classify_material_2_6.py` | `classified_materials_2_6.csv` |
| `2_3` -> `2_7` | `test2/classify_nouns_2_3.py` | `test_data/instance_form_long_comp1_form_comp2_form.csv` | `classified_nouns_2_3.csv` | `classified_nouns_2_3_zero_values.csv` | `test2/classify_material_2_7.py` | `classified_materials_2_7.csv` |
| `2_4` -> `2_8` | `test2/classify_nouns_2_4.py` | `test_data/sentence_comp_form_comp_id.csv` | `classified_nouns_2_4.csv` | `classified_nouns_2_4_zero_values.csv` | `test2/classify_material_2_8.py` | `classified_materials_2_8.csv` |

Before running a Test2 material script, create its input with `from_subtask1_extract_zero_values.py`. For example:

```bash
python from_subtask1_extract_zero_values.py classified_nouns_2_1.csv classified_nouns_2_1_zero_values.csv
```

The helper keeps only rows where `0/1` is `0`, removes the `0/1` column, and keeps the remaining input headers unchanged. The paired material script reads that helper output by default.

If you run several Test2 cases in the same directory, keep the default filenames per case or pass explicit output filenames so different experiments do not resume from each other's output.

### Test3: Phrase-Level Classification From Phrase/Context

These scripts classify whether a phrase matches a construction-level definition. The elative appositive scripts are `3_1` to `3_5`; the elative modifier/material scripts are `3_6` to `3_10`.

Test3 also has two subtasks:

- **Subtask 1**: run a `classify_nouns_3_*.py` script to identify elative appositive cases.
- **Subtask 2**: run the paired `classify_material_3_*.py` script only on rows that received `0` in subtask 1.

The Test3 input CSVs in `test_data/` are derived from `test_data/millest_mis_1000.csv`. The scripts preserve the input headers in the output and append `0/1`.

| Case | Noun script | Default noun input | Default noun output | Helper output / material input | Material script | Default material output |
| --- | --- | --- | --- | --- | --- | --- |
| `3_1` -> `3_6` | `test3/classify_nouns_3_1.py` | `test_data/instance_lemma.csv` | `classified_nouns_3_1.csv` | `classified_nouns_3_1_zero_values.csv` | `test3/classify_material_3_6.py` | `classified_materials_3_6.csv` |
| `3_2` -> `3_7` | `test3/classify_nouns_3_2.py` | `test_data/instance_form.csv` | `classified_nouns_3_2.csv` | `classified_nouns_3_2_zero_values.csv` | `test3/classify_material_3_7.py` | `classified_materials_3_7.csv` |
| `3_3` -> `3_8` | `test3/classify_nouns_3_3.py` | `test_data/instance_form_sentence.csv` | `classified_nouns_3_3.csv` | `classified_nouns_3_3_zero_values.csv` | `test3/classify_material_3_8.py` | `classified_materials_3_8.csv` |
| `3_4` -> `3_9` | `test3/classify_nouns_3_4.py` | `test_data/instance_form_long.csv` | `classified_nouns_3_4.csv` | `classified_nouns_3_4_zero_values.csv` | `test3/classify_material_3_9.py` | `classified_materials_3_9.csv` |
| `3_5` -> `3_10` | `test3/classify_nouns_3_5.py` | `test_data/instance_form_long_sentence.csv` | `classified_nouns_3_5.csv` | `classified_nouns_3_5_zero_values.csv` | `test3/classify_material_3_10.py` | `classified_materials_3_10.csv` |

Before running a Test3 material script, create its input with `from_subtask1_extract_zero_values.py`. For example:

```bash
python from_subtask1_extract_zero_values.py classified_nouns_3_1.csv classified_nouns_3_1_zero_values.csv
```

The helper keeps only rows where `0/1` is `0`, removes the `0/1` column, and keeps the remaining input headers unchanged. The paired material script reads that helper output by default.

If you run several Test3 cases in the same directory, keep the default filenames per case or pass explicit output filenames so different experiments do not resume from each other's output.

### Helper Script

| Script | Use case | Input | Output |
| --- | --- | --- | --- |
| `from_subtask1_extract_zero_values.py` | Bridge subtask 1 to subtask 2. Subtask 1 is any `classify_nouns*.py` script, which identifies the first target class for that test. The helper keeps only rows where subtask 1 returned `0`, so they can be passed to the paired subtask 2 `classify_material*.py` script. | Any `classify_nouns*.py` output CSV with a `0/1` label column. This applies to Test1, Test2, and Test3 outputs. | Same columns as the input, except the `0/1` column is removed. Only rows with `0/1 = 0` are written. |

## Usage Examples

Run the default Test2 `2_1` -> `2_5` workflow:

```bash
python test2/classify_nouns_2_1.py google/gemini-2.5-flash-lite
python from_subtask1_extract_zero_values.py classified_nouns_2_1.csv classified_nouns_2_1_zero_values.csv
python test2/classify_material_2_5.py google/gemini-2.5-flash-lite
```

Run the other Test2 pairs in the same pattern:

```bash
python test2/classify_nouns_2_2.py google/gemini-2.5-flash-lite
python from_subtask1_extract_zero_values.py classified_nouns_2_2.csv classified_nouns_2_2_zero_values.csv
python test2/classify_material_2_6.py google/gemini-2.5-flash-lite

python test2/classify_nouns_2_3.py google/gemini-2.5-flash-lite
python from_subtask1_extract_zero_values.py classified_nouns_2_3.csv classified_nouns_2_3_zero_values.csv
python test2/classify_material_2_7.py google/gemini-2.5-flash-lite

python test2/classify_nouns_2_4.py google/gemini-2.5-flash-lite
python from_subtask1_extract_zero_values.py classified_nouns_2_4.csv classified_nouns_2_4_zero_values.csv
python test2/classify_material_2_8.py google/gemini-2.5-flash-lite
```

Run the default Test3 `3_1` -> `3_6` workflow:

```bash
python test3/classify_nouns_3_1.py google/gemini-2.5-flash-lite
python from_subtask1_extract_zero_values.py classified_nouns_3_1.csv classified_nouns_3_1_zero_values.csv
python test3/classify_material_3_6.py google/gemini-2.5-flash-lite
```

Run the other Test3 pairs in the same pattern:

```bash
python test3/classify_nouns_3_2.py google/gemini-2.5-flash-lite
python from_subtask1_extract_zero_values.py classified_nouns_3_2.csv classified_nouns_3_2_zero_values.csv
python test3/classify_material_3_7.py google/gemini-2.5-flash-lite

python test3/classify_nouns_3_3.py google/gemini-2.5-flash-lite
python from_subtask1_extract_zero_values.py classified_nouns_3_3.csv classified_nouns_3_3_zero_values.csv
python test3/classify_material_3_8.py google/gemini-2.5-flash-lite

python test3/classify_nouns_3_4.py google/gemini-2.5-flash-lite
python from_subtask1_extract_zero_values.py classified_nouns_3_4.csv classified_nouns_3_4_zero_values.csv
python test3/classify_material_3_9.py google/gemini-2.5-flash-lite

python test3/classify_nouns_3_5.py google/gemini-2.5-flash-lite
python from_subtask1_extract_zero_values.py classified_nouns_3_5.csv classified_nouns_3_5_zero_values.csv
python test3/classify_material_3_10.py google/gemini-2.5-flash-lite
```

Run the default Test1 two-subtask workflow:

```bash
python test1/classify_nouns.py google/gemini-2.5-flash-lite
python from_subtask1_extract_zero_values.py classified_nouns_1_1.csv classified_nouns_1_1_zero_values.csv
python test1/classify_material.py google/gemini-2.5-flash-lite
```

Run the same Test1 workflow with explicit filenames:

```bash
python test1/classify_nouns.py google/gemini-2.5-flash-lite output_nouns.csv --input_csv path/to/lemma_list.csv
python from_subtask1_extract_zero_values.py output_nouns.csv material_input.csv
python test1/classify_material.py google/gemini-2.5-flash-lite output_material.csv --input_csv material_input.csv
```

Run TartuNLP EstLLM through Featherless.ai:

```bash
python test1/classify_nouns_estllm.py
python from_subtask1_extract_zero_values.py classified_nouns_1_1.csv classified_nouns_1_1_zero_values.csv
python test1/classify_material_estllm.py
```

Use `python3` instead of `python` if that is how Python is exposed in your environment.

## Troubleshooting

- **Missing API key**: check that `.env` exists in the directory you are running from, or pass `--env_file`.
- **Input CSV column error**: check the workflow table for the script you are running. Test1 scripts and single-column Test3 scripts require a header row and use the first column as the item to classify. Test2 scripts and multi-column Test3 scripts require the named columns shown in the table.
- **Missing `*_zero_values.csv` input**: material scripts read helper-generated input by default. Run `from_subtask1_extract_zero_values.py` on the paired `classified_nouns*.csv` output before running the material script.
- **No phrases found in material input**: check whether the paired noun output had any rows with `0/1 = 0`. If every row was classified as `1`, the helper output will contain only the header and there is nothing for the material script to classify.
- **Existing output header mismatch**: output headers are copied from the input CSV and then `0/1` is appended. If you reuse an output file from a different input, older script version, or another experiment, choose a new output filename or use `--overwrite`.
- **Unexpected commas in output**: input files may be comma-, semicolon-, or tab-delimited, but classifier outputs are always semicolon-delimited. Open the result as a semicolon-delimited CSV.
- **Interrupted run**: rerun the same command with the same output file; completed valid rows will be skipped.
- **API rate limits or temporary failures**: scripts retry HTTP `429`, `500`, `502`, `503`, and `504` responses up to `--max_retries`.