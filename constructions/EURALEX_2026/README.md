# Prompts Used in the Project

This folder contains the prompt templates used in the project **Semantic Clustering of Constructional Collexemes Using LLMs**.

## Project overview

The project investigates whether large language models can support the semantic clustering and labelling of constructional collexemes for constructicographic use. The case study focuses on the **Estonian Nominal Quantifier Construction (ENQC)** and its measure-noun slot fillers.

The prompts were used to test how well an LLM can group Estonian quantifier-related nouns into semantically meaningful clusters and assign interpretable labels to those clusters.

## Data

The experiments use two datasets:

- **Primary dataset:** 189 nouns
- **Scaling dataset:** 405 nouns

The data comes from:

- EKI Combined Dictionary
- Balanced Corpus of Estonian

## Prompt types

Five input types were tested in the informed free sorting task:

1. **Lemma-only prompts**  
   The model receives only the noun lemmas.

2. **Corpus phrase prompts**  
   The model receives example quantifier phrases from corpus data.

3. **Sentence prompts**  
   The model receives full corpus sentences containing the target collexemes.

4. **Dictionary definition prompts**  
   The model receives dictionary definitions for the target lemmas.

5. **Definition + usage example prompts**  
   The model receives both dictionary definitions and usage examples.

Additional prompts were used for:

- closed categorisation into predefined gold-standard clusters;
- labelling generated clusters;
- scaling the method from 190 to 405 nouns.

## Evaluation

The outputs were evaluated against a human-annotated gold standard containing **13 semantic clusters**.

The evaluation used:

- **Adjusted Mutual Information (AMI)**
- **Adjusted Rand Index (ARI)**
- **manual label-quality rating**

## Purpose of the prompts

The prompts are designed to test whether LLMs can provide a useful first-pass semantic organisation of constructional collexemes. The intended workflow is **human-in-the-loop**: the LLM proposes clusters and labels, while a human lexicographer reviews, corrects, and finalises the classification.

## Notes

The prompt files are intended for reproducibility and comparison between different input conditions. Each prompt should be treated as part of the experimental setup, since changes in wording, context, or input format may affect the resulting clusters and labels.
