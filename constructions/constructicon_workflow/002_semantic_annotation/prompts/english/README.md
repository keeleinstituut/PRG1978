# English prompt translations

This directory contains English translations of the Estonian prompt templates used for LLM semantic annotation.

NB! The LLMs were given the original Estonian prompts. These English versions are provided for documentation and readability only.

The complete prompt translations are split by test/experiment:

- `test1_prompts.md`: translations of word-level noun and lemma classification prompts from `code/LLM_API/test1`.
- `test2_prompts.md`: translations of phrase-level component/head relation prompts from `code/LLM_API/test2`.
- `test3_prompts.md`: translations of phrase-level construction-definition prompts from `code/LLM_API/test3`.

Each prompt file lists the source script and includes translated versions of both message parts:

- `SYSTEM_PROMPT`: the system message.
- `user_prompt`: the user message template. Placeholders such as `{word}`, `{sentence}`, `{comp1_form}`, or `{comp2_form}` are preserved from the original scripts.

The `legacy` folder contains preliminary prompt translations from early development stages. These prompts were not used in the article.
