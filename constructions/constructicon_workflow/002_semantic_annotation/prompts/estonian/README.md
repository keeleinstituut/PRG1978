# Estonian prompts

This directory contains the original Estonian prompt templates used for LLM semantic annotation.

The complete prompt sets are split by test/experiment:

- `test1_prompts.md`: word-level noun and lemma classification prompts from `code/LLM_API/test1`.
- `test2_prompts.md`: phrase-level component/head relation prompts from `code/LLM_API/test2`.
- `test3_prompts.md`: phrase-level construction-definition prompts from `code/LLM_API/test3`.

Each prompt file lists the source script and includes both message parts sent to the model:

- `SYSTEM_PROMPT`: the system message.
- `user_prompt`: the user message template. Placeholders such as `{word}`, `{sentence}`, `{comp1_form}`, or `{comp2_form}` are filled by the script at runtime.
