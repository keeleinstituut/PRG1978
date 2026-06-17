# Estonian prompts

This directory contains the original Estonian prompt templates used for LLM semantic annotation.

The complete prompt sets are split by test/experiment:

- `test1_prompts.md`: word-level noun and lemma classification prompts from `code/LLM_API/Test1`.
- `test2_prompts.md`: phrase-level component/head relation prompts from `code/LLM_API/Test2`.
- `test3_prompts.md`: phrase-level construction-definition prompts from `code/LLM_API/Test3`.

Each prompt file lists the source script and includes both message parts sent to the model:

- `SYSTEM_PROMPT`: the system message.
- `user_prompt`: the user message template. Placeholders such as `{word}`, `{sentence}`, `{comp1_form}`, or `{comp2_form}` are filled by the script at runtime.

`Experiment1_few_shot_prompt.txt` is the earlier standalone few-shot prompt note and is kept for reference. The same few-shot prompts are also included in `test1_prompts.md` together with the other Test1 prompts.
