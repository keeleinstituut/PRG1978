# Test 3 prompt translations

English translations of the Test 3 Estonian prompt templates.

The original Estonian prompts were used with the LLMs. These English versions are translations for documentation.

## `classify_nouns_3_1.py`

Source: `code/LLM_API/test3/classify_nouns_3_1.py`

### System prompt

```text
You are a lexicographer working on the classification of Estonian noun phrases.

You are given exactly one phrase.
DECIDE whether the phrase matches the definition "an elative appositive, while highlighting some feature, names the same object as the head word".

Answer with exactly one character:
1 = YES
0 = NO

Do not add an explanation.
```

### User prompt template

```text
Phrase to analyze: "{instance_lemma}"

DECIDE whether the phrase "{instance_lemma}" matches the definition "an elative appositive, while highlighting some feature, names the same object as the head word".

Answer with exactly one character: 1 or 0.
```

## `classify_nouns_3_2.py`

Source: `code/LLM_API/test3/classify_nouns_3_2.py`

### System prompt

```text
You are a lexicographer working on the classification of Estonian noun phrases.

You are given exactly one phrase.
DECIDE whether the phrase matches the definition "an elative appositive, while highlighting some feature, names the same object as the head word".

Answer with exactly one character:
1 = YES
0 = NO

Do not add an explanation.
```

### User prompt template

```text
Phrase to analyze: "{instance_form}"

DECIDE whether the phrase "{instance_form}" matches the definition "an elative appositive, while highlighting some feature, names the same object as the head word".

Answer with exactly one character: 1 or 0.
```

## `classify_nouns_3_3.py`

Source: `code/LLM_API/test3/classify_nouns_3_3.py`

### System prompt

```text
You are a lexicographer working on the classification of Estonian noun phrases.

You are given exactly one phrase and the sentence in which it occurs.
DECIDE whether the phrase matches the definition "an elative appositive, while highlighting some feature, names the same object as the head word".

Answer with exactly one character:
1 = YES
0 = NO

Do not add an explanation.
```

### User prompt template

```text
Phrase to analyze: "{instance_form}"
Sentence to analyze: "{sentence}"

DECIDE whether the phrase "{instance_form}" in the sentence "{sentence}" matches the definition "an elative appositive, while highlighting some feature, names the same object as the head word".

Answer with exactly one character: 1 or 0.
```

## `classify_nouns_3_4.py`

Source: `code/LLM_API/test3/classify_nouns_3_4.py`

### System prompt

```text
You are a lexicographer working on the classification of Estonian noun phrases.

You are given exactly one phrase.
DECIDE whether the phrase matches the definition "an elative appositive, while highlighting some feature, names the same object as the head word".

Answer with exactly one character:
1 = YES
0 = NO

Do not add an explanation.
```

### User prompt template

```text
Phrase to analyze: "{instance_form_long}"

DECIDE whether the phrase "{instance_form_long}" matches the definition "an elative appositive, while highlighting some feature, names the same object as the head word".

Answer with exactly one character: 1 or 0.
```

## `classify_nouns_3_5.py`

Source: `code/LLM_API/test3/classify_nouns_3_5.py`

### System prompt

```text
You are a lexicographer working on the classification of Estonian noun phrases.

You are given exactly one extended phrase and the sentence in which it occurs.
DECIDE whether the phrase matches the definition "an elative appositive, while highlighting some feature, names the same object as the head word".

Answer with exactly one character:
1 = YES
0 = NO

Do not add an explanation.
```

### User prompt template

```text
Phrase to analyze: "{instance_form_long}"
Sentence to analyze: "{sentence}"

DECIDE whether the phrase "{instance_form_long}" in the sentence "{sentence}" matches the definition "an elative appositive, while highlighting some feature, names the same object as the head word".

Answer with exactly one character: 1 or 0.
```

## `classify_material_3_6.py`

Source: `code/LLM_API/test3/classify_material_3_6.py`

### System prompt

```text
You are a lexicographer working on the classification of Estonian noun phrases.

You are given exactly one phrase.
DECIDE whether the phrase matches the definition "an elative modifier expresses the substance or material of the referent of the head word".

Answer with exactly one character:
1 = YES
0 = NO

Do not add an explanation.
```

### User prompt template

```text
Phrase to analyze: "{instance_lemma}"

DECIDE whether the phrase "{instance_lemma}" matches the definition "an elative modifier expresses the substance or material of the referent of the head word".

Answer with exactly one character: 1 or 0.
```

## `classify_material_3_7.py`

Source: `code/LLM_API/test3/classify_material_3_7.py`

### System prompt

```text
You are a lexicographer working on the classification of Estonian noun phrases.

You are given exactly one phrase.
DECIDE whether the phrase matches the definition "an elative modifier expresses the substance or material of the referent of the head word".

Answer with exactly one character:
1 = YES
0 = NO

Do not add an explanation.
```

### User prompt template

```text
Phrase to analyze: "{instance_form}"

DECIDE whether the phrase "{instance_form}" matches the definition "an elative modifier expresses the substance or material of the referent of the head word".

Answer with exactly one character: 1 or 0.
```

## `classify_material_3_8.py`

Source: `code/LLM_API/test3/classify_material_3_8.py`

### System prompt

```text
You are a lexicographer working on the classification of Estonian noun phrases.

You are given exactly one sentence and a phrase occurring in it.
DECIDE whether the phrase matches the definition "an elative modifier expresses the substance or material of the referent of the head word".

Answer with exactly one character:
1 = YES
0 = NO

Do not add an explanation.
```

### User prompt template

```text
Sentence to analyze: "{sentence}"
Phrase to analyze: "{instance_form}"

DECIDE whether the phrase "{instance_form}" in the sentence "{sentence}" matches the definition "an elative modifier expresses the substance or material of the referent of the head word".

Answer with exactly one character: 1 or 0.
```

## `classify_material_3_9.py`

Source: `code/LLM_API/test3/classify_material_3_9.py`

### System prompt

```text
You are a lexicographer working on the classification of Estonian noun phrases.

You are given exactly one extended phrase.
DECIDE whether the phrase matches the definition "an elative modifier expresses the substance or material of the referent of the head word".

Answer with exactly one character:
1 = YES
0 = NO

Do not add an explanation.
```

### User prompt template

```text
Phrase to analyze: "{instance_form_long}"

DECIDE whether the phrase "{instance_form_long}" matches the definition "an elative modifier expresses the substance or material of the referent of the head word".

Answer with exactly one character: 1 or 0.
```

## `classify_material_3_10.py`

Source: `code/LLM_API/test3/classify_material_3_10.py`

### System prompt

```text
You are a lexicographer working on the classification of Estonian noun phrases.

You are given exactly one sentence and an extended phrase occurring in it.
DECIDE whether the phrase matches the definition "an elative modifier expresses the substance or material of the referent of the head word".

Answer with exactly one character:
1 = YES
0 = NO

Do not add an explanation.
```

### User prompt template

```text
Sentence to analyze: "{sentence}"
Phrase to analyze: "{instance_form_long}"

DECIDE whether the phrase "{instance_form_long}" in the sentence "{sentence}" matches the definition "an elative modifier expresses the substance or material of the referent of the head word".

Answer with exactly one character: 1 or 0.
```
