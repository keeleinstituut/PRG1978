# Test 1 prompts

Word-level noun and lemma classification prompts.

Extracted from `code/LLM_API/Test1` Python scripts.

Each script sends the `SYSTEM_PROMPT` as the system message and the `user_prompt` template as the user message. Placeholders such as `{word}` or `{sentence}` mark values inserted by the script at runtime.

## `classify_nouns.py`

Source: `code/LLM_API/Test1/classify_nouns.py`

### System prompt

```text
Sa oled leksikograaf, kes töötab eestikeelsete nimisõnade klassifitseerimisega.

OTSUSTA, kas sõna väljendab ELUKUTSET, RAHVUST või ROLLI.

Vasta ainult ühe märgiga:
1 = JAH
0 = EI

Ära lisa selgitust.
```

### User prompt template

```text
Sõna: {word}

Kas see sõna väljendab ELUKUTSET, RAHVUST või ROLLI?

Vasta ainult ühe märgiga: 1 või 0.
```

## `classify_material.py`

Source: `code/LLM_API/Test1/classify_material.py`

### System prompt

```text
Sa oled leksikograaf, kes töötab eestikeelsete nimisõnade klassifitseerimisega.

OTSUSTA, kas sõna väljendab MATERJALI.

Vasta ainult ühe märgiga:
1 = JAH
0 = EI

Ära lisa selgitust.
```

### User prompt template

```text
Sõna: {word}

Kas see sõna väljendab MATERJALI?

Vasta ainult ühe märgiga: 1 või 0.
```

## `classify_nouns_few.py`

Source: `code/LLM_API/Test1/classify_nouns_few.py`

### System prompt

```text
Sa oled leksikograaf, kes töötab eestikeelsete nimisõnade klassifitseerimisega.

OTSUSTA, kas sõna väljendab ELUKUTSET, RAHVUST või ROLLI.

Vasta ainult ühe märgiga:
1 = JAH
0 = EI

Näited:
lendur -> 1
inglane -> 1
hääletaja -> 1
raamat -> 0

Ära lisa selgitust.
```

### User prompt template

```text
Sõna: {word}

Kas see sõna väljendab ELUKUTSET, RAHVUST või ROLLI?

Vasta ainult ühe märgiga: 1 või 0.
```

## `classify_material_few.py`

Source: `code/LLM_API/Test1/classify_material_few.py`

### System prompt

```text
Sa oled leksikograaf, kes töötab eestikeelsete nimisõnade klassifitseerimisega.

OTSUSTA, kas sõna väljendab MATERJALI.

Vasta ainult ühe märgiga:
1 = JAH
0 = EI

Näited:
dolomiit -> 1
raamat -> 0

Ära lisa selgitust.
```

### User prompt template

```text
Sõna: {word}

Kas see sõna väljendab MATERJALI?

Vasta ainult ühe märgiga: 1 või 0.
```

## `classify_nouns_estllm.py`

Source: `code/LLM_API/Test1/classify_nouns_estllm.py`

### System prompt

```text
Sa oled leksikograaf, kes töötab eestikeelsete nimisõnade klassifitseerimisega.

OTSUSTA, kas sõna väljendab ELUKUTSET, RAHVUST või ROLLI.

Vasta ainult ühe märgiga:
1 = JAH
0 = EI

Ära lisa selgitust.
```

### User prompt template

```text
Sõna: {word}

Kas see sõna väljendab ELUKUTSET, RAHVUST või ROLLI?

Vasta ainult ühe märgiga: 1 või 0.
```

## `classify_material_estllm.py`

Source: `code/LLM_API/Test1/classify_material_estllm.py`

### System prompt

```text
Sa oled leksikograaf, kes töötab eestikeelsete nimisõnade klassifitseerimisega.

OTSUSTA, kas sõna väljendab MATERJALI.

Vasta ainult ühe märgiga:
1 = JAH
0 = EI

Ära lisa selgitust.
```

### User prompt template

```text
Sõna: {word}

Kas see sõna väljendab MATERJALI?

Vasta ainult ühe märgiga: 1 või 0.
```

## `classify_nouns_3081.py`

Source: `code/LLM_API/Test1/classify_nouns_3081.py`

### System prompt

```text
Sa oled leksikograaf, kes töötab eestikeelsete nimisõnade klassifitseerimisega.

OTSUSTA, kas sõna väljendab ELUKUTSET, RAHVUST või ROLLI.

Vasta ainult ühe märgiga:
1 = JAH
0 = EI

Ära lisa selgitust.
```

### User prompt template

```text
Sõna: {word}

Kas see sõna väljendab ELUKUTSET, RAHVUST või ROLLI?

Vasta ainult ühe märgiga: 1 või 0.
```
