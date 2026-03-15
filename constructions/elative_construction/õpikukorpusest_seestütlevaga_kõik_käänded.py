import pickle
import pandas as pd

from tqdm import tqdm

PKL_PATH = "keeleoppija_sonaveeb_morph_syntax.df.pkl"
OUT_PATH = "täiendvormid_keeleõpikud_022026_ver3.csv"


def _get_first_ann(span):
    """Return first annotation dict from an EstNLTK span-like object, or {}."""
    try:
        anns = getattr(span, "annotations", None)
        if anns:
            return anns[0]
    except Exception:
        pass
    return {}


def _get_syn_attr(ann: dict, *keys, default=None):
    for k in keys:
        if k in ann and ann[k] is not None:
            return ann[k]
    return default


def _to_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _surface_from_span(word_span, raw_text: str) -> str:
    """
    Robust surface form extraction using base_span char offsets.
    Works even if span doesn't expose `.text`.
    """
    bs = getattr(word_span, "base_span", None)
    if bs is None:
        # fallback: try `.text`
        return getattr(word_span, "text", str(word_span))
    return raw_text[bs.start:bs.end]


# def _is_noun(morph_span) -> bool:
#     """True if any morph analysis has partofspeech == 'S'."""
#     try:
#         for ann in getattr(morph_span, "annotations", []):
#             if ann.get("partofspeech") == "S":
#                 return True
#     except Exception:
#         pass
#     return False

def _is_noun(morph_span) -> bool:
    """True if the *first* morph analysis has partofspeech == 'S'."""
    try:
        anns = getattr(morph_span, "annotations", [])
        if not anns:
            return False
        return anns[0].get("partofspeech") == "S"
    except Exception:
        return False


# def _is_elative_noun(morph_span) -> bool:
#     """
#     True if partofspeech == 'S' and morph 'form' contains case 'el' (elative),
#     e.g. 'sg el' or 'pl el'.
#     """
#     try:
#         for ann in getattr(morph_span, "annotations", []):
#             if ann.get("partofspeech") != "S":
#                 continue
#             form = (ann.get("form") or "").strip()
#             if "el" in form.split():
#                 return True
#     except Exception:
#         pass
#     return False

def _is_elative_noun(morph_span) -> bool:
    """
    True only if the *first* morph annotation is a noun (partofspeech == 'S')
    and its 'form' contains elative case 'el' (e.g. 'sg el', 'pl el').
    """
    try:
        anns = getattr(morph_span, "annotations", [])
        if not anns:
            return False

        ann0 = anns[0]
        if ann0.get("partofspeech") != "S":
            return False

        form = (ann0.get("form") or "").strip()
        return "el" in form.split()
    except Exception:
        return False


def _iter_sentence_word_indices(text_obj):
    """
    Yield lists of word indices belonging to each sentence.
    Uses char offsets from sentence spans + word spans.
    """
    if "sentences" not in text_obj.layers:
        # assume whole text is one "sentence"
        yield list(range(len(text_obj.words)))
        return

    words = list(text_obj.words)
    for sent in text_obj.sentences:
        sent_bs = sent.base_span
        idxs = []
        for i, w in enumerate(words):
            wbs = w.base_span
            if wbs.start >= sent_bs.start and wbs.end <= sent_bs.end:
                idxs.append(i)
        if idxs:
            yield idxs


def extract_adjacent_elative_head_pairs(korpus: pd.DataFrame) -> pd.DataFrame:
    """
    For each sentence(Text) in korpus:
      - take 'level'
      - find adjacent noun pair (noun1, noun2) where:
          * noun1 POS == S and case == el (elative) via morph_analysis
          * noun2 POS == S via morph_analysis
          * noun2 immediately after noun1 (id2 == id1 + 1)
          * noun2 is head of noun1 (head(noun1) == id(noun2))
          * noun1 dependency relation is nmod
    Returns a dataframe with columns: noun1, noun2, sentence, level
    """
    results = []

    for row in tqdm(korpus.itertuples(index=False), total=len(korpus), desc="Scanning"):
        # Adjust if your df uses different column names
        level = getattr(row, "level", None)
        text_obj = getattr(row, "text", None)
        if text_obj is None:
            continue

        # Ensure needed layers exist
        if "morph_analysis" not in text_obj.layers:
            text_obj.tag_layer("morph_analysis")

        # Your corpus uses stanza_syntax, not syntax
        syntax_layer_name = None
        if "stanza_syntax" in text_obj.layers:
            syntax_layer_name = "stanza_syntax"
        elif "syntax" in text_obj.layers:
            syntax_layer_name = "syntax"
        else:
            # try tagging (if available in your setup)
            try:
                text_obj.tag_layer("stanza_syntax")
                syntax_layer_name = "stanza_syntax"
            except Exception:
                # cannot evaluate dependency conditions without syntax
                continue

        raw_text = text_obj.text

        words_layer = text_obj.words
        morph_layer = text_obj.morph_analysis
        syn_layer = getattr(text_obj, syntax_layer_name)

        # Iterate sentence-by-sentence (robust even if each row is already 1 sentence)
        for idxs in _iter_sentence_word_indices(text_obj):
            # Build local token list (word-indexed)
            # We assume morph_layer and syn_layer align with words_layer by index.
            for j in range(len(idxs) - 1):
                i1 = idxs[j]
                i2 = idxs[j + 1]

                morph1 = morph_layer[i1]
                morph2 = morph_layer[i2]
                if not _is_elative_noun(morph1):
                    continue
                if not _is_noun(morph2):
                    continue

                syn1_ann = _get_first_ann(syn_layer[i1])
                syn2_ann = _get_first_ann(syn_layer[i2])

                id1 = _to_int(_get_syn_attr(syn1_ann, "id", "ID", "token_id"))
                id2 = _to_int(_get_syn_attr(syn2_ann, "id", "ID", "token_id"))
                head1 = _to_int(_get_syn_attr(syn1_ann, "head", "Head", "parent", "Parent"))
                deprel1 = _get_syn_attr(syn1_ann, "deprel", "DependencyType", "dep_rel", default="")

                # Conditions:
                # - adjacency by syntactic IDs
                if id1 is None or id2 is None:
                    continue
                if id2 != id1 + 1:
                    continue
                # - noun2 is head of noun1
                if head1 != id2:
                    continue
                # - noun1 dependency type is nmod
                if deprel1 != "nmod":
                    continue

                noun1 = _surface_from_span(words_layer[i1], raw_text)
                noun2 = _surface_from_span(words_layer[i2], raw_text)

                # Sentence string
                # If each row is one sentence, this is just raw_text. Otherwise slice by sentence span.
                # Here we keep it simple + safe:
                sentence_str = raw_text.strip()

                results.append({
                    "fraas": f"{noun1} {noun2}",
                    "lause": sentence_str,
                    "tase": level,
                })

    return pd.DataFrame(results)


def main():
    with open(PKL_PATH, "rb") as f:
        korpus = pickle.load(f)

    out_df = extract_adjacent_elative_head_pairs(korpus)

    # Semicolon-separated CSV, UTF-8
    out_df.to_csv(OUT_PATH, sep=";", index=False, encoding="utf-8")

    print(f"Wrote {len(out_df)} rows to: {OUT_PATH}")


if __name__ == "__main__":
    main()
