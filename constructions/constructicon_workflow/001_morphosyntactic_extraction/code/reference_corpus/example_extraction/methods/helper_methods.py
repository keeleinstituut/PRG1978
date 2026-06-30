#!/usr/bin/env python3


import os
from estnltk_core.converters.json_importer import json_to_text
from estnltk.taggers import VabamorfDisambiguator

from methods.tagmap import TAG_MAP


def read_texts_from_json(file_path: str) -> list:
    """
    Read and convert all json files in folder to EstNLTK Text-objects.
    """
    texts = []
    for fn in os.listdir(file_path):
        texts.append(json_to_text(file=f"{file_path}{fn}"))
    return texts


def disambiguate_vm_morph(text):
    """
    Disambiguating ambiguous Vabamorf analysis.
    """
    vabamorf_disambiguator = VabamorfDisambiguator()
    
    return vabamorf_disambiguator.retag(text)


def get_conx_head(sentence_words: list, conx_ids: list, conx_headids: list):
    """
    Finding and returning head of construction example.
    """
    conx_head_id = None

    for idx, headid in enumerate(conx_headids):
        if headid not in conx_ids:
            conx_head_id = conx_ids[idx]

    if conx_head_id:
        for stanza_word in sentence_words:
            if stanza_word.id == conx_head_id:
                return stanza_word
    else:
        return None
    

def get_form_code(stanza_word):
    """
    Converting Vabamorf tags to EKILEX tags and returning stanza word as a form code.
    """
    lemma_part = "|".join(stanza_word.morph_analysis.lemma)
    form_part = "|".join([TAG_MAP[form] for form in stanza_word.morph_analysis.form])
    pos_part = "|".join(stanza_word.morph_analysis.partofspeech)

    return lemma_part+"_"+form_part+"_"+pos_part


def display_example(idx: int, sentences_words: list, conxs_ids: list):
    """
    Displaying by index an example sentence, the construction example and construction example's head member.
    """
    for conx_ids in conxs_ids[idx]:
        ex = sentences_words[idx]
        sorted_conx_ids = sorted(conx_ids)
        sorted_conx_headids = [word.head for word in ex if word.id in sorted_conx_ids]
        head_id = [i for i, conx_id in enumerate(sorted_conx_headids) if conx_id not in sorted_conx_ids][0]
        head_member_id = sorted_conx_ids[head_id]

        print(" ".join([word.text for word in ex]))
        print(" ".join([word.text for word in ex if word.id in sorted_conx_ids]))
        print(" ".join([word.text for word in ex if word.id == head_member_id]))
        print()


def get_subtree(stanza_word, subtree=None):
    """
    Finding all children of given stanza word from syntax tree recursively. Returns given stanza word and all it's children as a list
    """
    if subtree is None:
        subtree = []

    subtree.append(stanza_word)
    
    children = stanza_word.children

    for child in children:
        get_subtree(child, subtree)

    return subtree
    