#!/usr/bin/env python3

from collections import Counter
import random

from methods.data_extraction.helper_methods import *


def lemmatize_phrase(members_list: list) -> str:
    """
    Helper method for creating a lemmatized phrase string from phrase members list.
    """
    phrase_lemma = ""

    for idx, stanza_word in enumerate(members_list):

        if idx == len(members_list)-1:
            phrase_lemma+=stanza_word.morph_analysis.lemma[0]
        else:
            phrase_lemma+=f"{stanza_word.text} "

    return phrase_lemma


def extract_example_phrases(lemma: str, member_name: str, sentences_list: list, matches: list[list[dict]]) -> list[list]:
    """
    Extracting example phrases.
    """
    found_examples: list[list] = []

    for idx1, sentence in enumerate(sentences_list):

        for idx2, match in enumerate(matches[idx1]):

            member = get_conx_member(match, member_name)

            if member.morph_analysis.lemma[0] == lemma:
                phrase = []

                for idx3, stanza_word in enumerate(sentence):

                    if stanza_word in match.values():
                        phrase.append(stanza_word)

                found_examples.append(phrase)

    return found_examples


def extract_example_sentences(phrase_lemma: str, n_examples: int, sentences_list: list, matches: list[list[dict]], randomized=False) -> list[list]:
    """
    Extracting and returning n example sentences for given phrase lemma.
    """
    found_examples: list[list] = []

    for idx1, sentence in enumerate(sentences_list):

        for idx2, match in enumerate(matches[idx1]):

            phrase = []

            for idx3, stanza_word in enumerate(sentence):

                if stanza_word in match.values():
                    phrase.append(stanza_word)

            found_phrase_lemma = lemmatize_phrase(phrase)

            if found_phrase_lemma == phrase_lemma:
                found_examples.append(sentence)

    if randomized:
        random.shuffle(found_examples)

    return found_examples[:n_examples]


def extract_n_most_common_phrases(lemma: str, n_most_common: int, member_name: str, sentences_list: list, matches: list[list[dict]]) -> list:
    """
    Extracting and returning n most common phrases and their lemmas.
    """

    all_examples: list[list] = extract_example_phrases(lemma, member_name, sentences_list, matches)

    all_examples_lemmatized: list[str] = [lemmatize_phrase(example) for idx, example in enumerate(all_examples)]

    most_common = Counter(all_examples_lemmatized).most_common(n_most_common)

    most_common_list_str = [el[0] for el in most_common]

    most_common_list_stanza = [example for idx, example in enumerate(all_examples) if lemmatize_phrase(example) in most_common_list_str]
    
    return most_common_list_str, most_common_list_stanza


