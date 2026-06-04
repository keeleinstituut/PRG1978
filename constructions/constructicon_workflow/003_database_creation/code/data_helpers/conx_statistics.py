#!/usr/bin/env python3

# Calculating construction member's raw frequency and salience

# Imports
import csv
import math
from collections import Counter

# CSV file of corpus form frequencies as input data
CORPUS_FORMS_FILE_PATH: str = "../data/opikukorpus_form_counts_ekilex_tags_disambig.csv"

CORPUS_FORM_COUNTS: dict[str, int] = {}

# Reading forms and frequencies from CSV
with open(CORPUS_FORMS_FILE_PATH, mode='r') as infile:
    reader = csv.reader(infile)
    header = next(reader, None) 
    for row in reader:
        if not row or len(row) < 2:
            continue
        CORPUS_FORM_COUNTS[row[0]] = int(row[1])


def summarize_form_counts(form_counts_dict: dict[str, int]) -> dict[str, int]:
    """
    Helper method to summarize the counts of different forms that share the same lemma.

    :param form_counts_dict: dictionary that has form codes as keys and their frequencies as values
    :type form_counts_dict: dict[str, int]
    """
    summarized_form_counts: dict[str, int] = {}

    for form in form_counts_dict.keys():
        lemma = form.split("_")[0]
        summarized_form_counts[lemma] = 0

    for form, count in form_counts_dict.items():
        lemma = form.split("_")[0]
        summarized_form_counts[lemma] += count
    
    return summarized_form_counts


def calculate_LL(N: int, f1: int, f2: int, O: int, base10: bool = True) -> float:
    """
    Calculating signed log-likelihood (G^2) statistic.
    
    :param N: corpus size (all forms)
    :type N: int
    :param f1: form's raw frequency over corpus
    :type f1: int
    :param f2: construction's raw frequency over corpus
    :type f2: int
    :param O: form's raw frequency over construction
    :type O: int
    :param base10: by default uses log10; if False, then natural logarithm is used
    :type base10: bool
    :return: Signed log-likelihood (LL). Pos => attraction; neg => repulsion
    :rtype: float
    """

    # Observed
    O11 = O
    O12 = f1 - O
    O21 = f2 - O
    O22 = N - f1 - f2 + O

    if min(O11, O12, O21, O22) < 0:
        raise ValueError("Contingency cells can't be negative. Check f1, f2, O, and N.")
    if N <= 0:
        raise ValueError("Corpus size N must be positive.")
    
    # Expected
    E11 = (f1 * f2) / N
    E12 = (f1 * (N - f2)) / N
    E21 = ((N - f1) * f2) / N
    E22 = ((N - f1) * (N - f2)) / N

    logf = math.log10 if base10 else math.log

    # Helper: contribution of a single cell (0 if observed is 0 or expected is 0)
    def term(o, e):
        if o == 0 or e == 0:
            return 0.0
        return o * logf(o / e)
    
    # Signed G^2
    sign = -1.0 if (O11 - E11) < 0 else 1.0
    G2 = sign * 2.0 * (term(O11, E11) + term(O12, E12) + term(O21, E21) + term(O22, E22))

    return G2


def get_n_most_common_by_freq(all_member_forms: list[str], n_most_common: int) -> dict[str, int]:
    """
    Finds the raw frequencies of forms that appear at a specific position of given construction, returns n most common forms.
    
    :param all_member_forms: list of all forms that appear at a specific position of given construction
    :type all_member_forms: list[str]
    :param n_most_common: number of most common forms to be returned
    :type n_most_common: int
    :return: dictionary of n most common forms with their raw frequencies. Ordered by frequency in descending order
    :rtype: dict[str, int]
    """
    most_common = Counter(all_member_forms).most_common(n_most_common)
    most_common_dict: dict[str, int] = {el[0]: el[1] for el in most_common}

    return summarize_form_counts(most_common_dict)


def get_n_most_attracted(all_member_forms: list[str], conx_freq: int, n_most_attracted: int, corpus_form_counts: dict[str, int] = CORPUS_FORM_COUNTS) -> list[tuple[str, float]]:
    """
    Finds the LL-scores of forms that appear at a specific position of given construction, returns n most attracted forms.
    
    :param all_member_forms: list of all forms that appear at a specific position of given construction. Presented as lemma/s_form/s_partofspeech/es (e.g. jultuma|jultunud|jultunud|jultunud_nud||sg n|pl n_V|A|A|A)
    :type all_member_forms: list[str]
    :param conx_freq: construction's raw frequency over corpus
    :type conx_freq: int
    :param n_most_attracted: number of most attracted forms to be returned
    :type n_most_attracted: int
    :param corpus_form_counts: frequency list of all forms in corpus. Forms are presented as lemma/s_form/s_partofspeech/es (nt uks_sg all_S: 2)
    :type corpus_form_counts: dict[str, int]
    :return: dictionary of n most attracted forms with their LL-scores. Ordered by LL-score in descending order
    :rtype: list[tuple[str, int]]
    """
    member_form_counts: dict[str, int] = dict(Counter(all_member_forms))
    summarized_member_form_counts: dict[str, int] = summarize_form_counts(member_form_counts)

    corpus_form_counts_subset: dict[str, int] = {}

    for form, count in corpus_form_counts.items():
        if form in all_member_forms:
            corpus_form_counts_subset[form] = count

    summarized_corpus_form_counts: dict[str, int] = summarize_form_counts(corpus_form_counts_subset)

    corpus_size = sum(corpus_form_counts.values())

    form_ll_scores: list = []

    for form, form_freq in summarized_member_form_counts.items():
        form_ll_score = calculate_LL(corpus_size, summarized_corpus_form_counts[form], conx_freq, form_freq)
        form_ll_scores.append(tuple([form, form_ll_score]))

    # Only forms with positive LL scores are kept, as the search is for attracted, not repulsed construction members
    form_ll_scores = [score for score in form_ll_scores if score[1] > 0]
    sorted_form_ll_scores = sorted(form_ll_scores, key=lambda x: x[1], reverse=True)
    sorted_form_ll_scores_top_n = sorted_form_ll_scores[:n_most_attracted]

    sorted_form_ll_scores_top_n_dct: dict[str, float] = {item[0]: item[1] for item in sorted_form_ll_scores_top_n}

    return sorted_form_ll_scores_top_n_dct


def get_statistics(all_member_forms: list[str], conx_freq: int, n_most_common: int, n_most_attracted: int, corpus_form_counts: dict[str, int] = CORPUS_FORM_COUNTS) -> dict[str, list[int, float]]:
    """
    Finds n most common and n most attracted word forms, calculates both raw frequencies and LL-scores of these forms.
    Returns these word forms with their raw frequencies and LL-scores.
    
    :param all_member_forms: list of all forms that appear at a specific position of given construction. Presented as lemma/s_form/s_partofspeech/es (e.g. jultuma|jultunud|jultunud|jultunud_nud||sg n|pl n_V|A|A|A)
    :type all_member_forms: list[str]
    :param conx_freq: construction's raw frequency over corpus
    :type conx_freq: int
    :param n_most_common: number of most common forms to be returned
    :type n_most_common: int
    :param n_most_attracted: number of most attracted forms to be returned
    :type n_most_attracted: int
    :return: dictionaries of n most common ja n most attracted forms with their raw frequencies and LL-scores. Duplicates are removed from most common forms
    :rtype: dict[str, list[int, float]]
    """
    most_common_form_scores: dict[str, list[int, float]] = {}
    most_attracted_form_scores: dict[str, list[int, float]] = {}

    # Get n most common and n most attracted word forms
    most_common = get_n_most_common_by_freq(all_member_forms, n_most_common)
    most_attracted = get_n_most_attracted(all_member_forms, conx_freq, n_most_attracted)

    # Calculate frequencies for n most attracted word forms
    member_form_counts = dict(Counter(all_member_forms))
    summarized_member_form_counts: dict[str, int] = summarize_form_counts(member_form_counts)

    most_attracted_form_counts = {key: summarized_member_form_counts[key] for key in most_attracted.keys()}

    # Find and summarize relevant form counts over whole corpus
    corpus_form_counts_subset: dict[str, int] = {}

    for form, count in corpus_form_counts.items():
        if form in all_member_forms:
            corpus_form_counts_subset[form] = count

    summarized_corpus_form_counts: dict[str, int] = summarize_form_counts(corpus_form_counts_subset)

    # Calculate LL-scores for n most common word forms
    corpus_size = sum(corpus_form_counts.values())
    most_common_ll_scores = {form: calculate_LL(corpus_size, summarized_corpus_form_counts[form], conx_freq, freq) for form, freq in most_common.items()}

    # Fill the dictionary of most common word forms with frequencies and LL-scores
    # Ignore forms that are already present among most attracted forms
    for form, freq in most_common.items():
        if form not in most_attracted_form_counts.keys():
            most_common_form_scores[form] = [freq, most_common_ll_scores[form]]
    
    # Fill the dictionary of most attracted word forms with frequencies and LL-scores
    for key, value in most_attracted.items():
        most_attracted_form_scores[key] = [most_attracted_form_counts[key], value]

    return most_common_form_scores, most_attracted_form_scores










