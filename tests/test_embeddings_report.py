from scripts.embeddings_network import token_coverage


def test_coverage_is_one_when_every_token_is_known():
    sentences = [["free", "call"], ["call", "now"]]
    assert token_coverage(sentences, {"free", "call", "now"}) == 1.0


def test_coverage_counts_occurrences_not_distinct_words():
    sentences = [["aa", "aa", "aa", "rare"]]
    assert token_coverage(sentences, {"aa"}) == 0.75


def test_coverage_of_empty_corpus_is_zero():
    assert token_coverage([], set()) == 0.0
