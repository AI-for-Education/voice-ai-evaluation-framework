import pandas as pd

from egra_eval.report.summarize import summary_phonological_by_category


def make_row(audio_type: str):
    # minimal row with required phonological columns present
    base = {
        "audio_type": audio_type,
        # provide zeroed TP/FP/FN columns so aggregation runs
        "S_TP": 0.0,
        "S_FP": 0.0,
        "S_FN": 0.0,
        "D_TP": 0.0,
        "D_FP": 0.0,
        "D_FN": 0.0,
        "I_TP": 0.0,
        "I_FP": 0.0,
        "I_FN": 0.0,
    }
    return base


def test_summary_has_expected_category_keys():
    audio_types = [
        "passage_num11",
        "full_syllable1_grid",
        "iso_syllable1_1",
        "full_nonword_grid",
        "iso_non_word1",
        "full_letter_grid",
        "iso_letter_1",
    ]

    rows = [make_row(a) for a in audio_types]
    df = pd.DataFrame(rows)

    res = summary_phonological_by_category(df)

    # Expect overall key and keys for some category combos
    assert "__OVERALL__" in res
    assert "passage_passage" in res
    assert "syllables_grid" in res
    assert "nonwords_grid" in res
    assert "letters_grid" in res
*** End Patch