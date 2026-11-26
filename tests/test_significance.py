import numpy as np
import pytest

from pepcopilot.eval.significance import bootstrap_ci, mcnemar_exact, paired_bootstrap


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    x = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    mean, lo, hi = bootstrap_ci(x, n_boot=2000, seed=1)
    assert mean == pytest.approx(0.7)
    assert lo <= mean <= hi
    assert 0.0 <= lo and hi <= 1.0
    assert bootstrap_ci(x, n_boot=2000, seed=1) == (mean, lo, hi)


def test_bootstrap_ci_constant_sample_has_zero_width():
    assert bootstrap_ci([0.5] * 8, n_boot=500) == (0.5, 0.5, 0.5)


def test_bootstrap_ci_shrinks_with_more_questions():
    rng = np.random.default_rng(0)
    small = rng.integers(0, 2, 20)
    big = np.tile(small, 10)          # same mean, 10x more questions
    _, lo_s, hi_s = bootstrap_ci(small, n_boot=2000)
    _, lo_b, hi_b = bootstrap_ci(big, n_boot=2000)
    assert (hi_b - lo_b) < (hi_s - lo_s)


def test_paired_bootstrap_identical_systems():
    a = [1, 0, 1, 1]
    assert paired_bootstrap(a, a) == (0.0, 0.0, 0.0, 1.0)


def test_paired_bootstrap_clear_winner_is_significant():
    a, b = [1.0] * 30, [0.0] * 25 + [1.0] * 5
    diff, lo, hi, p = paired_bootstrap(a, b, n_boot=2000)
    assert diff == pytest.approx(25 / 30)
    assert lo > 0 and p < 0.01


def test_paired_bootstrap_one_question_gap_is_not_significant():
    # the README case: Hit@5 0.962 vs 0.942 on 52 questions = one question apart
    a = [1.0] * 50 + [0.0] * 2
    b = [1.0] * 49 + [0.0] * 3
    _, lo, hi, p = paired_bootstrap(a, b, n_boot=2000)
    assert lo <= 0 <= hi and p > 0.05


def test_paired_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [1])


def test_mcnemar_exact_known_values():
    a = [True] * 12 + [False] * 2 + [True] * 10
    b = [False] * 12 + [True] * 2 + [True] * 10
    n01, n10, p = mcnemar_exact(a, b)
    assert (n01, n10) == (12, 2)
    # 2 * P(X <= 2), X ~ Bin(14, 1/2) = 2 * (1 + 14 + 91) / 2**14
    assert p == pytest.approx(2 * 106 / 2 ** 14)


def test_mcnemar_exact_no_discordant_pairs():
    assert mcnemar_exact([True, False], [True, False]) == (0, 0, 1.0)
