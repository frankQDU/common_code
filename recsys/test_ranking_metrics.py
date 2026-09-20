"""对照《排序评估指标手册》的数值例与边界条件。"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recsys.ranking_metrics import (
    HANDBOOK_EXAMPLE,
    MetricUndefinedError,
    average_precision,
    dcg_at_k,
    gauc_score,
    handbook_example_report,
    mean_average_precision,
    mean_ndcg,
    mean_reciprocal_rank,
    ndcg_at_k,
    reciprocal_rank,
    roc_auc_score,
)


LOG2_3 = math.log2(3.0)


class TestHandbookExample(unittest.TestCase):
    """手册第 6 章贯通例：用户 A 6 条、用户 B 3 条。"""

    @classmethod
    def setUpClass(cls):
        cls.report = handbook_example_report()
        cls.y = HANDBOOK_EXAMPLE["y_true"]
        cls.s = HANDBOOK_EXAMPLE["y_score"]
        cls.g = HANDBOOK_EXAMPLE["group_id"]

    def test_per_user_auc(self):
        self.assertAlmostEqual(self.report["auc_A"], 7 / 8)
        self.assertAlmostEqual(self.report["auc_B"], 1 / 2)

    def test_global_auc_is_not_gauc(self):
        self.assertAlmostEqual(self.report["auc_global"], 13 / 18)
        self.assertNotAlmostEqual(self.report["auc_global"], self.report["gauc"]["impression"])

    def test_gauc_weighting_schemes(self):
        gauc = self.report["gauc"]
        self.assertAlmostEqual(gauc["uniform"], 11 / 16)  # (7/8 + 1/2) / 2
        self.assertAlmostEqual(gauc["impression"], 3 / 4)  # (6*(7/8)+3*(1/2)) / 9
        self.assertAlmostEqual(gauc["click"], 3 / 4)  # (2*(7/8)+1*(1/2)) / 3
        self.assertAlmostEqual(gauc["pair"], 4 / 5)  # (8*(7/8)+2*(1/2)) / 10

    def test_ndcg_binary_exp_equals_linear(self):
        # 二值标签下 2^r-1 = r
        ndcg_a_exp = ndcg_at_k(self.y[self.g == "A"], self.s[self.g == "A"], gain="exp")
        ndcg_a_lin = ndcg_at_k(self.y[self.g == "A"], self.s[self.g == "A"], gain="linear")
        self.assertAlmostEqual(ndcg_a_exp, ndcg_a_lin)
        # DCG_A = 1 + 1/log2(4) = 1.5, IDCG_A = 1 + 1/log2(3)
        expected_a = 1.5 / (1.0 + 1.0 / LOG2_3)
        self.assertAlmostEqual(self.report["ndcg_A"], expected_a)
        expected_b = (1.0 / LOG2_3) / 1.0
        self.assertAlmostEqual(self.report["ndcg_B"], expected_b)

    def test_map_and_mrr(self):
        self.assertAlmostEqual(self.report["ap_A"], 5 / 6)
        self.assertAlmostEqual(self.report["ap_B"], 1 / 2)
        self.assertAlmostEqual(self.report["map"], 2 / 3)
        self.assertAlmostEqual(self.report["rr_A"], 1.0)
        self.assertAlmostEqual(self.report["rr_B"], 0.5)
        self.assertAlmostEqual(self.report["mrr"], 0.75)


class TestAucTiesAndUndefined(unittest.TestCase):
    def test_perfect_and_reversed(self):
        y = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(roc_auc_score(y, np.array([4.0, 3.0, 2.0, 1.0])), 1.0)
        self.assertAlmostEqual(roc_auc_score(y, np.array([1.0, 2.0, 3.0, 4.0])), 0.0)

    def test_tie_counts_as_half(self):
        # 1 个正、1 个负、同分 → AUC = 0.5
        self.assertAlmostEqual(roc_auc_score([1, 0], [0.7, 0.7]), 0.5)
        # 正 0.9, 0.5；负 0.5, 0.1
        # pairs: 0.9>0.5, 0.9>0.1, 0.5=0.5, 0.5>0.1 → (1+1+0.5+1)/4 = 3.5/4
        self.assertAlmostEqual(roc_auc_score([1, 1, 0, 0], [0.9, 0.5, 0.5, 0.1]), 0.875)

    def test_undefined_single_class(self):
        with self.assertRaises(MetricUndefinedError):
            roc_auc_score([1, 1, 1], [0.1, 0.2, 0.3])
        with self.assertRaises(MetricUndefinedError):
            roc_auc_score([0, 0], [0.1, 0.2])

    def test_mann_whitney_matches_pairwise(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=40)
        s = rng.normal(size=40)
        if y.sum() == 0 or y.sum() == 40:
            return
        pairwise = 0.0
        n_pos = n_neg = 0
        for i, yi in enumerate(y):
            if yi != 1:
                continue
            n_pos += 1
            for j, yj in enumerate(y):
                if yj != 0:
                    continue
                if s[i] > s[j]:
                    pairwise += 1.0
                elif s[i] == s[j]:
                    pairwise += 0.5
        n_neg = int((y == 0).sum())
        self.assertAlmostEqual(roc_auc_score(y, s), pairwise / (n_pos * n_neg))


class TestGaucSkipInvalidGroups(unittest.TestCase):
    def test_skips_all_positive_user(self):
        y = [1, 0, 1, 1]
        s = [0.9, 0.1, 0.8, 0.7]
        g = ["u1", "u1", "u2", "u2"]
        out = gauc_score(y, s, g, weight="uniform")
        self.assertEqual(out["n_valid_groups"], 1)
        self.assertEqual(out["n_skipped_groups"], 1)
        self.assertAlmostEqual(out["gauc"], 1.0)

    def test_all_groups_invalid(self):
        with self.assertRaises(MetricUndefinedError):
            gauc_score([1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4], ["a", "a", "b", "b"])


class TestNdcgMapMrr(unittest.TestCase):
    def test_ndcg_perfect_is_one(self):
        self.assertAlmostEqual(ndcg_at_k([3, 2, 1, 0], [4, 3, 2, 1], gain="exp"), 1.0)

    def test_ndcg_exp_gain_graded(self):
        # 排序后相关性 1, 3, 0；K=3
        # DCG = (2^1-1)/1 + (2^3-1)/log2(3) + 0 = 1 + 7/log2(3)
        # IDCG = (2^3-1)/1 + (2^1-1)/log2(3) = 7 + 1/log2(3)
        y = np.array([1.0, 3.0, 0.0])
        s = np.array([0.9, 0.5, 0.1])  # 把 1 排到 3 前面
        dcg = (2**1 - 1) / 1.0 + (2**3 - 1) / LOG2_3
        idcg = (2**3 - 1) / 1.0 + (2**1 - 1) / LOG2_3
        self.assertAlmostEqual(ndcg_at_k(y, s, k=3, gain="exp"), dcg / idcg)

    def test_ndcg_zero_when_no_relevant(self):
        self.assertEqual(ndcg_at_k([0, 0, 0], [0.3, 0.2, 0.1]), 0.0)

    def test_ap_trec_penalizes_missing(self):
        # 两个相关，只在 top1 命中一个；TREC AP@1 = (1/1) / 2 = 0.5
        # min(R,K) 归一化则 = 1
        y = [1, 1, 0]
        s = [0.9, 0.1, 0.8]
        self.assertAlmostEqual(average_precision(y, s, k=1, normalize="R"), 0.5)
        self.assertAlmostEqual(average_precision(y, s, k=1, normalize="min_R_K"), 1.0)

    def test_mrr_only_first_hit(self):
        y = [0, 1, 1]
        s = [0.9, 0.8, 0.7]
        self.assertAlmostEqual(reciprocal_rank(y, s), 0.5)
        # 第二个相关不影响 RR
        self.assertAlmostEqual(reciprocal_rank([0, 1, 0], s), 0.5)

    def test_map_mrr_skip_empty_query(self):
        y = [1, 0, 0, 0]
        s = [0.2, 0.9, 0.8, 0.7]
        g = ["q1", "q1", "q2", "q2"]
        m = mean_average_precision(y, s, g)
        r = mean_reciprocal_rank(y, s, g)
        n = mean_ndcg(y, s, g)
        self.assertEqual(m["n_skipped_groups"], 1)
        self.assertEqual(r["n_skipped_groups"], 1)
        self.assertEqual(n["n_skipped_groups"], 1)
        # q1: 相关在第 2 位
        self.assertAlmostEqual(m["map"], 0.5)
        self.assertAlmostEqual(r["mrr"], 0.5)


class TestDcgConvention(unittest.TestCase):
    def test_position_one_has_no_discount(self):
        self.assertAlmostEqual(dcg_at_k([1], gain="linear"), 1.0)
        self.assertAlmostEqual(dcg_at_k([1, 1], gain="linear"), 1.0 + 1.0 / math.log2(3))


if __name__ == "__main__":
    unittest.main()
