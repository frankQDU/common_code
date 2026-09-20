"""排序评估指标参考实现：AUC / GAUC / NDCG / MAP / MRR。

公式、分组、加权与失效条件见 ``docs/排序评估指标手册.md``。
本模块只负责把手册中的定义算出来，不改变定义本身。
"""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence]

GainType = Literal["linear", "exp"]
GaucWeight = Literal["uniform", "impression", "click", "pair"]
MapNormalize = Literal["R", "min_R_K"]


class MetricUndefinedError(ValueError):
    """指标在当前样本上无定义（例如 AUC 缺正类或缺负类）。"""


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，实际 shape={arr.shape}")
    return arr.astype(np.float64, copy=False)


def _as_1d_int(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，实际 shape={arr.shape}")
    return arr.astype(np.int64, copy=False)


def _rankdata_average(scores: np.ndarray) -> np.ndarray:
    """升序平均秩（结赋予 mid-rank），秩从 1 开始。"""
    n = scores.size
    if n == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # 1-based ranks of this tie group: (i+1) .. j
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def roc_auc_score(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """全局 ROC-AUC（Wilcoxon–Mann–Whitney 形式，结记 0.5）。

    .. math::

        \\widehat{\\mathrm{AUC}}
        = \\frac{1}{n^+ n^-}
          \\sum_{i:y_i=1}\\sum_{j:y_j=0}
          \\Big[ \\mathbf{1}[s_i>s_j] + \\tfrac12 \\mathbf{1}[s_i=s_j] \\Big]

    缺正类或缺负类时抛 ``MetricUndefinedError``。
    """
    y = _as_1d_float(y_true, "y_true")
    s = _as_1d_float(y_score, "y_score")
    if y.size != s.size:
        raise ValueError(f"y_true 与 y_score 长度不一致: {y.size} vs {s.size}")
    if y.size == 0:
        raise MetricUndefinedError("空样本无法计算 AUC")

    pos = y > 0
    n_pos = int(pos.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise MetricUndefinedError(
            f"AUC 无定义: n+={n_pos}, n-={n_neg}（需要同时存在正负样本）"
        )

    ranks = _rankdata_average(s)
    # Mann–Whitney U = R_pos - n_pos*(n_pos+1)/2，AUC = U / (n_pos n_neg)
    r_pos = float(ranks[pos].sum())
    u = r_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _group_slices(group_id: np.ndarray) -> Dict[object, np.ndarray]:
    """返回 {group_value: index_array}，组顺序按首次出现。"""
    groups: Dict[object, list] = {}
    for i, gid in enumerate(group_id.tolist()):
        groups.setdefault(gid, []).append(i)
    return {gid: np.asarray(idx, dtype=np.int64) for gid, idx in groups.items()}


def gauc_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    group_id: ArrayLike,
    *,
    weight: GaucWeight = "impression",
) -> Dict[str, object]:
    """分组 AUC（默认按曝光数加权）。

    .. math::

        \\mathrm{GAUC}
        = \\frac{\\sum_{u \\in \\mathcal{U}_{\\mathrm{valid}}} w_u \\, \\mathrm{AUC}_u}
                {\\sum_{u \\in \\mathcal{U}_{\\mathrm{valid}}} w_u}

    ``weight``:
      - ``uniform``: :math:`w_u=1`
      - ``impression``: :math:`w_u=n_u`（曝光/样本数，DIN 默认）
      - ``click``: :math:`w_u=n_u^+`
      - ``pair``: :math:`w_u=n_u^+ n_u^-`（与 AUC 方差更匹配）

    只有正类或只有负类的组会被丢弃。返回 dict，含 gauc、覆盖率与各组明细。
    """
    y = _as_1d_float(y_true, "y_true")
    s = _as_1d_float(y_score, "y_score")
    g = np.asarray(group_id)
    if not (y.size == s.size == g.size):
        raise ValueError("y_true / y_score / group_id 长度必须一致")
    if y.size == 0:
        raise MetricUndefinedError("空样本无法计算 GAUC")
    if weight not in ("uniform", "impression", "click", "pair"):
        raise ValueError(f"未知 weight={weight}")

    details = []
    num = 0.0
    den = 0.0
    n_groups = 0
    n_skipped = 0
    n_samples_used = 0

    for gid, idx in _group_slices(g).items():
        n_groups += 1
        y_u = y[idx]
        n_u = int(y_u.size)
        n_pos = int((y_u > 0).sum())
        n_neg = n_u - n_pos
        if n_pos == 0 or n_neg == 0:
            n_skipped += 1
            details.append(
                {
                    "group_id": gid,
                    "auc": None,
                    "n": n_u,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "weight": 0.0,
                    "valid": False,
                }
            )
            continue
        auc_u = roc_auc_score(y_u, s[idx])
        if weight == "uniform":
            w_u = 1.0
        elif weight == "impression":
            w_u = float(n_u)
        elif weight == "click":
            w_u = float(n_pos)
        else:
            w_u = float(n_pos * n_neg)
        num += w_u * auc_u
        den += w_u
        n_samples_used += n_u
        details.append(
            {
                "group_id": gid,
                "auc": auc_u,
                "n": n_u,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "weight": w_u,
                "valid": True,
            }
        )

    if den <= 0:
        raise MetricUndefinedError("没有同时包含正负样本的有效组，GAUC 无定义")

    return {
        "gauc": num / den,
        "weight": weight,
        "n_groups": n_groups,
        "n_valid_groups": n_groups - n_skipped,
        "n_skipped_groups": n_skipped,
        "group_coverage": (n_groups - n_skipped) / n_groups,
        "sample_coverage": n_samples_used / y.size,
        "details": details,
    }


def _sort_by_score_desc(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """按预测分降序排列标签；同分用稳定排序保持输入相对顺序。"""
    order = np.argsort(-s, kind="mergesort")
    return y[order]


def dcg_at_k(relevances: ArrayLike, k: Optional[int] = None, gain: GainType = "exp") -> float:
    """Discounted Cumulative Gain。

    位置从 1 开始：

    .. math::

        \\mathrm{DCG@}K = \\sum_{i=1}^{K} \\frac{g(r_i)}{\\log_2(i+1)}

    ``gain='exp'`` 时 :math:`g(r)=2^r-1`；``gain='linear'`` 时 :math:`g(r)=r`。
    """
    r = _as_1d_float(relevances, "relevances")
    if k is not None:
        r = r[: max(int(k), 0)]
    if r.size == 0:
        return 0.0
    if gain == "exp":
        gains = np.power(2.0, r) - 1.0
    elif gain == "linear":
        gains = r
    else:
        raise ValueError(f"未知 gain={gain}")
    discounts = np.log2(np.arange(2, r.size + 2, dtype=np.float64))
    return float(np.sum(gains / discounts))


def ndcg_at_k(
    y_true: ArrayLike,
    y_score: ArrayLike,
    k: Optional[int] = None,
    gain: GainType = "exp",
) -> float:
    """单列表 NDCG@K。无正增益（IDCG=0）时返回 0。"""
    y = _as_1d_float(y_true, "y_true")
    s = _as_1d_float(y_score, "y_score")
    if y.size != s.size:
        raise ValueError("y_true 与 y_score 长度不一致")
    ranked = _sort_by_score_desc(y, s)
    dcg = dcg_at_k(ranked, k=k, gain=gain)
    ideal = np.sort(y)[::-1]
    idcg = dcg_at_k(ideal, k=k, gain=gain)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def mean_ndcg(
    y_true: ArrayLike,
    y_score: ArrayLike,
    group_id: ArrayLike,
    k: Optional[int] = None,
    gain: GainType = "exp",
    *,
    skip_no_relevant: bool = True,
) -> Dict[str, object]:
    """按组计算 NDCG 再等权平均。"""
    y = _as_1d_float(y_true, "y_true")
    s = _as_1d_float(y_score, "y_score")
    g = np.asarray(group_id)
    if not (y.size == s.size == g.size):
        raise ValueError("y_true / y_score / group_id 长度必须一致")

    values = []
    n_skipped = 0
    for _, idx in _group_slices(g).items():
        y_u = y[idx]
        if skip_no_relevant and not np.any(y_u > 0):
            n_skipped += 1
            continue
        values.append(ndcg_at_k(y_u, s[idx], k=k, gain=gain))
    if not values:
        raise MetricUndefinedError("没有有效组，无法计算 mean NDCG")
    arr = np.asarray(values, dtype=np.float64)
    n_groups = len(_group_slices(g))
    return {
        "ndcg": float(arr.mean()),
        "k": k,
        "gain": gain,
        "n_groups": n_groups,
        "n_valid_groups": len(values),
        "n_skipped_groups": n_skipped,
        "per_group": arr.tolist(),
    }


def average_precision(
    y_true: ArrayLike,
    y_score: ArrayLike,
    k: Optional[int] = None,
    *,
    normalize: MapNormalize = "R",
) -> float:
    """单列表 Average Precision。

    相关文档位置为 :math:`k_1,\\ldots,k_{R'}`（截断后）时：

    .. math::

        \\mathrm{AP@}K = \\frac{1}{Z} \\sum_{j=1}^{R'} \\frac{j}{k_j}

    ``normalize='R'``（TREC）：:math:`Z=R`，截断外的相关文档会惩罚 AP。
    ``normalize='min_R_K'``：:math:`Z=\\min(R,K)`，只在窗口内归一化。
    无相关文档时返回 0。
    """
    y = _as_1d_float(y_true, "y_true")
    s = _as_1d_float(y_score, "y_score")
    if y.size != s.size:
        raise ValueError("y_true 与 y_score 长度不一致")
    binary = (y > 0).astype(np.int64)
    r_total = int(binary.sum())
    if r_total == 0:
        return 0.0
    ranked = _sort_by_score_desc(binary.astype(np.float64), s)
    if k is not None:
        ranked = ranked[: max(int(k), 0)]
    hits = np.cumsum(ranked)
    precisions = hits / np.arange(1, ranked.size + 1)
    ap_sum = float(np.sum(precisions * ranked))
    if normalize == "R":
        z = float(r_total)
    elif normalize == "min_R_K":
        z = float(min(r_total, ranked.size))
    else:
        raise ValueError(f"未知 normalize={normalize}")
    if z <= 0:
        return 0.0
    return ap_sum / z


def mean_average_precision(
    y_true: ArrayLike,
    y_score: ArrayLike,
    group_id: ArrayLike,
    k: Optional[int] = None,
    *,
    normalize: MapNormalize = "R",
    skip_no_relevant: bool = True,
) -> Dict[str, object]:
    """MAP：各组 AP 的等权平均。"""
    y = _as_1d_float(y_true, "y_true")
    s = _as_1d_float(y_score, "y_score")
    g = np.asarray(group_id)
    if not (y.size == s.size == g.size):
        raise ValueError("y_true / y_score / group_id 长度必须一致")

    values = []
    n_skipped = 0
    for _, idx in _group_slices(g).items():
        y_u = y[idx]
        if skip_no_relevant and not np.any(y_u > 0):
            n_skipped += 1
            continue
        values.append(average_precision(y_u, s[idx], k=k, normalize=normalize))
    if not values:
        raise MetricUndefinedError("没有有效组，无法计算 MAP")
    arr = np.asarray(values, dtype=np.float64)
    n_groups = len(_group_slices(g))
    return {
        "map": float(arr.mean()),
        "k": k,
        "normalize": normalize,
        "n_groups": n_groups,
        "n_valid_groups": len(values),
        "n_skipped_groups": n_skipped,
        "per_group": arr.tolist(),
    }


def reciprocal_rank(
    y_true: ArrayLike,
    y_score: ArrayLike,
    k: Optional[int] = None,
) -> float:
    """单列表 Reciprocal Rank：第一个相关文档位次的倒数；没有则 0。"""
    y = _as_1d_float(y_true, "y_true")
    s = _as_1d_float(y_score, "y_score")
    if y.size != s.size:
        raise ValueError("y_true 与 y_score 长度不一致")
    ranked = _sort_by_score_desc((y > 0).astype(np.float64), s)
    if k is not None:
        ranked = ranked[: max(int(k), 0)]
    hits = np.where(ranked > 0)[0]
    if hits.size == 0:
        return 0.0
    return 1.0 / float(hits[0] + 1)


def mean_reciprocal_rank(
    y_true: ArrayLike,
    y_score: ArrayLike,
    group_id: ArrayLike,
    k: Optional[int] = None,
    *,
    skip_no_relevant: bool = True,
) -> Dict[str, object]:
    """MRR：各组 RR 的等权平均。"""
    y = _as_1d_float(y_true, "y_true")
    s = _as_1d_float(y_score, "y_score")
    g = np.asarray(group_id)
    if not (y.size == s.size == g.size):
        raise ValueError("y_true / y_score / group_id 长度必须一致")

    values = []
    n_skipped = 0
    for _, idx in _group_slices(g).items():
        y_u = y[idx]
        if skip_no_relevant and not np.any(y_u > 0):
            n_skipped += 1
            continue
        values.append(reciprocal_rank(y_u, s[idx], k=k))
    if not values:
        raise MetricUndefinedError("没有有效组，无法计算 MRR")
    arr = np.asarray(values, dtype=np.float64)
    n_groups = len(_group_slices(g))
    return {
        "mrr": float(arr.mean()),
        "k": k,
        "n_groups": n_groups,
        "n_valid_groups": len(values),
        "n_skipped_groups": n_skipped,
        "per_group": arr.tolist(),
    }


# ---------------------------------------------------------------------------
# 手册第 6 章使用的贯通数值例
# ---------------------------------------------------------------------------

HANDBOOK_EXAMPLE = {
    "group_id": np.array(["A", "A", "A", "A", "A", "A", "B", "B", "B"]),
    "y_true": np.array([1, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.float64),
    "y_score": np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.1, 0.95, 0.6, 0.2]),
}


def handbook_example_report() -> Dict[str, object]:
    """复现手册第 6 章数值例，便于对照手算。"""
    y = HANDBOOK_EXAMPLE["y_true"]
    s = HANDBOOK_EXAMPLE["y_score"]
    g = HANDBOOK_EXAMPLE["group_id"]
    mask_a = g == "A"
    mask_b = g == "B"
    gauc_by_weight = {
        w: gauc_score(y, s, g, weight=w)["gauc"]
        for w in ("uniform", "impression", "click", "pair")
    }
    return {
        "auc_global": roc_auc_score(y, s),
        "auc_A": roc_auc_score(y[mask_a], s[mask_a]),
        "auc_B": roc_auc_score(y[mask_b], s[mask_b]),
        "gauc": gauc_by_weight,
        "ndcg_A": ndcg_at_k(y[mask_a], s[mask_a], gain="exp"),
        "ndcg_B": ndcg_at_k(y[mask_b], s[mask_b], gain="exp"),
        "mean_ndcg": mean_ndcg(y, s, g, gain="exp")["ndcg"],
        "ap_A": average_precision(y[mask_a], s[mask_a]),
        "ap_B": average_precision(y[mask_b], s[mask_b]),
        "map": mean_average_precision(y, s, g)["map"],
        "rr_A": reciprocal_rank(y[mask_a], s[mask_a]),
        "rr_B": reciprocal_rank(y[mask_b], s[mask_b]),
        "mrr": mean_reciprocal_rank(y, s, g)["mrr"],
    }


def _fmt(x: float) -> str:
    return f"{x:.6f}"


def main() -> None:
    report = handbook_example_report()
    print("手册第 6 章贯通数值例")
    print(f"  全局 AUC          = {_fmt(report['auc_global'])}   (手算 13/18)")
    print(f"  AUC_A / AUC_B     = {_fmt(report['auc_A'])} / {_fmt(report['auc_B'])}   (7/8, 1/2)")
    print("  GAUC")
    print(f"    uniform         = {_fmt(report['gauc']['uniform'])}   (11/16)")
    print(f"    impression      = {_fmt(report['gauc']['impression'])}   (3/4)")
    print(f"    click           = {_fmt(report['gauc']['click'])}   (3/4)")
    print(f"    pair            = {_fmt(report['gauc']['pair'])}   (4/5)")
    print(f"  NDCG_A / NDCG_B   = {_fmt(report['ndcg_A'])} / {_fmt(report['ndcg_B'])}")
    print(f"  mean NDCG         = {_fmt(report['mean_ndcg'])}")
    print(f"  AP_A / AP_B       = {_fmt(report['ap_A'])} / {_fmt(report['ap_B'])}   (5/6, 1/2)")
    print(f"  MAP               = {_fmt(report['map'])}   (2/3)")
    print(f"  RR_A / RR_B       = {_fmt(report['rr_A'])} / {_fmt(report['rr_B'])}   (1, 1/2)")
    print(f"  MRR               = {_fmt(report['mrr'])}   (3/4)")


if __name__ == "__main__":
    main()
