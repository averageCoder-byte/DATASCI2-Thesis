from pathlib import Path

import pandas as pd

from scipy.stats import mannwhitneyu, spearmanr



def load_data(score_path, label_path):
    scores = pd.read_csv(score_path)
    labels = pd.read_csv(label_path)

    required_scores = {
        "sequence_id",
        "anomaly_score",
    }

    required_labels = {
        "sequence_id",
        "proxy_anomaly",
        "anomalous_timestep_count",
    }

    missing_scores = required_scores - set(scores.columns)
    missing_labels = required_labels - set(labels.columns)

    if missing_scores:
        raise ValueError(
            f"Missing score columns: {sorted(missing_scores)}"
        )

    if missing_labels:
        raise ValueError(
            f"Missing label columns: {sorted(missing_labels)}"
        )

    return scores, labels


def compare_score_distributions(scores, labels):
    df = scores.merge(
        labels[
            [
                "sequence_id",
                "proxy_anomaly",
                "anomalous_timestep_count",
            ]
        ],
        on="sequence_id",
        how="inner",
        validate="one_to_one",
    )
    if len(df) != len(scores):
        raise ValueError(
            "Some sequences in the score file "
            "could not be matched to proxy labels."
        )

    print()
    print("Proxy-label / GMM score comparison:")
    print()

    for label, group_name in [
        (0, "Proxy-normal"),
        (1, "Proxy-anomalous"),
    ]:
        group = df.loc[
            df["proxy_anomaly"] == label,
            "anomaly_score",
        ]

        if group.empty:
            print(f"{group_name}: no sequences")
            continue

        print(f"{group_name} sequences:")
        print(f"  Count:   {len(group):,}")
        print(f"  Mean:    {group.mean():.6f}")
        print(f"  Median:  {group.median():.6f}")
        print(f"  Std:     {group.std():.6f}")
        print(f"  Min:     {group.min():.6f}")
        print(f"  Max:     {group.max():.6f}")
        print(f"  90th:    {group.quantile(0.90):.6f}")
        print(f"  95th:    {group.quantile(0.95):.6f}")
        print(f"  99th:    {group.quantile(0.99):.6f}")
        print()

    return df

def calculate_effect_size(normal_scores, anomalous_scores):
    """
    Calculate rank-biserial correlation from the Mann-Whitney U statistic.

    Positive values mean anomalous sequences tend to have higher
    anomaly scores than normal sequences.
    """
    u_stat, p_value = mannwhitneyu(
        anomalous_scores,
        normal_scores,
        alternative="greater"
    )

    n_anomalous = len(anomalous_scores)
    n_normal = len(normal_scores)

    # Probability that a randomly selected anomalous score
    # exceeds a randomly selected normal score.
    probability_superiority = u_stat / (n_anomalous * n_normal)

    # Rank-biserial correlation
    rank_biserial = 2 * probability_superiority - 1

    return u_stat, p_value, probability_superiority, rank_biserial

def calculate_severity_correlation(df):
    severity = df["anomalous_timestep_count"]
    scores = df["anomaly_score"]

    rho, p_value = spearmanr(
        severity,
        scores,
    )

    return rho, p_value

def analyze_severity_groups(df):
    anomalous = df.loc[
        df["proxy_anomaly"] == 1
    ].copy()

    anomalous["severity_group"] = pd.cut(
        anomalous["anomalous_timestep_count"],
        bins=[0, 1, 3, 5, 10, 20, 30, 40, float("inf")],
        labels=[
            "1",
            "2-3",
            "4-5",
            "6-10",
            "11-20",
            "21-30",
            "31-40",
            "41+",
        ],
    )

    summary = (
        anomalous
        .groupby(
            "severity_group",
            observed=False,
        )["anomaly_score"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
        )
    )

    print("\nGMM score by proxy-anomaly severity:")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compare GMM anomaly scores against "
            "rule-based proxy labels."
        )
    )

    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    scores, labels = load_data(
        args.scores,
        args.labels,
    )

    df = compare_score_distributions(
        scores,
        labels,
    )

    print(f"Matched sequences: {len(df):,}")

    normal_scores = df.loc[
        df["proxy_anomaly"] == 0,
        "anomaly_score",
    ].to_numpy()

    anomalous_scores = df.loc[
        df["proxy_anomaly"] == 1,
        "anomaly_score",
    ].to_numpy()

    u_stat, p_value, probability_superiority, rank_biserial = (
        calculate_effect_size(
            normal_scores,
            anomalous_scores,
        )
    )

    print("\nStatistical separation:")
    print(f"  Mann-Whitney U:              {u_stat:,.0f}")
    print(f"  p-value:                     {p_value:.6e}")
    print(
        f"  Probability of superiority:  "
        f"{probability_superiority:.4f}"
    )
    print(
        f"  Rank-biserial correlation:   "
        f"{rank_biserial:.4f}"
    )

    rho, severity_p_value = calculate_severity_correlation(df)

    print("\nProxy severity / GMM score correlation:")
    print(f"  Spearman rho: {rho:.4f}")
    print(f"  p-value:      {severity_p_value:.6e}")

    analyze_severity_groups(df)


if __name__ == "__main__":
    main()