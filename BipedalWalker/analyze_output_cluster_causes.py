import argparse
import pickle
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


CONCEPTS = [
    ("positive_tilt", "Pos. Tilt"),
    ("negative_tilt", "Neg. Tilt"),
    ("contact_loss", "Contact Loss"),
    ("single_leg_support", "Single-Leg"),
    ("near_obstacle", "Near Obstacle"),
    ("low_forward_speed", "Low Speed"),
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Generate one crash-aligned concept-temporal heatmap for selected "
            "QDFuzz output-diversity clusters. The script writes only the final "
            "heatmap PDF."
        )
    )
    parser.add_argument(
        "--selection-log",
        type=Path,
        default=script_dir / "selection_log.pkl",
        help="Path to selection_log.pkl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "concept_temporal_cluster_heatmap.pdf",
        help="Output PDF path. This is the only file written.",
    )
    parser.add_argument(
        "--clusters",
        default="auto",
        help=(
            "Comma-separated output cluster IDs to plot, or 'auto' to choose "
            "the clusters with the largest concept-profile differences."
        ),
    )
    parser.add_argument(
        "--num-display-clusters",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of clusters to display when --clusters=auto.",
    )
    parser.add_argument(
        "--trajectories-per-cluster",
        type=int,
        default=2,
        help="Number of representative trajectories to draw for each displayed cluster.",
    )
    parser.add_argument(
        "--representative-pool-fraction",
        type=float,
        default=0.35,
        help=(
            "Fraction of profile-nearest trajectories considered similar enough "
            "when choosing a visually different second example."
        ),
    )
    parser.add_argument("--window", type=int, default=50, help="Crash-tail window in steps.")
    parser.add_argument("--k", type=int, default=None, help="Force output-cluster count.")
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize padded trajectories before PCA. Off by default to match qdfuzz/plot.py.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["median", "mean"],
        default="median",
        help="How to aggregate all trajectories inside each cluster at each time step.",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def parse_cluster_ids(value: str) -> List[int]:
    clusters = []
    for item in value.split(","):
        item = item.strip()
        if item:
            clusters.append(int(item))
    if not clusters:
        raise ValueError("--clusters must contain at least one cluster ID")
    return clusters


def should_auto_select_clusters(value: str) -> bool:
    return value.strip().lower() == "auto"


def state_key_and_array(state: Any) -> Tuple[bytes, np.ndarray]:
    if isinstance(state, bytes):
        if len(state) == 15 * np.dtype(np.int32).itemsize:
            dtype = np.int32
        elif len(state) == 15 * np.dtype(np.int64).itemsize:
            dtype = np.int64
        else:
            raise ValueError(f"unexpected mutate_state byte length: {len(state)}")
        return state, np.frombuffer(state, dtype=dtype).astype(np.int64)

    array = np.asarray(state)
    if array.ndim != 1:
        array = array.reshape(-1)
    return array.tobytes(), array.astype(np.int64)


def deduplicate_log(log_data: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    state_to_entry: Dict[bytes, Dict[str, Any]] = {}

    for source_index, entry in enumerate(log_data):
        if not isinstance(entry, dict) or "mutate_state" not in entry:
            continue

        try:
            state_key, state_array = state_key_and_array(entry["mutate_state"])
        except Exception:
            continue

        entry_copy = entry.copy()
        entry_copy["_source_index"] = source_index
        entry_copy["_mutate_state_array"] = state_array

        old_entry = state_to_entry.get(state_key)
        if old_entry is None:
            state_to_entry[state_key] = entry_copy
        elif bool(entry_copy.get("did_crash", False)) and not bool(old_entry.get("did_crash", False)):
            state_to_entry[state_key] = entry_copy

    return list(state_to_entry.values())


def extract_crash_records(deduplicated_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for dedup_index, entry in enumerate(deduplicated_log):
        if not bool(entry.get("did_crash", False)):
            continue
        trajectory = entry.get("output_trajectory")
        if trajectory is None:
            continue

        trajectory_array = np.asarray(trajectory, dtype=np.float32)
        if trajectory_array.ndim != 2 or trajectory_array.shape[0] == 0:
            continue

        records.append(
            {
                "dedup_index": dedup_index,
                "source_index": entry.get("_source_index"),
                "trajectory": trajectory_array,
                "survival_steps": int(entry.get("survival_steps", len(trajectory_array))),
            }
        )
    return records


def padded_trajectory_matrix(records: List[Dict[str, Any]]) -> np.ndarray:
    max_len = max(record["trajectory"].shape[0] for record in records)
    flattened = []
    for record in records:
        trajectory = record["trajectory"]
        pad_len = max_len - trajectory.shape[0]
        if pad_len > 0:
            trajectory = np.pad(trajectory, ((0, pad_len), (0, 0)), mode="constant")
        flattened.append(trajectory.reshape(-1))
    return np.asarray(flattened, dtype=np.float32)


def choose_cluster_count(
    reduced_data: np.ndarray,
    forced_k: int | None,
    max_k: int,
) -> Tuple[int, Dict[int, float]]:
    n_samples = reduced_data.shape[0]
    max_k = min(max_k, n_samples - 1)
    if max_k < 2:
        raise ValueError("Need at least two crash trajectories for clustering.")

    if forced_k is not None:
        if forced_k < 2 or forced_k > max_k:
            raise ValueError(f"--k must be in [2, {max_k}]")
        labels = KMeans(n_clusters=forced_k, random_state=42, n_init=10).fit_predict(reduced_data)
        return forced_k, {forced_k: float(silhouette_score(reduced_data, labels))}

    best_k = 2
    labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(reduced_data)
    best_score = float(silhouette_score(reduced_data, labels))
    scores = {2: best_score}

    for k in range(3, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(reduced_data)
        score = float(silhouette_score(reduced_data, labels))
        scores[k] = score
        if score >= best_score * 1.20:
            best_score = score
            best_k = k

    return best_k, scores


def cluster_records(
    records: List[Dict[str, Any]],
    forced_k: int | None,
    max_k: int,
    pca_components: int,
    standardize: bool,
) -> Tuple[np.ndarray, int, Dict[int, float]]:
    matrix = padded_trajectory_matrix(records)
    if standardize:
        matrix = StandardScaler().fit_transform(matrix)

    n_components = min(matrix.shape[0], matrix.shape[1], pca_components)
    reduced_data = PCA(n_components=n_components, random_state=42).fit_transform(matrix)
    best_k, scores = choose_cluster_count(reduced_data, forced_k, max_k)
    labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(reduced_data)
    return labels, best_k, scores


def clip01(value: np.ndarray | float) -> np.ndarray | float:
    return np.clip(value, 0.0, 1.0)


def trajectory_concept_matrix(trajectory: np.ndarray, window: int) -> np.ndarray:
    tail = trajectory[-min(window, len(trajectory)) :]
    values = np.full((len(CONCEPTS), window), np.nan, dtype=np.float32)
    offset = window - len(tail)

    hull_angle = tail[:, 0] if tail.shape[1] > 0 else np.zeros(len(tail))
    x_velocity = tail[:, 2] if tail.shape[1] > 2 else np.zeros(len(tail))

    leg1_contact = tail[:, 8] >= 0.5 if tail.shape[1] > 8 else np.zeros(len(tail), dtype=bool)
    leg2_contact = tail[:, 13] >= 0.5 if tail.shape[1] > 13 else np.zeros(len(tail), dtype=bool)
    both_air = (~leg1_contact) & (~leg2_contact)
    single_contact = np.logical_xor(leg1_contact, leg2_contact)

    lidar = tail[:, 14:24] if tail.shape[1] >= 24 else np.ones((len(tail), 1), dtype=np.float32)
    min_lidar = np.nanmin(lidar, axis=1)

    series = [
        clip01(np.maximum(hull_angle, 0.0) / 2.5),
        clip01(np.maximum(-hull_angle, 0.0) / 2.5),
        both_air.astype(np.float32),
        single_contact.astype(np.float32),
        clip01((0.30 - min_lidar) / 0.30),
        clip01((0.30 - x_velocity) / 0.60),
    ]

    for row_index, concept_series in enumerate(series):
        values[row_index, offset:] = concept_series

    return values


def aggregate_concept_matrices(matrices: np.ndarray, aggregation: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if aggregation == "mean":
            return np.nanmean(matrices, axis=0)
        return np.nanmedian(matrices, axis=0)


def matrix_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.nan_to_num(left, nan=0.0).reshape(-1)
    right_flat = np.nan_to_num(right, nan=0.0).reshape(-1)
    return float(np.sqrt(np.mean((left_flat - right_flat) ** 2)))


def build_cluster_profiles(
    records: List[Dict[str, Any]],
    labels: np.ndarray,
    window: int,
    aggregation: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, int], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    profiles: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    cluster_matrices: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for cluster_id in sorted(int(x) for x in np.unique(labels)):
        indices = np.where(labels == cluster_id)[0]
        matrices = np.stack(
            [trajectory_concept_matrix(records[index]["trajectory"], window) for index in indices],
            axis=0,
        )
        profiles[cluster_id] = aggregate_concept_matrices(matrices, aggregation)
        counts[cluster_id] = int(len(indices))
        cluster_matrices[cluster_id] = (indices, matrices)

    return profiles, counts, cluster_matrices


def choose_most_different_clusters(
    profiles: Dict[int, np.ndarray],
    num_clusters: int,
) -> Tuple[List[int], float]:
    available_clusters = sorted(profiles)
    if len(available_clusters) < num_clusters:
        raise ValueError(
            f"Cannot display {num_clusters} clusters because only "
            f"{len(available_clusters)} clusters are available."
        )

    best_clusters: Tuple[int, ...] | None = None
    best_score = -np.inf
    for candidate_clusters in combinations(available_clusters, num_clusters):
        score = sum(
            matrix_distance(profiles[left], profiles[right])
            for left, right in combinations(candidate_clusters, 2)
        )
        if score > best_score:
            best_score = score
            best_clusters = candidate_clusters

    if best_clusters is None:
        raise ValueError("Could not choose display clusters.")
    return list(best_clusters), float(best_score)


def select_display_clusters(
    cluster_argument: str,
    profiles: Dict[int, np.ndarray],
    num_display_clusters: int,
) -> Tuple[List[int], str, float | None]:
    if should_auto_select_clusters(cluster_argument):
        selected_clusters, score = choose_most_different_clusters(profiles, num_display_clusters)
        return selected_clusters, "auto", score

    selected_clusters = parse_cluster_ids(cluster_argument)
    available_clusters = set(profiles)
    missing = [cluster_id for cluster_id in selected_clusters if cluster_id not in available_clusters]
    if missing:
        raise ValueError(f"Unknown cluster IDs: {missing}")
    return selected_clusters, "manual", None


def select_representative_trajectory_panels(
    records: List[Dict[str, Any]],
    selected_clusters: List[int],
    profiles: Dict[int, np.ndarray],
    counts: Dict[int, int],
    cluster_matrices: Dict[int, Tuple[np.ndarray, np.ndarray]],
    trajectories_per_cluster: int,
    representative_pool_fraction: float,
) -> List[Tuple[int, int, int, Any, int, float, np.ndarray]]:
    if trajectories_per_cluster < 1:
        raise ValueError("--trajectories-per-cluster must be at least 1")
    if not 0.0 < representative_pool_fraction <= 1.0:
        raise ValueError("--representative-pool-fraction must be in (0, 1]")

    panels: List[Tuple[int, int, int, Any, int, float, np.ndarray]] = []
    for cluster_id in selected_clusters:
        record_indices, matrices = cluster_matrices[cluster_id]
        distances = np.asarray(
            [matrix_distance(matrix, profiles[cluster_id]) for matrix in matrices],
            dtype=np.float32,
        )
        by_profile_distance = np.argsort(distances)
        representative_order = [int(by_profile_distance[0])]

        while len(representative_order) < min(trajectories_per_cluster, len(matrices)):
            chosen_index = None
            for fraction in [representative_pool_fraction, 0.50, 0.75, 1.0]:
                pool_size = max(
                    len(representative_order) + 1,
                    int(np.ceil(len(matrices) * fraction)),
                )
                candidate_indices = [
                    int(index)
                    for index in by_profile_distance[:pool_size]
                    if int(index) not in representative_order
                ]
                if not candidate_indices:
                    continue

                chosen_index = max(
                    candidate_indices,
                    key=lambda index: (
                        min(
                            matrix_distance(matrices[index], matrices[selected_index])
                            for selected_index in representative_order
                        ),
                        -float(distances[index]),
                    ),
                )
                if any(
                    matrix_distance(matrices[chosen_index], matrices[selected_index]) > 1e-6
                    for selected_index in representative_order
                ) or fraction == 1.0:
                    break

            if chosen_index is None:
                break
            representative_order.append(chosen_index)

        for trajectory_rank, matrix_index in enumerate(representative_order, start=1):
            record_index = int(record_indices[int(matrix_index)])
            record = records[record_index]
            panels.append(
                (
                    cluster_id,
                    trajectory_rank,
                    counts[cluster_id],
                    record.get("source_index"),
                    int(record["dedup_index"]),
                    float(distances[int(matrix_index)]),
                    matrices[int(matrix_index)],
                )
            )

    return panels


def draw_cluster_temporal_heatmap(
    panels: List[Tuple[int, int, int, Any, int, float, np.ndarray]],
    output_path: Path,
    window: int,
    selection_mode: str,
) -> None:
    concept_labels = [label for _, label in CONCEPTS]
    cluster_order = []
    for cluster_id, *_ in panels:
        if cluster_id not in cluster_order:
            cluster_order.append(cluster_id)

    n_rows = len(cluster_order)
    n_cols = max(trajectory_rank for _, trajectory_rank, *_ in panels)
    panel_lookup = {
        (cluster_id, trajectory_rank): matrix
        for cluster_id, trajectory_rank, _, _, _, _, matrix in panels
    }

    fig_width = 3.30
    fig_height = max(2.15, 0.92 * n_rows + 0.38)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    image = None
    x_ticks = [0, window // 2, window - 1]
    x_labels = [f"-{window - 1}", f"-{window // 2}", "0"]
    panel_index = 0

    for row_index, cluster_id in enumerate(cluster_order):
        for col_index in range(n_cols):
            trajectory_rank = col_index + 1
            ax = axes[row_index][col_index]
            matrix = panel_lookup.get((cluster_id, trajectory_rank))
            if matrix is None:
                ax.axis("off")
                continue

            image = ax.imshow(
                matrix,
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
                interpolation="nearest",
                rasterized=True,
            )
            ax.set_title(f"C{cluster_id}-T{trajectory_rank}", fontsize=7, pad=1)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, fontsize=6)
            ax.set_yticks(np.arange(len(concept_labels)))
            if col_index == 0:
                ax.set_yticklabels(concept_labels, fontsize=6)
            else:
                ax.tick_params(labelleft=False)
            if row_index != n_rows - 1:
                ax.tick_params(labelbottom=False)
            ax.set_xticks(np.arange(-0.5, window, 10), minor=True)
            ax.set_yticks(np.arange(-0.5, len(concept_labels), 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.18, alpha=0.28)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.tick_params(axis="both", which="major", length=2, pad=1)
            ax.text(
                0.5,
                -0.17,
                f"({chr(97 + panel_index)})",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                clip_on=False,
            )
            panel_index += 1

    fig.text(0.54, 0.030, "Steps Before Crash", ha="center", fontsize=7)
    fig.subplots_adjust(left=0.28, right=0.86, bottom=0.19, top=0.93, wspace=0.08, hspace=0.52)
    colorbar_axis = fig.add_axes([0.88, 0.19, 0.022, 0.73])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Strength", fontsize=7)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.ax.tick_params(labelsize=6, length=2, pad=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_data = load_pickle(args.selection_log.resolve())
    if not isinstance(log_data, list):
        raise TypeError("selection_log.pkl must contain a list of dictionaries")

    deduplicated_log = deduplicate_log(log_data)
    records = extract_crash_records(deduplicated_log)
    if len(records) < 2:
        raise ValueError(f"Need at least two crash trajectories, found {len(records)}")

    labels, best_k, _ = cluster_records(
        records=records,
        forced_k=args.k,
        max_k=args.max_k,
        pca_components=args.pca_components,
        standardize=args.standardize,
    )
    profiles, counts, cluster_matrices = build_cluster_profiles(
        records=records,
        labels=labels,
        window=args.window,
        aggregation=args.aggregation,
    )
    selected_clusters, selection_mode, difference_score = select_display_clusters(
        cluster_argument=args.clusters,
        profiles=profiles,
        num_display_clusters=args.num_display_clusters,
    )
    panels = select_representative_trajectory_panels(
        records=records,
        selected_clusters=selected_clusters,
        profiles=profiles,
        counts=counts,
        cluster_matrices=cluster_matrices,
        trajectories_per_cluster=args.trajectories_per_cluster,
        representative_pool_fraction=args.representative_pool_fraction,
    )
    draw_cluster_temporal_heatmap(
        panels=panels,
        output_path=args.output.resolve(),
        window=args.window,
        selection_mode=selection_mode,
    )

    print(f"Output cluster count: {best_k}")
    if difference_score is None:
        print(f"Displayed clusters: {selected_clusters} (manual)")
    else:
        print(
            f"Displayed clusters: {selected_clusters} "
            f"(auto max pairwise concept-profile distance={difference_score:.4f})"
        )
    print("Displayed representative trajectories:")
    for cluster_id, trajectory_rank, count, source_index, dedup_index, distance, _ in panels:
        print(
            f"  Cluster {cluster_id}, T{trajectory_rank}: "
            f"cluster_n={count}, source_index={source_index}, "
            f"dedup_index={dedup_index}, distance_to_cluster_profile={distance:.4f}"
        )
    print(f"Wrote heatmap: {args.output.resolve()}")


if __name__ == "__main__":
    main()
