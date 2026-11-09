from pathlib import Path
import argparse
import json
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from mutils import misc
from paths import GEN_RESULTS


def get_results(remaining_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True)
    parser.add_argument("--datasets_root", type=str, default="./_datasets/Classification_3D/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        '--accum_iter', default=1, type=int,
        help='Accumulate gradient iterations (for increasing the effective'
            ' batch size under memory constraints). (default: %(default)s)'
    )
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results.')
    parser.add_argument(
        '--metrics_only', action='store_true',
        help='Only compute and save the retrieval metrics (no t-SNE, no qualitative examples).',
    )
    args = parser.parse_args(remaining_args)
    misc.fix_seeds(args.seed)
    dataset_name = Path(args.root).parent.stem
    model_name = Path(args.root).stem
    features_fn = (
        Path(args.root) / "cached_features.npz"
    )
    assert features_fn.exists(), features_fn
    args.datasets_root = Path(args.datasets_root)
    args.output_dir = Path("./__outputs_feature_analysis")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = np.load(features_fn)
    features_train = torch.tensor(features["features_train"])[:, 0, 0]
    targets_train = torch.tensor(features["targets_train"])[:, 0]
    features_val = torch.tensor(features["features_val"])[:, 0, 0]
    targets_val = torch.tensor(features["targets_val"])[:, 0]
    features_test = torch.tensor(features["features_test"])[:, 0, 0]
    targets_test = torch.tensor(features["targets_test"])[:, 0]
    num_classes = features["num_classes"].item()
    print(features_train.shape, targets_train.shape)
    print(features_val.shape, targets_val.shape)
    print(features_test.shape, targets_test.shape)
    print("num_classes", num_classes)
    if not args.metrics_only:
        dataset_dir = args.datasets_root / dataset_name
        classes_list = sorted((dataset_dir / "train").glob("*/"))
        def get_samples_ordered(subset):
            samples = []
            root = dataset_dir / subset
            for cls in sorted(root.iterdir()):
                if not cls.is_dir():
                    continue
                for vol in sorted(cls.iterdir()):
                    if vol.suffix == '.npz':
                        samples.append(vol.stem)
            return samples
        sample_ids_train = get_samples_ordered("train")
        sample_ids_val   = get_samples_ordered("val")
        sample_ids_test  = get_samples_ordered("test")
        ####### t-SNE
        class_names = [c.stem for c in classes_list]
        all_features = torch.cat([features_train, features_val, features_test], dim=0)
        all_targets  = torch.cat([targets_train,  targets_val,  targets_test],  dim=0)
        print(f"Running t-SNE on {len(all_features)} samples...")
        embeddings = TSNE(
            n_components=2,
            perplexity=30,
            random_state=args.seed,
            n_jobs=-1,
        ).fit_transform(all_features.numpy())
        fig, ax = plt.subplots(figsize=(12, 10))
        for class_idx in range(num_classes):
            mask = all_targets.numpy() == class_idx
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                label=class_names[class_idx],
                s=10,
                alpha=0.6,
            )
        ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.set_title(f"{model_name} – t-SNE (train + val + test)")
        fig.tight_layout()
        fig.patch.set_alpha(0)
        ax.set_facecolor('white')
        fig.savefig(args.output_dir / f"{dataset_name}_{model_name}_tsne.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        metadata_fn = dataset_dir / 'metadata.json'
        if metadata_fn.exists():
            with open(metadata_fn, 'r') as f:
                metadata = json.load(f)
            all_ids = sample_ids_train + sample_ids_val + sample_ids_test
            # Extract vendor for each sample, in the same order as embeddings
            vendors = [
                metadata[sid]["Vendor"] if sid in metadata else "Unknown"
                for sid in all_ids
            ]
            vendors = np.array(vendors)
            unique_vendors = sorted(set(vendors))
            fig, ax = plt.subplots(figsize=(12, 10))
            for vendor in unique_vendors:
                mask = vendors == vendor
                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    label=vendor,
                    s=10,
                    alpha=0.6,
                )
            ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax.set_title(f"{model_name} – t-SNE by vendor (train + val + test)")
            fig.tight_layout()
            fig.patch.set_alpha(0)
            ax.set_facecolor('white')
            fig.savefig(args.output_dir / f"{dataset_name}_{model_name}_tsne_vendor.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Vendors found: {unique_vendors}")
    ####### Retrieval
    K = [1, 5, 10]
    # Normalize features for cosine similarity
    def l2_normalize(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    train_feats = l2_normalize(features_train).numpy()  # (N_train, D)
    train_labels = targets_train.numpy()
    query_feats  = l2_normalize(torch.cat([features_val, features_test], dim=0)).numpy()
    query_labels = torch.cat([targets_val, targets_test], dim=0).numpy()
    print(f"Computing cosine similarity ({len(query_feats)} queries x {len(train_feats)} gallery)...")
    sim_matrix = cosine_similarity(query_feats, train_feats)  # (N_query, N_train)
    results = {}
    for k in K:
        top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:]  # (N_query, k)
        top_k_labels  = train_labels[top_k_indices]              # (N_query, k)
        correct = (top_k_labels == query_labels[:, None]).any(axis=1)
        acc = correct.mean() * 100
        results[k] = acc
        print(f"  Top-{k} accuracy: {acc:.2f}%")
    # Also compute mAP (mean Average Precision)
    def compute_map(sim_matrix, query_labels, train_labels):
        num_queries = sim_matrix.shape[0]
        average_precisions = []
        for i in range(num_queries):
            query_label = query_labels[i]
            similarities = sim_matrix[i]
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_labels = train_labels[sorted_indices]
            # Compute precision at each rank
            relevant = (sorted_labels == query_label).astype(int)
            cumulative_relevant = np.cumsum(relevant)
            precision_at_k = cumulative_relevant / (np.arange(len(relevant)) + 1)
            # Average Precision for this query
            if relevant.sum() > 0:
                average_precision = (precision_at_k * relevant).sum() / relevant.sum()
                average_precisions.append(average_precision)
        return np.mean(average_precisions) * 100 if average_precisions else 0.0
    map = compute_map(sim_matrix, query_labels, train_labels)
    # Save results
    results_fn = args.output_dir / "retrieval.csv"
    # Columns: "model_name", "dataset_name", "top_1_acc", "top_5_acc", "top_10_acc", "mAP"
    df = pd.DataFrame([{
        "model_name": model_name,
        "dataset_name": dataset_name,
        "top_1_acc": results[1],
        "top_5_acc": results[5],
        "top_10_acc": results[10],
        "mAP": map,
    }])
    if results_fn.exists() and not args.overwrite:
        df_existing = pd.read_csv(results_fn)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(results_fn, index=False)
    print(f"Saved to {results_fn}")
    if args.metrics_only:
        return
    ####### Qualitative Retrieval Examples
    N_QUERIES = 5
    N_NEIGHBORS = 5
    rng = np.random.default_rng(args.seed)
    # Build index: sample_id -> (class_idx, subset)
    def build_id_map(subset):
        id_map = {}
        for class_dir in classes_list:
            target = dataset_dir / subset / class_dir.stem
            if not target.is_dir():
                continue
            for p in target.rglob("*"):
                if p.is_file() and p.suffix.lower() == ".npz":
                    id_map[p.stem] = (class_dir.stem, p)
        return id_map
    id_map_train = build_id_map("train")
    id_map_query = build_id_map("val")
    id_map_query.update(build_id_map("test"))
    def load_central_bscan(path: Path) -> np.ndarray:
        vol = np.load(path)["vol"]  # (N_slices, H, W)
        return vol[vol.shape[0] // 2]
    # Pick random query indices
    query_indices = rng.choice(len(query_feats), size=N_QUERIES, replace=False)
    fig, axes = plt.subplots(
        N_QUERIES, 1 + N_NEIGHBORS,
        figsize=(3 * (1 + N_NEIGHBORS), 3 * N_QUERIES),
    )
    all_query_ids = sample_ids_val + sample_ids_test
    for row, q_idx in enumerate(query_indices):
        q_id    = all_query_ids[q_idx]
        q_class, q_path = id_map_query[q_id]
        # Top-k neighbors from train
        top_k_idx = np.argsort(sim_matrix[q_idx])[::-1][:N_NEIGHBORS]
        # Query image
        ax = axes[row, 0]
        ax.imshow(load_central_bscan(q_path), cmap="gray")
        ax.set_title(f"Query\n{q_class}", fontsize=7, color="steelblue")
        ax.axis("off")
        # Blue border for query
        for spine in ax.spines.values():
            spine.set_edgecolor("steelblue")
            spine.set_linewidth(3)
            spine.set_visible(True)
        # Neighbor images
        for col, n_idx in enumerate(top_k_idx):
            n_id    = sample_ids_train[n_idx]
            n_class, n_path = id_map_train[n_id]
            score   = sim_matrix[q_idx, n_idx]
            correct = n_class == q_class
            ax = axes[row, col + 1]
            ax.imshow(load_central_bscan(n_path), cmap="gray")
            ax.set_title(f"#{col+1} ({score:.2f})\n{n_class}", fontsize=7,
                         color="green" if correct else "red")
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor("green" if correct else "red")
                spine.set_linewidth(3)
                spine.set_visible(True)
    fig.suptitle(f"{model_name} – Retrieval examples", fontsize=12)
    fig.tight_layout()
    fig.patch.set_alpha(0)
    for a in axes.flat:
        a.set_facecolor('white')
    fig.savefig(args.output_dir / f"{dataset_name}_{model_name}_retrieval_qualitative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved qualitative retrieval plot.")


def load_features_and_targets(root, seed):
    """Shared loading logic used by run_tsne, run_retrieval, and
    analyze_class_similarity."""
    misc.fix_seeds(seed)
    dataset_name = Path(root).parent.stem
    model_name = Path(root).stem
    features_fn = Path(root) / "cached_features.npz"
    assert features_fn.exists(), features_fn
    features = np.load(features_fn)
    features_train = torch.tensor(features["features_train"])[:, 0, 0]
    targets_train = torch.tensor(features["targets_train"])[:, 0]
    features_val = torch.tensor(features["features_val"])[:, 0, 0]
    targets_val = torch.tensor(features["targets_val"])[:, 0]
    features_test = torch.tensor(features["features_test"])[:, 0, 0]
    targets_test = torch.tensor(features["targets_test"])[:, 0]
    num_classes = features["num_classes"].item()
    return {
        'dataset_name': dataset_name,
        'model_name': model_name,
        'features_train': features_train,
        'targets_train': targets_train,
        'features_val': features_val,
        'targets_val': targets_val,
        'features_test': features_test,
        'targets_test': targets_test,
        'num_classes': num_classes,
    }


def get_class_names(datasets_root, dataset_name):
    dataset_dir = Path(datasets_root) / dataset_name
    classes_list = sorted((dataset_dir / "train").glob("*/"))
    return [c.stem for c in classes_list], dataset_dir


def get_samples_ordered(dataset_dir, subset):
    samples = []
    root = dataset_dir / subset
    for cls in sorted(root.iterdir()):
        if not cls.is_dir():
            continue
        for vol in sorted(cls.iterdir()):
            if vol.suffix == '.npz':
                samples.append(vol.stem)
    return samples
model_names = {
    "octcube3d_60_linear_w_cls_patch": "OCTCube",
    "CLIFO3D_linear_w_2avg": "CLIFO",
    "CLIFO3D_60_linear_w_2avg": "CLIFO",
}
GEN_DIR = Path(__file__).resolve().parent / "__gen_results" / "supp" / "feature_analysis"   # t-SNE and dendrogram figures


def run_tsne_intelli(remaining_args):
    """
    Loads cached features and produces t-SNE visualizations (by class,
    and by vendor if metadata is available). Split out from the
    combined get_results function for clarity.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True)
    parser.add_argument("--datasets_root", type=str, default="./_datasets/Classification_3D/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30)
    args = parser.parse_args(remaining_args)
    intelli_classes = {
        "Healthy": "Control",
        "Normal": "Control",
        "Multiple_Sclerosis": "MS",
        "NORMAL": "Control",
        "control": "Control",
        "early": "Early glaucoma",
        "mid_advanced": "Int./Adv. glaucoma",
        "non": "Control",
        "DryAMD": "Dry AMD",
        "WetAMD": "Wet AMD",
        "NonAMD": "Control",
        "MacularHole_123": "MH (1-3)",
        "MacularHole_4": "MH (4)",
        "ME_DR": "ME and DR",
        "glaucoma": "Glaucoma",
        "cnv1": "CNV-1",
        "cnv2": "CNV-2",
        "cnv3": "CNV-3",
        "dme": "DME",
        "ga": "GA",
        "healthy": "Control",
        "iamd": "iAMD",
        "rvo": "RVO",
        "stargardt": "Stargardt",
        "DRU": "Drusen",
    }
    data = load_features_and_targets(args.root, args.seed)
    dataset_name, model_name = data['dataset_name'], data['model_name']
    class_names, dataset_dir = get_class_names(args.datasets_root, dataset_name)
    output_dir = GEN_RESULTS / 'retrieval'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = GEN_RESULTS / 'retrieval'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = GEN_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    all_features = torch.cat([data['features_train'], data['features_val'], data['features_test']], dim=0)
    all_targets = torch.cat([data['targets_train'], data['targets_val'], data['targets_test']], dim=0)
    print(f"Running t-SNE on {len(all_features)} samples...")
    embeddings = TSNE(
        n_components=2, perplexity=args.perplexity, random_state=args.seed, n_jobs=-1,
    ).fit_transform(all_features.numpy())
    # --- t-SNE colored by class ---
    model_name_disp = model_names.get(model_name, model_name)
    model_name_disp = model_name_disp.replace('CLIFO', 'CLIFO')
    model_name_save = model_name_disp
    if "CLIFO" in model_name_disp:
         model_name_save = 'OURS'
    bullet_size = 10
    figsize = (4, 4)
    fig, ax = plt.subplots(figsize=figsize)
    for class_idx in range(data['num_classes']):
        mask = all_targets.numpy() == class_idx
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], label=intelli_classes.get(class_names[class_idx], class_names[class_idx]), s=bullet_size, alpha=0.6)
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    # Disable axis numbers
    ax.set_xticks([])
    ax.set_yticks([])
    # If OCTCube, do not show legend
    if model_name_disp == "OCTCube":
        ax.legend().set_visible(False)
    ax.set_title(f"{model_name_disp}", fontweight="bold" if "CLIFO" in model_name_disp else "normal")
    fig.patch.set_alpha(0)
    ax.set_facecolor('white')
    fig.savefig(output_dir / f"{dataset_name}_{model_name_save}_tsne.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- t-SNE colored by vendor, if metadata available ---
    metadata_fn = dataset_dir / 'metadata.json'
    if metadata_fn.exists():
        sample_ids_train = get_samples_ordered(dataset_dir, "train")
        sample_ids_val = get_samples_ordered(dataset_dir, "val")
        sample_ids_test = get_samples_ordered(dataset_dir, "test")
        all_ids = sample_ids_train + sample_ids_val + sample_ids_test
        with open(metadata_fn, 'r') as f:
            metadata = json.load(f)
        vendors = np.array([metadata[sid]["Vendor"] if sid in metadata else "Unknown" for sid in all_ids])
        unique_vendors = sorted(set(vendors))
        fig, ax = plt.subplots(figsize=(12, 10))
        for vendor in unique_vendors:
            mask = vendors == vendor
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], label=vendor, s=10, alpha=0.6)
        ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.set_title(f"{model_name} – t-SNE by vendor (train + val + test)")
        fig.tight_layout()
        fig.patch.set_alpha(0)
        ax.set_facecolor('white')
        fig.savefig(output_dir / f"{dataset_name}_{model_name_save}_tsne_vendor.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Vendors found: {unique_vendors}")
    print(f"Saved t-SNE plots to {output_dir}")


# Module-level friendly class-name mapping, reused across t-SNE and
# dendrogram plots so labels are consistent everywhere.
INTELLI_CLASS_NAMES = {
    "Healthy": "Control", "Normal": "Control", "Multiple_Sclerosis": "MS",
    "NORMAL": "Control", "control": "Control", "early": "Early glaucoma",
    "mid_advanced": "Int./Adv. glaucoma", "non": "Control",
    "DryAMD": "Dry AMD", "WetAMD": "Wet AMD", "NonAMD": "Control",
    "MacularHole_123": "MH (1-3)", "MacularHole_4": "MH (4)",
    "ME_DR": "ME and DR", "glaucoma": "Glaucoma",
    "cnv1": "CNV-1",
    "cnv2": "CNV-2",
    "cnv3": "CNV-3",
    "dme": "DME",
    "ga": "GA",
    "healthy": "Control",
    "iamd": "iAMD",
    "rvo": "RVO",
    "stargardt": "Stargardt",
    "DRU": "Drusen",
}


def analyze_class_similarity(remaining_args):
    """
    Computes pairwise cosine similarity between class centroids in the
    raw (non-reduced) feature space, plus a hierarchical clustering
    dendrogram (a t-SNE-independent, quantitative way to check whether
    classes are organized in a clinically meaningful order).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True)
    parser.add_argument("--datasets_root", type=str, default="./_datasets/Classification_3D/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold_frac", type=float, default=0.7,
                         help="Fraction of the max linkage height used as "
                              "the color_threshold (scipy default is 0.7). "
                              "Lower = more/finer colored clusters.")
    args = parser.parse_args(remaining_args)

    data = load_features_and_targets(args.root, args.seed)
    dataset_name, model_name = data['dataset_name'], data['model_name']
    model_name_disp = model_names.get(model_name, model_name)
    model_name_disp = model_name_disp.replace('CLIFO', 'CLIFO')
    model_name_save = model_name_disp
    if "CLIFO" in model_name_disp:
         model_name_save = 'OURS'
    class_names, _ = get_class_names(args.datasets_root, dataset_name)
    display_names = [INTELLI_CLASS_NAMES.get(c, c) for c in class_names]

    output_dir = GEN_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = torch.cat([data['features_train'], data['features_val'], data['features_test']], dim=0).numpy()
    all_targets = torch.cat([data['targets_train'], data['targets_val'], data['targets_test']], dim=0).numpy()

    centroids = []
    for c in range(data['num_classes']):
        mask = all_targets == c
        centroids.append(all_features[mask].mean(axis=0))
    centroids = np.stack(centroids)

    sim_matrix = cosine_similarity(centroids)
    sim_df = pd.DataFrame(sim_matrix, index=class_names, columns=class_names)
    print(f"\n{model_name_disp}, class centroid cosine similarity ({dataset_name}):")
    print(sim_df.round(3))

    csv_fn = output_dir / f"{dataset_name}_{model_name_save}_class_similarity.csv"
    sim_df.to_csv(csv_fn)
    print(f"Saved similarity matrix to {csv_fn}")

    # --- Dendrogram ---
    dist_matrix = 1 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)  # avoid tiny negative values from float error
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='average')

    FONT_SIZE = 14
    plt.rcParams.update({"font.size": FONT_SIZE})

    fig, ax = plt.subplots(figsize=(7, 5))
    color_threshold = args.threshold_frac * Z[:, 2].max()
    ddata = dendrogram(
        Z,
        labels=display_names,
        ax=ax,
        color_threshold=color_threshold,
        leaf_rotation=45,
        leaf_font_size=FONT_SIZE,
    )
    ax.set_xticklabels(ax.get_xticklabels(), ha="right")

    # Color each leaf label to match its own link color, as computed by
    # scipy's default (distance-based) coloring -- no fixed k, so the
    # number of distinct colors reflects how well-separated the classes
    # actually are in feature space.
    leaves_color_list = ddata['leaves_color_list']  # one color per leaf, in plotted order
    for tick_label, color in zip(ax.get_xticklabels(), leaves_color_list):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    ax.set_title(model_name_disp, fontsize=FONT_SIZE + 3, fontweight="bold" if "CLIFO" in model_name_disp else "normal")
    ax.set_ylabel("Cosine distance", fontsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    fig.tight_layout()
    fig.patch.set_alpha(0)
    ax.set_facecolor('white')

    fig_fn = output_dir / f"{dataset_name}_{model_name_save}_class_dendrogram.pdf"
    fig_fn.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_fn, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dendrogram to {fig_fn}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str)
    args, extras = parser.parse_known_args()
    globals()[args.function](extras)
