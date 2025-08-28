import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import math
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import torch
from pathlib import Path
from collections import defaultdict

# optional extras
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

# def plot_loss(param_combinations, train_loss_per_param, val_loss_per_param, plot_dir, plot_name):
#     plt.figure(figsize=(8, 6))
#     for i, param in enumerate(param_combinations):
#         plt.plot(train_loss_per_param[i], label=f'Train ({param})', linestyle='-', linewidth=2)
#         plt.plot(val_loss_per_param[i], label=f'Validation ({param})', linestyle='--', linewidth=2)
#
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title(f'Training and Validation Losses for Different Hyperparameter Combinations')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), fontsize='small')
#
#     plot_path = os.path.join(plot_dir, plot_name)
#     plt.savefig(plot_path)
#     print(f"Saved loss plots as PDF files at {plot_path}.")

# def plot_auroc_auprc(type, param_combinations, values, plot_dir, name):
#     plt.figure(figsize=(8, 6))
#     for i, param in enumerate(param_combinations):
#         plt.plot(values[i], label=f'{type} ({param})', linestyle=':')
#     plt.xlabel('Epochs')
#     plt.ylabel(type)
#     plt.title(f'{type} for Different Parameter Combinations')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), fontsize='small')
#     plt.grid(True)
#     plot_path = os.path.join(plot_dir, f'{type}_{name}.pdf')
#     plt.savefig(plot_path)
#     print(f"Saved loss plots as PDF files at {plot_path}.")

def plot_roc_pr(type, true_labels, probs, x_name, y_name, plot_dir, name):
    plt.clf()
    if type == 'ROC':
        fpr, tpr, thresholds = roc_curve(true_labels, probs)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUROC=0.5)')
        x, y = fpr, tpr

        # optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)

    elif type == 'PR':
        precision, recall, thresholds = precision_recall_curve(true_labels, probs)
        x, y = recall, precision
        # Add baseline
        pos_proportion = np.mean(true_labels)
        plt.axhline(y=pos_proportion, color='gray', linestyle='--', label=f'Random (AUPR={pos_proportion:.2f})')

        # optimal threshold (closest to (1,1))
        dist_to_corner = (1 - recall) ** 2 + (1 - precision) ** 2
        optimal_idx = np.argmin(dist_to_corner)
    else:
        raise ValueError("Type must be 'ROC' or 'PR'")

    au = auc(x, y)

    optimal_threshold = thresholds[optimal_idx]
    plt.scatter(x[optimal_idx], y[optimal_idx], marker='o', color='red',
                label=f'Optimal Threshold = {optimal_threshold:.2f}')

    plt.plot(x, y, color='b', lw=2, label=f'AU = {au:.2f}')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f'{type} Curve')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f'{type}-{name}.pdf')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved {type} as PDF file at {plot_path}.")


# visualizes the embeddings from the different splits (train, val, test)
# using PCA, T-SNE, and UMAP
def visualize_embeddings(split_embeddings, output_dir, model_name):
    """
    split_embeddings : {train: train_embed, test: test_embed}
    each embed is a dict of keys: "sequences", "embeddings", "labels"
      train_embed embeddings of shape (N_train,  emb_size)
      test_embed embeddings of shape (N_test,   emb_size)

    do dimension reduction for each embedding based on 1."PCA" 2."T-SNE" 3."UMAP"
    shape code each label
    color code each split
    """
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)
    umap_model = umap.UMAP(n_components=2)

    # define different colors for train and test
    colors = {
        'train': '#AEC6CF',  # pastel blue
        'test': '#FFB347',  # pastel orange
    }
    # define different markers for labels 0 and 1
    markers = {0: 'o', 1: '^'}

    # loop through each dimensionality reduction method
    for method, model in zip(["PCA", "T-SNE", "UMAP"], [pca, tsne, umap_model]):
        plt.figure(figsize=(8, 6))

        # loop through each split
        for split, embed_data in split_embeddings.items():
            embeddings = embed_data['embeddings']
            labels = embed_data['labels']

            # ensure embeddings are on CPU and convert to numpy array
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()


            if isinstance(labels, list):
                # convert labels to numpy array if they are a list
                try:
                    labels = np.concatenate([np.array(label) for label in labels], axis=0)
                except ValueError:
                    raise ValueError(f"Error: Inconsistent label shapes in {split}.")
            elif isinstance(labels, torch.Tensor):
                # move tensor to CPU and convert to numpy array
                labels = labels.cpu().numpy()

            # apply dimensionality reduction
            reduced_embeddings = model.fit_transform(embeddings)

            # Scatter plot
            for label in np.unique(labels):  # loop over unique labels (0 and 1)
                label_indices = np.where(labels == label)[0]
                plt.scatter(
                    reduced_embeddings[label_indices, 0], reduced_embeddings[label_indices, 1],
                    c=[colors[split]] * len(label_indices),  # Use color for the split
                    marker=markers[label],  # Use different marker for label 0 and 1
                    s=5
                )

        plt.title(f'{method} Embedding Visualization for {model_name}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

        # separate legend for color (split)
        for split, color in colors.items():
            plt.scatter([], [], color=color, label=f'Split: {split}')

        # separate legend for shapes (labels)
        for label, marker in markers.items():
            plt.scatter([], [], color='black', marker=marker, label=f'Label: {label}')

        plt.legend()

        # save the plot
        output_path = os.path.join(output_dir, method)
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f"{model_name}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

def plot_promoter_position_probability(predictions, sequence_length, abs_start, output_dir, name,
                         strand="+",
                         agg="mean",                 # 'mean', 'max', or 'median'
                         min_coverage=1,             # mask positions covered by < this many windows
                         smooth="gaussian",          # None | 'moving' | 'gaussian' | 'savgol' | 'spline'
                         smooth_param=51,            # window for moving/savgol; ~bandwidth for gaussian/spline
                         savgol_poly=2):
    """
    predictions: list of (start, end, prob), 0- or 1-based — adjust ranges accordingly
    sequence_length: int
    smooth_param:
        moving    -> window length (odd recommended)
        gaussian  -> sigma in positions (e.g., ~ window_size/2)
        savgol    -> window length (must be odd)
        spline    -> smoothing factor 's' (bigger => smoother)
    """

    # 1) collect probabilities per position
    probs_by_pos = defaultdict(list)

    for start, end, prob in predictions:
        s = max(0, int(start))
        e = min(sequence_length - 1, int(end))
        if e >= s:
            for pos in range(s, e + 1):
                probs_by_pos[pos].append(prob)

    # 2) aggregate
    agg_probs = np.full(sequence_length, np.nan, dtype=float)
    counts = np.zeros(sequence_length, dtype=int)

    for pos, plist in probs_by_pos.items():
        counts[pos] = len(plist)
        if counts[pos] < min_coverage:
            continue
        if agg == "mean":
            agg_probs[pos] = np.mean(plist)
        elif agg == "max":
            agg_probs[pos] = np.max(plist)
        elif agg == "median":
            agg_probs[pos] = np.median(plist)
        else:
            raise ValueError(f"Unknown agg='{agg}', must be mean|max|median")

    # 3) smoothing
    x = np.arange(sequence_length)
    y = agg_probs.copy()
    y_smooth = y.copy()
    if smooth and np.nanmax(counts) > 0:
        # interpolate small gaps so filters work (don’t invent at long uncovered stretches)
        # linear interp over NaNs
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            y_interp = np.interp(x, x[mask], y[mask])
        else:
            y_interp = y.copy()

        if smooth == 'moving':
            w = max(3, int(smooth_param) // 2 * 2 + 1)  # force odd
            kernel = np.ones(w) / w
            y_s = np.convolve(y_interp, kernel, mode='same')

        elif smooth == 'gaussian':
            sigma = float(smooth_param)
            y_s = gaussian_filter1d(y_interp, sigma=sigma, mode='nearest')

        elif smooth == 'savgol':
            w = max(5, int(smooth_param) // 2 * 2 + 1)  # odd and >= poly+2
            y_s = savgol_filter(y_interp, window_length=w, polyorder=savgol_poly, mode='interp')

        elif smooth == 'spline':
            # s ≈ (sequence_length) / smooth_param is a handy heuristic, or just pass a fixed s
            s = float(smooth_param)
            spl = UnivariateSpline(x[mask], y[mask], s=s)
            y_s = spl(x)

        else:
            y_s = y_interp

        # keep masked areas as NaN so they don't draw
        y_smooth = np.where(np.isnan(y), np.nan, y_s)

    # 3) plot smoothed
    plt.figure(figsize=(18, 6))
    plt.plot(x, agg_probs, linewidth=1, alpha=0.5, label=f'{agg} probability')
    plt.plot(x, y_smooth, linewidth=2, label=f'smoothed ({smooth})')
    plt.axhline(0.5, linestyle='--', linewidth=1, label='Random Predictor')

    # 4) plot actual probabilities
    for start, end, prob in predictions:
        s = max(0, int(start))
        e = min(sequence_length - 1, int(end))
        if e >= s:
            plt.hlines(prob, s, e, colors="gray", alpha=0.3, linewidth=1)

    plt.title(f'Positional Distribution of Predicted Probability for {name}')
    plt.xlabel(f'Sequence Position ({abs_start}-{abs_start+sequence_length}:"{strand}")')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.legend()

    # change x-axis display
    import matplotlib.ticker as ticker
    def make_rel_pos_formatter(strand):
        if strand == "+":  # plus strand: -1000 .. +200
            def formatter(val, pos):
                rel = -1000 + int(val)  # index 0 = -1000, index 1200 = +200
                return f"{rel:+d}"
        else:  # minus strand: +200 .. -1000
            def formatter(val, pos):
                rel = 200 - int(val)  # index 0 = +200, index 1200 = -1000
                return f"{rel:+d}"
        return formatter

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(make_rel_pos_formatter(strand)))

    output_dir = os.path.join(output_dir, agg)
    output_dir = os.path.join(output_dir, f"smooth_{smooth}")
    os.makedirs(output_dir, exist_ok=True)

    out = os.path.join(output_dir, f"positional-dist-{name}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    return out

def plot_group_saliency_maps_per_class(logger, npz_file_path, output_dir):
    """
    Load data from an .npz and produce two saliency grid plots:
      - one for samples predicted correctly
      - one for samples predicted wrongly

    Each row corresponds to a sample; x is sequence position; color = summed attribution.
    The per-sample [start, end] interval is highlighted on its row.
    The y-axis shows a friendly sequence name parsed from `uid`.
    """
    npz_path = Path(npz_file_path)
    if not npz_path.is_file():
        logger.log_message(f"Error: File not found – {npz_path}")
        return

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            attributions = data['attributions']       # (L, C) per sample
            sequences = data['sequences']             # unused here, but loaded for parity
            labels = data['labels']                   # not used for grouping in this function
            uids = data['uids']
            peak_start_ends = data['peak_start_ends'] # (start, end) per sample
            correct_labels = data['correct_labels']   # bool/int per sample: 1=correct, 0=wrong
    except KeyError as e:
        logger.log_message(f"Error: Missing expected array in NPZ – {e}")
        return
    except Exception as e:
        logger.log_message(f"Error loading {npz_path}: {e}")
        return

    num_samples = len(attributions)
    if num_samples == 0:
        logger.log_message("No samples found in the file.")
        return

    logger.log_message(f"{num_samples} samples found; assembling correctness grids…")

    # Precompute 1D saliency (sum over channels) and lengths
    seq_scores_list = []
    lengths = []
    for attrs in attributions:  # attrs shape (L, C)
        seq_scores = attrs.sum(axis=1)
        seq_scores_list.append(seq_scores)
        lengths.append(len(seq_scores))

    lengths = np.array(lengths)
    max_len = int(lengths.max())

    # Indices grouped by correctness
    idx_correct = [i for i, c in enumerate(correct_labels) if bool(c)]
    idx_wrong   = [i for i, c in enumerate(correct_labels) if not bool(c)]

    def build_matrix(indices, desired_center_col=None):
        """
        indices: list of sample indices to include in this grid.

        Returns:
            M: (n, W) masked array of scores (NaN-masked where padded)
            row_names: list of y-axis labels (parsed from uid)
            row_spans: list of (start, end) in row order, shifted to grid coords
            row_lengths: list of lengths in row order
        """

        def peak_stats(seq_scores, start, end, mode="abs"):
            """Compute relative contribution of the peak vs rest for one sequence."""
            L = len(seq_scores)
            start = max(0, min(int(start), L - 1))
            end = max(0, min(int(end), L - 1))
            if end < start:
                start, end = end, start

            if mode == "pos":
                s_use = np.clip(seq_scores, 0, None)
            elif mode == "abs":
                s_use = np.abs(seq_scores)
            elif mode == "raw":
                s_use = seq_scores
            else:
                raise ValueError("mode must be 'pos', 'abs', or 'raw'")

            peak = s_use[start:end + 1]
            mask = np.ones(L, dtype=bool);
            mask[start:end + 1] = False
            off = s_use[mask]

            eps = 1e-12
            peak_sum = peak.sum()
            off_sum = off.sum() if off.size else 0.0
            total = peak_sum + off_sum + eps

            peak_mean = peak.mean()
            off_mean = off.mean() if off.size else 0.0
            peak_fraction = peak_sum / total  # length-sensitive
            fold_change = (peak_mean + eps) / (off_mean + eps)  # mean_ratio, length-invariant
            log2fc = np.log2(fold_change) if off.size else np.inf  # around 0 → easy to see “no change.”

            length_share = len(peak) / L
            concentration = (peak_fraction + eps) / (length_share + eps)  # length-normalized focus

            return peak_mean, log2fc, concentration

        if len(indices) == 0:
            return None, [], [], []

        def _name_from_uid(uid):
            """
            Best-effort extractor for a human-friendly sequence name from `uid`.
            - If uid is a dict, try common fields.
            - If uid looks like a path, use basename (without extension).
            - Else, try splitting on common separators and take the first chunk.
            - Fallback: str(uid)
            """
            try:
                # dict-like
                if isinstance(uid, dict):
                    for k in ("name", "sequence_name", "seq_name", "uid", "id"):
                        if k in uid and uid[k]:
                            return str(uid[k])
                    return str(uid)

                s = str(uid.detach().cpu().numpy().astype(str).item())

                # path-like
                if any(sep in s for sep in ("/", "\\")):
                    base = os.path.basename(s)
                    if "." in base:
                        base = ".".join(base.split(".")[:-1]) or base
                    return base

                # split on common separators
                for sep in ("||", " | ", "|", "::", ":", ";", ","):
                    if sep in s:
                        return s.split(sep)[0]

                # if there's whitespace, take the first token
                if " " in s:
                    return s.split()[0]

                return s
            except Exception:
                return str(uid)

        row_names, row_spans, row_lengths = [], [], []
        rows = []  # store (seq_scores, L, start, end, name)
        for i in indices:
            s = seq_scores_list[i]  # shape (L,)
            L = len(s)
            start, end = map(int, peak_start_ends[i])
            c = 0.5 * (start + end)  # center of the peak

            peak_mean, log2fc, concentration = peak_stats(s, start, end)
            label = f"{_name_from_uid(uids[i])} [log2fc={log2fc:.2f}, Cons={concentration:.2f}]"

            rows.append((log2fc, peak_mean, s, L, start, end, label, c))

        # sort rows by log2fc (descending)
        rows.sort(key=lambda tup: tup[0], reverse=True)

        centers = [tup[-1] for tup in rows]
        n = len(rows)
        # If the user didn't pick a column, target the center of the figure.
        # We'll first compute offsets, then derive total width and the exact center column.
        # Start with preliminary desired_center_col if provided; otherwise we’ll set later.
        if desired_center_col is None:
            # We’ll set this after computing the final width so the column is truly centered.
            target_col = None
        else:
            target_col = int(desired_center_col)

        # Initial integer offsets to place each row so that center -> target_col.
        # If target_col is None, we still compute relative offsets (we’ll shift them later).
        rough_offsets = []
        for c in centers:
            off = 0 if target_col is None else int(round(target_col - c))
            rough_offsets.append(off)

        # Ensure all offsets are non-negative by applying a global shift if needed
        min_off = min(rough_offsets) if rough_offsets else 0
        global_shift = -min_off if min_off < 0 else 0
        offsets = [off + global_shift for off in rough_offsets]

        # Compute width needed so no row overflows on the right
        widths_needed = [off + L for off, (_, _, _,  L, _, _, _, _) in zip(offsets, rows)]
        W = max(widths_needed) if widths_needed else 0

        # If target_col was not set, choose the true middle column of the final width
        if target_col is None:
            target_col = W // 2
            # Recompute offsets to align centers to this target_col, then re-fit W
            offsets = [int(round(target_col - c)) for c in centers]
            min_off = min(offsets)
            global_shift = -min_off if min_off < 0 else 0
            offsets = [off + global_shift for off in offsets]
            widths_needed = [off + L for off, (_, _, _,  L, _, _, _, _) in zip(offsets, rows)]
            W = max(widths_needed)

        # Build matrix and shifted spans
        M = np.full((n, W), np.nan, dtype=float)
        shifted_spans, out_names, out_lengths = [], [], []

        for r, (off, (_, _, s, L, start, end, name, _)) in enumerate(zip(offsets, rows)):
            # Place the row starting at offset
            M[r, off:off + L] = s
            # Shift the span by the same offset
            shifted_spans.append((start + off, end + off))
            out_names.append(name)
            out_lengths.append(L)

        M = np.ma.masked_invalid(M)
        return M, out_names, shifted_spans, out_lengths

    def plot_grid(M, row_names, row_spans, row_lengths, title, out_path):
        if M is None or M.shape[0] == 0:
            logger.log_message(f"No samples for {title}; skipping.")
            return

        # Color scale (per figure)
        vmin = float(M.min()) if np.isfinite(M.min()) else 0.0
        # vmax = float(M.max()) if np.isfinite(M.max()) else 1.0
        vmax = 100
        if math.isclose(vmin, vmax):
            vmin, vmax = vmin - 1e-6, vmax + 1e-6

        # Figure sizing
        fig_h = max(2.5, min(0.35 * M.shape[0] + 1.2, 20.0))
        fig_w = max(8.0, min(12.0, 0.02 * M.shape[1] + 6.0))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        im = ax.imshow(
            M,
            aspect='auto',
            interpolation='nearest',
            origin='upper',
            norm=Normalize(vmin=vmin, vmax=vmax),
            cmap='viridis'
        )

        # X axis
        ax.set_xlabel("Position")
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_xticks(np.linspace(0, M.shape[1]-1, num=min(11, M.shape[1])))

        # Y axis: sequence names
        n_rows = M.shape[0]
        ax.set_ylabel("Sequence")
        if n_rows <= 50:
            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels(row_names)
        else:
            show_idx = np.linspace(0, n_rows - 1, 50, dtype=int)
            ax.set_yticks(show_idx)
            ax.set_yticklabels([row_names[i] for i in show_idx])

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Summed attribution")

        # Row-wise highlight rectangles for [start, end]
        for r, ((start, end), L) in enumerate(zip(row_spans, row_lengths)):
            start_clamped = max(0, min(int(start), L-1))
            end_clamped = max(0, min(int(end), L-1))
            if end_clamped < start_clamped:
                start_clamped, end_clamped = end_clamped, start_clamped

            rect = Rectangle(
                (start_clamped - 0.5, r - 0.5),
                (end_clamped - start_clamped + 1),
                1.0,
                fill=True,
                alpha=0.18,
                edgecolor='none'
            )
            ax.add_patch(rect)

            outline = Rectangle(
                (start_clamped - 0.5, r - 0.5),
                (end_clamped - start_clamped + 1),
                1.0,
                fill=False,
                linewidth=0.6,
                color='white'
            )
            ax.add_patch(outline)

        ax.set_title(title)

        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.log_message(f"Saved {title} → {out_path}")

    # Build matrices and plot
    M_ok,  names_ok,  spans_ok,  lens_ok  = build_matrix(idx_correct)
    M_bad, names_bad, spans_bad, lens_bad = build_matrix(idx_wrong)

    out_ok  = os.path.join(output_dir, "saliency_grid_correct.png")
    out_bad = os.path.join(output_dir, "saliency_grid_wrong.png")

    plot_grid(M_ok,  names_ok,  spans_ok,  lens_ok,
              title=f"Saliency grid — predicted correctly (#samples={len(idx_correct)})",
              out_path=out_ok)

    plot_grid(M_bad, names_bad, spans_bad, lens_bad,
              title=f"Saliency grid — predicted wrongly (#samples={len(idx_wrong)})",
              out_path=out_bad)

    logger.log_message(f"Finished correctness-based group plots. Saved to {output_dir}.")