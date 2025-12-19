import numpy as np
import mne
import pandas as pd
from mne.time_frequency import psd_array_welch
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd


def annotate_high_amplitude_per_channel(
    raw: mne.io.BaseRaw,
    threshold: float,
    pre: float = 0.0,
    post: float = 0.0,
    desc: str = "BAD_amp",
) -> mne.Annotations:
    sfreq = raw.info["sfreq"]
    ch_idx = mne.pick_types(raw.info, eeg=True)
    data, times = raw[ch_idx, :]  # (n_ch, n_times)

    all_onsets = []
    all_durations = []
    all_desc = []
    all_ch_names = []

    for i, idx in enumerate(ch_idx):
        ch_name = raw.ch_names[idx]
        ch_data = data[i]

        bad_mask = np.abs(ch_data) > threshold
        if not np.any(bad_mask):
            continue

        bad_idx = np.where(bad_mask)[0]
        splits = np.where(np.diff(bad_idx) > 1)[0] + 1
        clusters = np.split(bad_idx, splits)

        for cl in clusters:
            t_start = times[cl[0]] - pre
            t_stop = times[cl[-1]] + post

            t_start = max(t_start, times[0])
            t_stop = min(t_stop, times[-1])
            if t_stop <= t_start:
                continue

            all_onsets.append(t_start)
            all_durations.append(t_stop - t_start)
            all_desc.append(desc)
            all_ch_names.append([ch_name])

    if not all_onsets:
        return mne.Annotations([], [], [], orig_time=raw.info["meas_date"])

    annot = mne.Annotations(
        onset=all_onsets,
        duration=all_durations,
        description=all_desc,
        ch_names=all_ch_names,
        orig_time=raw.info["meas_date"],
    )
    return annot


def merge_overlapping_annotations(annot: mne.Annotations) -> mne.Annotations:
    """Объединить пересекающиеся/соприкасающиеся аннотации
    с одинаковыми description и ch_names.
    """
    if len(annot) == 0:
        return annot

    onsets = np.asarray(annot.onset, float)
    durations = np.asarray(annot.duration, float)
    desc = np.asarray(annot.description, dtype=object)

    starts = onsets
    stops = onsets + durations

    has_ch_names = getattr(annot, "ch_names", None) is not None
    if has_ch_names:
        ch_names_list = [
            tuple(chs) if chs is not None and len(chs) > 0 else ()
            for chs in annot.ch_names
        ]
    else:
        ch_names_list = [()] * len(annot)

    groups = {}
    for i, (d, chs) in enumerate(zip(desc, ch_names_list)):
        key = (d, chs)
        groups.setdefault(key, []).append(i)

    merged_onsets = []
    merged_durations = []
    merged_desc = []
    merged_ch_names = []

    for (key_desc, key_chs), idxs in groups.items():
        idxs = np.array(idxs, int)
        s = starts[idxs]
        e = stops[idxs]

        order = np.argsort(s)
        s = s[order]
        e = e[order]

        cur_start = s[0]
        cur_end = e[0]

        for st, en in zip(s[1:], e[1:]):
            if st <= cur_end:
                cur_end = max(cur_end, en)
            else:
                merged_onsets.append(cur_start)
                merged_durations.append(cur_end - cur_start)
                merged_desc.append(key_desc)
                merged_ch_names.append(list(key_chs))
                cur_start, cur_end = st, en

        merged_onsets.append(cur_start)
        merged_durations.append(cur_end - cur_start)
        merged_desc.append(key_desc)
        merged_ch_names.append(list(key_chs))

    if not has_ch_names:
        merged = mne.Annotations(
            onset=merged_onsets,
            duration=merged_durations,
            description=merged_desc,
            orig_time=annot.orig_time,
        )
    else:
        merged = mne.Annotations(
            onset=merged_onsets,
            duration=merged_durations,
            description=merged_desc,
            ch_names=merged_ch_names,
            orig_time=annot.orig_time,
        )
    return merged


def create_channel_bad_masks(raw, picks, bad_prefix="bad"):
    """Матрица масок (n_picks, n_times): True = артефакт в данном канале/время."""
    sfreq = raw.info["sfreq"]
    n_times = raw.n_times
    picks = np.atleast_1d(picks)
    ch_names = np.array(raw.ch_names)

    bad_masks = np.zeros((len(picks), n_times), dtype=bool)

    ann = raw.annotations
    has_ch = getattr(ann, "ch_names", None) is not None

    for i_ann, a in enumerate(ann):
        desc = a["description"]
        if not desc.lower().startswith(bad_prefix):
            continue

        onset = float(a["onset"])
        duration = float(a["duration"])
        onset_sample = max(int(np.round(onset * sfreq)), 0)
        end_sample = min(
            onset_sample + int(np.round(duration * sfreq)),
            n_times,
        )
        if end_sample <= onset_sample:
            continue

        if has_ch:
            ann_chs = ann.ch_names[i_ann]
            if ann_chs is None or len(ann_chs) == 0:
                affected = np.ones(len(picks), dtype=bool)
            else:
                ann_chs = set(ann_chs)
                affected = np.array(
                    [ch_names[idx] in ann_chs for idx in picks], dtype=bool
                )
        else:
            affected = np.ones(len(picks), dtype=bool)

        bad_masks[affected, onset_sample:end_sample] = True

    return bad_masks


def compute_band_power_array(data, sfreq, fmin, fmax):
    """PSD Уэлча для 1D массива, интеграл мощности в диапазоне."""
    if data.size == 0:
        return np.nan

    n_per_seg = int(min(len(data), sfreq * 1.0))
    psd, freqs = psd_array_welch(
        data[np.newaxis, :],
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_per_seg=n_per_seg,
        average="mean",
        verbose=False,
    )
    return np.trapz(psd[0], freqs)


def sliding_band_powers_to_df_per_channel(
    raw,
    win_len=20.0,
    step=10.0,
    bands=None,
    min_clean_ratio=0.5,
    bad_prefix="bad",
):
    """DataFrame: 'time' + <chan>_<band> мощности или NaN."""
    if bands is None:
        bands = {
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "smr": (12, 15),
        }

    sfreq = raw.info["sfreq"]
    times = raw.times

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    ch_names = np.array(raw.ch_names)[picks]

    bad_masks = create_channel_bad_masks(raw, picks, bad_prefix=bad_prefix)

    win_centers = []
    cur_start = 0.0
    while cur_start + win_len <= times[-1]:
        cur_stop = cur_start + win_len
        win_centers.append((cur_start + cur_stop) / 2.0)
        cur_start += step
    win_centers = np.array(win_centers)

    cols = ["time"]
    for ch in ch_names:
        for band_name in bands.keys():
            cols.append(f"{ch}_{band_name}")
    df = pd.DataFrame(index=np.arange(len(win_centers)), columns=cols, dtype=float)
    df["time"] = win_centers

    for wi, center in enumerate(win_centers):
        start = center - win_len / 2.0
        stop = center + win_len / 2.0

        start_sample = int(np.round(start * sfreq))
        stop_sample = int(np.round(stop * sfreq))
        start_sample = max(start_sample, 0)
        stop_sample = min(stop_sample, raw.n_times)
        if stop_sample <= start_sample:
            continue

        data_win, _ = raw[picks, start_sample:stop_sample]

        for ci, ch_name in enumerate(ch_names):
            window_bad = bad_masks[ci, start_sample:stop_sample]
            clean_samples = ~window_bad
            clean_ratio = clean_samples.sum() / len(clean_samples)
            if clean_ratio < min_clean_ratio:
                continue

            x = data_win[ci, clean_samples]
            if x.size < sfreq * 0.5:
                continue

            for band_name, (fmin, fmax) in bands.items():
                val = compute_band_power_array(x, sfreq, fmin, fmax)
                val = val / (fmax - fmin)
                df.at[wi, f"{ch_name}_{band_name}"] = val

    return df

def plot_annotation(raw):
    sfreq = raw.info["sfreq"]
    times = raw.times
    ch_names = raw.ch_names
    n_ch = len(ch_names)
    n_t = raw.n_times
    
    bad_mat = np.zeros((n_ch, n_t), dtype=bool)
    ann = raw.annotations
    has_ch = getattr(ann, "ch_names", None) is not None
    
    for i_ann, a in enumerate(ann):
        desc = a["description"]
        if not desc.lower().startswith("bad"):
            continue
    
        onset = float(a["onset"])
        duration = float(a["duration"])
        start = max(int(np.round(onset * sfreq)), 0)
        stop = min(start + int(np.round(duration * sfreq)), n_t)
        if stop <= start:
            continue
    
        if has_ch:
            ann_chs = ann.ch_names[i_ann]
            if ann_chs is None or len(ann_chs) == 0:
                affected_idx = np.arange(n_ch)
            else:
                affected_idx = [ch_names.index(ch) for ch in ann_chs if ch in ch_names]
        else:
            affected_idx = np.arange(n_ch)
    
        bad_mat[affected_idx, start:stop] = True
    
    fig, ax = plt.subplots(figsize=(15, 8))
    img = ax.imshow(
        bad_mat,
        aspect="auto",
        interpolation="nearest",
        extent=[times[0], times[-1], -0.5, n_ch - 0.5],
        cmap="Reds",
        origin="lower",
    )
    ax.set_yticks(np.arange(n_ch))
    ax.set_yticklabels(ch_names)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Каналы")
    ax.set_title("BAD-аннотации по всем каналам")
    plt.colorbar(img, ax=ax, label="BAD (1) / OK (0)")
    plt.tight_layout()
    
    
def plot_bands_power(df_bands, bands, dir_name=None, resp_name=None):
    """
    df_bands: DataFrame с колонками 'time' и <chan>_<band>.
    bands: dict, ключи = имена ритмов (theta/alpha/...).
    """
    time = df_bands["time"].values
    band_names = list(bands.keys())

    # извлекаем имена каналов из колонок "<chan>_<band>"
    pattern = r"^(.*)_(" + "|".join(band_names) + r")$"
    ch_names = []
    for col in df_bands.columns:
        m = re.match(pattern, col)
        if m:
            ch = m.group(1)
            if ch not in ch_names:
                ch_names.append(ch)

    # --- 1. ритмы по каналам (subplot на каждый канал) ---
    n_channels = len(ch_names)
    n_cols = 2
    n_rows = (n_channels + 1) // 2

    fig1, axes1 = plt.subplots(
        n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=True
    )
    axes1 = np.array(axes1).reshape(-1)

    for ch_idx, ch_name in enumerate(ch_names):
        ax = axes1[ch_idx]
        has_any = False
        for band_name in band_names:
            col = f"{ch_name}_{band_name}"
            if col not in df_bands.columns:
                continue
            y = df_bands[col].values
            if np.all(np.isnan(y)):
                continue
            ax.plot(time, np.log10(y), label=band_name.upper(), linewidth=1.5, alpha=0.9)
            has_any = True
        ax.set_title(ch_name)
        ax.grid(True, alpha=0.3)
        if has_any:
            ax.legend(fontsize=8)

    # удалить пустые оси
    for idx in range(n_channels, len(axes1)):
        fig1.delaxes(axes1[idx])

    axes1[-1].set_xlabel("Время, с")
    fig1.suptitle("Ритмы по каналам", fontsize=16)
    plt.tight_layout()
    plt.show()

    # --- 2. каждый ритм по всем каналам (4 subplot) ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    axes2 = axes2.flatten()

    for i, band_name in enumerate(band_names[:4]):  # максимум 4 ритма
        ax = axes2[i]
        has_any = False
        for ch_name in ch_names:
            col = f"{ch_name}_{band_name}"
            if col not in df_bands.columns:
                continue
            y = df_bands[col].values
            if np.all(np.isnan(y)):
                continue
            ax.plot(time, np.log10(y), label=ch_name, alpha=0.7)
            has_any = True
        ax.set_title(f"{band_name.upper()} по каналам")
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Мощность")
        ax.grid(True, alpha=0.3)
        if has_any:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    if dir_name is not None and resp_name is not None:
        fig1.savefig(
            f"{dir_name}/{resp_name}_channel_powers_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            f"{dir_name}/{resp_name}_rythms_powers_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()