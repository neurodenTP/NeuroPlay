import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

warnings.filterwarnings(
    "ignore",
    message="nperseg = .* is greater than input length",
    category=UserWarning,
)

%matplotlib inline

from neuroplay_src import (
    annotate_high_amplitude_per_channel,
    merge_overlapping_annotations,
    sliding_band_powers_to_df_per_channel,
    plot_annotation,
    plot_bands_power
)

#%% --- параметры записи ---
file_name = "2025.12.16-14.23.52.121.edf"
dir_name = "sample_data"
resp_name = "noname"

edf_path = dir_name + "/" + file_name

#%% --- 1. загрузка EDF ---
raw = mne.io.read_raw_edf(edf_path, preload=True)
raw.set_montage("standard_1020", match_case=False)

#%% --- 2. первичная визуализация ---
# Можно убрать лишние каналы
%matplotlib qt5
raw.plot(start=0.0, duration=10, block=True, scalings=dict(eeg=100e-6))
%matplotlib inline

#%%
raw.plot(start=0.0, duration=10, block=True, scalings=dict(eeg=100e-6))
raw.plot_psd(fmin=0.5, fmax=62.5)
plt.show()

#%% --- 3. фильтры ---
raw.filter(1.0, 40.0, picks="eeg")
raw.notch_filter(50, picks="eeg")

raw.plot(start=0.0, duration=10, block=True, scalings=dict(eeg=100e-6))
raw.plot_psd(fmin=0.5, fmax=62.5)
plt.show()

#%% --- 4. аннотации по амплитуде ---
# Помечаем участки с высокой амлитудой, в расчете ритмов их не учитываем
annot = annotate_high_amplitude_per_channel(
    raw,
    threshold=120e-6,
    pre=0.2,
    post=0.2,
    desc="BAD_amp",
)
annot = merge_overlapping_annotations(annot)
raw.set_annotations(annot, emit_warning=False)

plot_annotation(raw)
plt.show()

#%% --- 6. ICA --- 
good_ch_names = mne.pick_channels(raw.ch_names, include=raw.ch_names,
                                  exclude=raw.info['bads'])

ica = mne.preprocessing.ICA(n_components=len(good_ch_names), random_state=97, 
                            max_iter="auto")
ica.fit(raw.copy(), picks="eeg")
ica.plot_components()
plt.show()

eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="Fp1")
print("Подозрительные blink-компоненты:", eog_indices, "scores:", eog_scores)

#%%
ica.exclude = [2]  # выбираем те комоненты которые необходимо убрать
raw_ica = ica.apply(raw.copy())

raw_ica.plot(start=0.0, duration=10, block=True, scalings=dict(eeg=100e-6))
raw_ica.plot_psd(fmin=0.5, fmax=62.5)
plt.show()

#%% --- 7. расчёт мощностей в ритмах (скользящее окно) ---
bands = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "smr": (12, 15),
}

df_bands = sliding_band_powers_to_df_per_channel(
    raw_ica,
    win_len=20.0,
    step=10.0,
    bands=bands,
    min_clean_ratio=0.8,
    bad_prefix="bad",
)

plot_bands_power(df_bands, bands, dir_name=dir_name, resp_name=resp_name)
#%% --- 8. экспорт ---
out_path = dir_name + "/" + resp_name + "_band_powers.xlsx"
df_bands.to_excel(out_path, sheet_name="Powers", index=False)
print("Экспортировано в", out_path)
print(f"Размер: {df_bands.shape[0]} окон × {df_bands.shape[1]} колонок")

#%% --- 9. финальная визуализация ---
%matplotlib qt5
raw_clean.plot(start=0.0, duration=10, block=True, scalings=dict(eeg=100e-6))
%matplotlib inline