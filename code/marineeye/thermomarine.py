#!/usr/bin/env python3
"""
Core library for ThermoMarine SST forecasting.
Holds classes only—no side effects.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import argparse, logging, os, sys
import numpy as np, pandas as pd, xarray as xr, tensorflow as tf
import matplotlib.pyplot as plt, matplotlib.animation as animation
import cartopy.crs as ccrs, cartopy.feature as cfeature
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dense
import copernicusmarine

# ─── Logger (must come before Config.parse uses it) ────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
logger = logging.getLogger("ThermoMarine")


# ------------------------------------------------------------------- #
# 1)  Config  (defaults + CLI parser)                                 #
# ------------------------------------------------------------------- #
class Config:
    DEFAULTS = {
        "file": "sst.nc",
        "var": "analysed_sst",
        "bbox": (-180, 180, -90, 90),
        "n_lags": 7,
        "horizon": 30,
        "epochs": 50,
        "fps": 5,
    }

    @staticmethod
    def parse() -> argparse.Namespace:
        p = argparse.ArgumentParser("ThermoMarine forecaster")
        p.add_argument("--file", default=Config.DEFAULTS["file"])
        p.add_argument("--start"), p.add_argument("--end")
        p.add_argument("--var", default=Config.DEFAULTS["var"])
        p.add_argument("--bbox", nargs=4, type=float)
        p.add_argument("--n_lags", type=int, default=Config.DEFAULTS["n_lags"])
        p.add_argument("--horizon", type=int, default=Config.DEFAULTS["horizon"])
        p.add_argument("--epochs", type=int, default=Config.DEFAULTS["epochs"])
        p.add_argument("--animate", action="store_true")
        p.add_argument("--fps", type=int, default=Config.DEFAULTS["fps"])
        args = p.parse_args()

        # interactive mode
        if len(sys.argv) == 1:
            logger.info("Interactive prompts (blank = default)")
            args.start = input("Start (YYYY-MM-DD): ") or None
            args.end = input("End   (YYYY-MM-DD): ") or None
            args.var = input(f"Variable [{args.var}]: ") or args.var
            bbox = input("BBox lon_min lon_max lat_min lat_max: ")
            if bbox:
                args.bbox = tuple(map(float, bbox.split()))
            args.n_lags = int(input(f"Lags [{args.n_lags}]: ") or args.n_lags)
            args.horizon = int(input(f"Horizon [{args.horizon}]: ") or args.horizon)
            args.epochs = int(input(f"Epochs [{args.epochs}]: ") or args.epochs)
            args.animate = input("Animate? (y/n) [n]: ").lower() == "y"

        if args.bbox is None:
            args.bbox = Config.DEFAULTS["bbox"]
        return args


# ------------------------------------------------------------------- #
# 2)  Data helper                                                     #
# ------------------------------------------------------------------- #
class DataHelper:
    @staticmethod
    def set_seed(seed=42):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)

    @staticmethod
    def download_dataset(start_date: str | None, end_date: str | None) -> xr.Dataset:
        return copernicusmarine.open_dataset(
            dataset_id="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
            variables=["analysed_sst", "analysis_error"],
            minimum_longitude=-179.97500610351562,
            maximum_longitude=179.97500610351562,
            minimum_latitude=-89.9749984741211,
            maximum_latitude=89.9749984741211,
            start_datetime=start_date,
            end_datetime=end_date,
        )

    @staticmethod
    def load_ts(
        ds: xr.Dataset,
        var: str,
        bbox: Tuple[float, float, float, float],
        start: str,
        end: str,
    ) -> pd.Series:
        ds = ds.sel(time=slice(start, end))
        lon0, lon1, lat0, lat1 = bbox
        da = ds[var].sel(longitude=slice(lon0, lon1), latitude=slice(lat0, lat1))
        if "depth" in da.dims:
            da = da.mean("depth")
        ts = da.mean(("latitude", "longitude")).to_series().dropna()
        return ts

    @staticmethod
    def windowize(ts: pd.Series, n_lags: int):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(ts.values.reshape(-1, 1)).squeeze()
        X = np.stack([scaled[i - n_lags : i] for i in range(n_lags, len(scaled))])
        y = scaled[n_lags:]
        return X.reshape(-1, n_lags, 1), y, scaler


# ------------------------------------------------------------------- #
# 3)  Model helper                                                    #
# ------------------------------------------------------------------- #
class ModelHelper:
    CHECKPOINT = "best.keras"

    @staticmethod
    def build(n_lags: int) -> Sequential:
        m = Sequential(
            [
                Conv1D(
                    32, 3, padding="causal", activation="relu", input_shape=(n_lags, 1)
                ),
                LSTM(64),
                Dense(1),
            ]
        )
        m.compile("adam", loss=tf.keras.losses.Huber(delta=1.0))
        return m

    @staticmethod
    def train(m: Sequential, X, y, epochs: int, topup: int = 3):
        if Path(ModelHelper.CHECKPOINT).exists():
            m.load_weights(ModelHelper.CHECKPOINT, skip_mismatch=True)
            epochs = min(topup, epochs)
        split = int(len(X) * 0.8)
        m.fit(
            X[:split],
            y[:split],
            validation_data=(X[split:], y[split:]),
            epochs=epochs,
            batch_size=16,
            callbacks=[
                EarlyStopping(patience=2, restore_best_weights=True),
                ModelCheckpoint(ModelHelper.CHECKPOINT, save_best_only=True),
            ],
            verbose=2,
        )

    @staticmethod
    def forecast(
        m: Sequential, ts: pd.Series, scaler: MinMaxScaler, n_lags: int, horizon: int
    ) -> pd.Series:
        seq = scaler.transform(ts.values.reshape(-1, 1)).flatten().tolist()
        preds = []
        for _ in range(horizon):
            yhat = float(
                m.predict(np.asarray(seq[-n_lags:]).reshape(1, n_lags, 1), verbose=0)
            )
            preds.append(yhat)
            seq.append(yhat)
        preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).squeeze()
        idx = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=horizon)
        return pd.Series(preds, idx, name=f"{ts.name}_fc")

    @staticmethod
    def save_final(m: Sequential, path="thermomarine_final.keras"):
        m.save(path)
        logger.info("Full model saved to %s", path)


# ------------------------------------------------------------------- #
# 4)  Visualizer                                                      #
# ------------------------------------------------------------------- #
class Visualizer:
    @staticmethod
    def plot_ts(hist: pd.Series, fc: pd.Series):
        plt.figure(figsize=(12, 6))
        plt.plot(hist, label="Historic")
        plt.axvline(hist.index[-1], ls=":", color="k", label="Forecast start")
        plt.plot(fc, "--o", label="Forecast")
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def animate_sst(
        ds: xr.Dataset,
        var,
        bbox,
        start,
        end,
        fps=5,
        cmap="inferno",
        outfile="sst_anim.mp4",
    ):
        da = ds[var].sel(
            longitude=slice(bbox[0], bbox[1]),
            latitude=slice(bbox[2], bbox[3]),
            time=slice(start, end),
        )
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(
            subplot_kw={"projection": proj}, figsize=(10, 6), facecolor="black"
        )
        ax.add_feature(cfeature.LAND, facecolor="black")
        ax.coastlines(linewidth=0.3, color="lightcyan")
        mesh = ax.pcolormesh(
            da.longitude,
            da.latitude,
            da.isel(time=0),
            cmap=cmap,
            shading="auto",
            transform=proj,
        )
        fig.colorbar(mesh, ax=ax, label=f"{var} (°C)")

        def update(i):
            mesh.set_array(da.isel(time=i).values.ravel())
            ax.set_title(
                str(pd.Timestamp(da.time.values[i]).date()), color="white", fontsize=10
            )
            return (mesh,)

        ani = animation.FuncAnimation(
            fig, update, frames=len(da.time), blit=True, interval=1000 / fps
        )
        ani.save(outfile, writer="ffmpeg", fps=fps, dpi=200)
        logger.info("Animation saved → %s", outfile)