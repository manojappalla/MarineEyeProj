# algal_gui.py
# PyQt5 UI file generated from algal_dashboard.ui
# ------------------------------------------------
import os
import glob
import numpy as np
import xarray as xr
from PyQt5.QtCore import QThread, pyqtSignal
import copernicusmarine
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
from sklearn.preprocessing import MinMaxScaler

# Worker Thread
class Worker(QThread):
    output_signal = pyqtSignal(str)
    done_signal = pyqtSignal()

    def __init__(self, days):
        super().__init__()
        self.days = days

    def get_latest_file(self, pattern):
        """Returns the most recently modified file matching the pattern."""
        files = glob.glob(pattern)
        return max(files, key=os.path.getmtime) if files else None

    def normalize(self, data_array):
        flat = data_array.flatten()
        norm = MinMaxScaler().fit_transform(flat.reshape(-1, 1)).reshape(data_array.shape)
        return norm

    def compute_ahp_weights(self):
        ahp_matrix = np.array([
            [1,   3,   5],
            [1/3, 1,   2],
            [1/5, 1/2, 1]
        ])
        col_sum = ahp_matrix.sum(axis=0)
        norm_matrix = ahp_matrix / col_sum
        return norm_matrix.mean(axis=1)

    def run(self):
        try:
            # Get parameters from environment
            min_lon = float(os.getenv('MIN_LON'))
            max_lon = float(os.getenv('MAX_LON'))
            min_lat = float(os.getenv('MIN_LAT'))
            max_lat = float(os.getenv('MAX_LAT'))
            start_date = os.getenv('START_DATE')
            end_date = os.getenv('END_DATE')
            variables = os.getenv('VARIABLES').split(',')

            self.output_signal.emit("[INFO] Logging into Copernicus Marine API...")

            # Download files if needed
            if "chl" in variables and not self.get_latest_file("chl*.nc"):
                self.output_signal.emit("[INFO] Downloading CHL...")
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_bgc_myint_0.25deg_P1D-m",
                    variables=["chl"],
                    minimum_longitude=min_lon, maximum_longitude=max_lon,
                    minimum_latitude=min_lat, maximum_latitude=max_lat,
                    start_datetime=f"{start_date}T00:00:00",
                    end_datetime=f"{end_date}T00:00:00",
                    minimum_depth=0.5, maximum_depth=1.5,
                    output_filename="chl.nc"
                )

            if "sst" in variables and not self.get_latest_file("sst*.nc"):
                self.output_signal.emit("[INFO] Downloading SST...")
                copernicusmarine.subset(
                    dataset_id="C3S-GLO-SST-L4-REP-OBS-SST",
                    variables=["analysed_sst"],
                    minimum_longitude=min_lon, maximum_longitude=max_lon,
                    minimum_latitude=min_lat, maximum_latitude=max_lat,
                    start_datetime=f"{start_date}T00:00:00",
                    end_datetime=f"{end_date}T00:00:00",
                    output_filename="sst.nc"
                )

            if "sss" in variables and not self.get_latest_file("sss*.nc"):
                self.output_signal.emit("[INFO] Downloading SSS...")
                copernicusmarine.subset(
                    dataset_id="cmems_obs-mob_glo_phy-sss_nrt_multi_P1D",
                    variables=["sos"],
                    minimum_longitude=min_lon, maximum_longitude=max_lon,
                    minimum_latitude=min_lat, maximum_latitude=max_lat,
                    start_datetime=f"{start_date}T00:00:00",
                    end_datetime=f"{end_date}T00:00:00",
                    minimum_depth=0, maximum_depth=0,
                    output_filename="sss.nc"
                )

            # Process latest files
            chl = xr.open_dataset(self.get_latest_file("chl*.nc"))["chl"].isel(depth=0) if "chl" in variables else None
            sst = xr.open_dataset(self.get_latest_file("sst*.nc"))["analysed_sst"] if "sst" in variables else None
            sss = xr.open_dataset(self.get_latest_file("sss*.nc"))["sos"].isel(depth=0) if "sss" in variables else None

            # Interpolate to common grid
            if chl is not None and sst is not None:
                sst = sst.interp_like(chl)
            if chl is not None and sss is not None:
                sss = sss.interp_like(chl)

            weights = self.compute_ahp_weights()
            self.output_signal.emit(f"[INFO] AHP Weights -> CHL: {weights[0]:.2f}, SST: {weights[1]:.2f}, SSS: {weights[2]:.2f}")

            # Determine reference dataset
            ref_ds = chl if chl is not None else sst if sst is not None else sss
            if ref_ds is None:
                raise ValueError("No valid datasets available for processing")

            num_timesteps = min(len(ref_ds.time), self.days)
            risk_maps = []
            time_labels = []
            
            for t in range(num_timesteps):
                chl_norm = self.normalize(chl.isel(time=t).values) if chl is not None else 0
                sst_norm = self.normalize(sst.isel(time=t).values) if sst is not None else 0
                sss_norm = self.normalize(sss.isel(time=t).values) if sss is not None else 0

                composite = weights[0]*chl_norm + weights[1]*sst_norm + weights[2]*sss_norm
                risk = np.zeros_like(composite)
                risk[composite > 0.66] = 2
                risk[(composite > 0.33) & (composite <= 0.66)] = 1
                
                risk_maps.append(risk)
                time_labels.append(str(ref_ds.time.values[t])[:10])

            # Generate animation
            self.output_signal.emit("[INFO] Generating risk animation...")
            fig = plt.figure(figsize=(10, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            lon = ref_ds.longitude.values
            lat = ref_ds.latitude.values
            lon_grid, lat_grid = np.meshgrid(lon, lat)

            im = ax.pcolormesh(lon_grid, lat_grid, risk_maps[0], 
                              cmap='RdYlGn_r', vmin=0, vmax=2, shading='auto')
            ax.coastlines()
            ax.set_extent([min_lon, max_lon, min_lat, max_lat])
            
            cbar = plt.colorbar(im, ticks=[0, 1, 2], ax=ax)
            cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])
            ax.set_title(f"Algal Bloom Risk - {time_labels[0]}")

            def update(i):
                im.set_array(risk_maps[i].flatten())
                ax.set_title(f"Algal Bloom Risk - {time_labels[i]}")
                return [im]

            ani = FuncAnimation(fig, update, frames=len(risk_maps), interval=1000)
            ani.save("risk_animation.gif", writer="pillow")
            self.output_signal.emit("[INFO] Saved risk animation as risk_animation.gif")

            # Clean up
            for pattern in ["chl*.nc", "sst*.nc", "sss*.nc"]:
                for f in glob.glob(pattern):
                    try:
                        os.remove(f)
                    except PermissionError:
                        self.output_signal.emit(f"[WARNING] Could not delete {f}")

            self.done_signal.emit()

        except Exception as e:
            self.output_signal.emit(f"[ERROR] {str(e)}")

