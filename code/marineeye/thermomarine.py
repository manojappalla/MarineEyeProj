import ee
import geemap
import geopandas as gpd
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio


class SSTExtractor:
    """
    Extracts Sea Surface Temperature (SST) time series from Google Earth Engine
    for a region defined by a shapefile.

    Attributes:
        shapefile_path (str): Path to the shapefile defining the region of interest.
        start_date (str): Start date for the time series (format: 'YYYY-MM-DD').
        end_date (str): End date for the time series (format: 'YYYY-MM-DD').
        df (pd.DataFrame): DataFrame containing the extracted SST time series.
    """

    def __init__(self, shapefile_path: str, start_date: str, end_date: str):
        """
        Initializes the SSTExtractor and Earth Engine API.

        Args:
            shapefile_path (str): Path to the shapefile.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        """
        self.shapefile_path = shapefile_path
        self.start_date = start_date
        self.end_date = end_date
        self.df = None

    def load_geometry(self):
        """
        Loads the shapefile and converts it to an Earth Engine Geometry.

        Returns:
            ee.Geometry: Geometry object representing the shapefile region.
        """
        gdf = gpd.read_file(self.shapefile_path)
        return geemap.geopandas_to_ee(gdf)

    def extract_time_series(self):
        """
        Extracts the mean SST values from Earth Engine for the defined region and time range.
        Stores the result in `self.df` as a Pandas DataFrame.
        """
        region = self.load_geometry()

        sst = (
            ee.ImageCollection("NOAA/CDR/OISST/V2_1")
            .filterDate(self.start_date, self.end_date)
            .filterBounds(region)
            .select("sst")
        )

        def extract(image):
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=25000, maxPixels=1e13
            )
            return ee.Feature(
                None,
                {"date": image.date().format("YYYY-MM-dd"), "sst": stats.get("sst")},
            )

        features = sst.map(extract)
        sst_fc = ee.FeatureCollection(features)
        self.df = geemap.ee_to_df(sst_fc)
        self.df.dropna(inplace=True)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df.sort_values("date", inplace=True)
        self.df["sst"] = self.df["sst"] * 0.01  # Convert to °C

    def save_csv(self, output_path: str):
        """
        Saves the SST time series DataFrame to a CSV file.

        Args:
            output_path (str): File path where CSV will be saved.
        """
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            print(f"Saved SST time series to: {output_path}")


class SSTForecaster:
    """
    Forecasts SST using the Prophet model and visualizes the results.

    Attributes:
        df (pd.DataFrame): Time series DataFrame loaded from CSV.
        prophet_model (Prophet): Trained Prophet model.
        forecast (pd.DataFrame): Forecasted future values.
    """

    def __init__(self, csv_path: str):
        """
        Loads SST time series from a CSV file.

        Args:
            csv_path (str): Path to the CSV containing 'date' and 'sst' columns.
        """
        self.df = pd.read_csv(csv_path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df.dropna(inplace=True)
        self.df.sort_values("date", inplace=True)
        self.df.set_index("date", inplace=True)
        self.prophet_model = None
        self.forecast = None

    def train_prophet(self, seasonality_mode: str = "multiplicative"):
        """
        Trains a Prophet model on the loaded SST time series.

        Args:
            seasonality_mode (str): Type of seasonality ('additive' or 'multiplicative').
        """
        df_prophet = self.df.reset_index().rename(columns={"date": "ds", "sst": "y"})
        self.prophet_model = Prophet(seasonality_mode=seasonality_mode)
        self.prophet_model.fit(df_prophet)

    def forecast_future(self, days: int = 30):
        """
        Forecasts SST for a specified number of days into the future.

        Args:
            days (int): Number of days to forecast.
        """
        if self.prophet_model is None:
            raise RuntimeError("Model not trained. Call train_prophet() first.")
        future = self.prophet_model.make_future_dataframe(periods=days)
        self.forecast = self.prophet_model.predict(future)

    def plot_forecast(self, forecast_days: int = 30) -> str:
        """
        Plots full observed SST and next forecasted SST using Plotly as two clean lines.
        Returns the plot as an HTML string (for use in QWebEngineView).

        Args:
            forecast_days (int): Number of forecast days to display.

        Returns:
            str: HTML string of the Plotly figure.
        """
        if self.forecast is None or self.df is None:
            raise RuntimeError("Forecast not computed or data not available.")

        # Full observed data
        observed = self.df.reset_index()
        last_date = observed["date"].max()

        # Forecasted part only
        forecast_future = self.forecast[self.forecast["ds"] > last_date].iloc[
            :forecast_days
        ]

        # Concatenate yhat predictions that overlap with observed range
        forecast_history = self.forecast[self.forecast["ds"] <= last_date]

        # Observed SST line
        trace_obs = go.Scatter(
            x=observed["date"],
            y=observed["sst"],
            mode="lines",
            name="Observed SST",
            line=dict(color="black"),
        )

        # Forecast SST line (includes overlapping yhat for observed + future)
        trace_forecast = go.Scatter(
            x=pd.concat([forecast_history["ds"], forecast_future["ds"]]),
            y=pd.concat([forecast_history["yhat"], forecast_future["yhat"]]),
            mode="lines",
            name="Forecast SST",
            line=dict(color="blue", dash="dash"),
        )

        layout = go.Layout(
            title=f"SST Forecast (Next {forecast_days} Days)",
            xaxis=dict(title="Date"),
            yaxis=dict(title="SST (°C)"),
            template="plotly_white",
            legend=dict(x=0.01, y=0.99),
        )

        fig = go.Figure(data=[trace_obs, trace_forecast], layout=layout)

        return fig.to_html(include_plotlyjs="cdn")
