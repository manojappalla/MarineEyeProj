import ee
import geemap
import geopandas as gpd
import joblib
from typing import List, Tuple
import pandas as pd
from dask import delayed, compute


class Config:
    """
    A configuration class for storing all input parameters required for
    model training and image classification workflows.

    Attributes:
        aoi_shapefile_path (str): Path to the shapefile defining the area of interest (AOI).
        training_shapefile_path (str): Path to the shapefile containing labeled training data.
        start_date (str): Start date for image collection filtering in the format 'YYYY-MM-DD'.
        end_date (str): End date for image collection filtering in the format 'YYYY-MM-DD'.
        bands (List[str]): List of band names (or computed indices like NDVI) to be used as features.
        model_output_path (str): File path where the trained model will be saved.
    """

    def __init__(
        self,
        aoi_shapefile_path: str,
        training_shapefile_path: str,
        start_date: str,
        end_date: str,
        bands: List[str],
        model_output_path: str,
    ):
        self.aoi_shapefile_path = aoi_shapefile_path
        self.training_shapefile_path = training_shapefile_path
        self.start_date = start_date
        self.end_date = end_date
        self.bands = bands
        self.model_output_path = model_output_path


class DataProcessor:
    """
    A class responsible for loading AOI and processing Earth Engine imagery
    for analysis or classification. Uses configuration provided via a Config object.

    Attributes:
        config (Config): Configuration object containing user-specified parameters.
        aoi (ee.FeatureCollection): The area of interest, converted from a shapefile.
    """

    def __init__(self, config: Config):
        """
        Initializes the DataProcessor with configuration and loads the AOI shapefile.

        Args:
            config (Config): The configuration object with paths, dates, bands, etc.
        """
        self.config = config
        self.aoi = self._load_aoi()

    def _load_aoi(self) -> ee.FeatureCollection:
        """
        Loads the AOI shapefile and converts it into an Earth Engine FeatureCollection.

        Returns:
            ee.FeatureCollection: Area of interest usable in Earth Engine.
        """
        gdf = gpd.read_file(self.config.aoi_shapefile_path)
        return geemap.gdf_to_ee(gdf)

    @staticmethod
    def mask_clouds(image: ee.Image) -> ee.Image:
        """
        Masks clouds and cloud shadows from a Landsat image using QA_PIXEL band.

        Args:
            image (ee.Image): A single Landsat image with QA_PIXEL band.

        Returns:
            ee.Image: Cloud-masked image with scaling applied and metadata retained.
        """
        cloud_shadow_bit_mask = ee.Number(2).pow(3).int()
        clouds_bit_mask = ee.Number(2).pow(5).int()
        qa = image.select("QA_PIXEL")
        mask = (
            qa.bitwiseAnd(cloud_shadow_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(clouds_bit_mask).eq(0))
        )
        return (
            image.updateMask(mask)
            .divide(10000)
            .copyProperties(image, ["system:time_start"])
        )

    @staticmethod
    def add_indices(img: ee.Image) -> ee.Image:
        """
        Adds spectral indices to the input image including NDVI, NDMI, MNDWI,
        simple ratios, and GCVI.

        Args:
            img (ee.Image): Input image with surface reflectance bands.

        Returns:
            ee.Image: Image with additional computed index bands.
        """
        ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
        ndmi = img.normalizedDifference(["SR_B7", "SR_B3"]).rename("NDMI")
        mndwi = img.normalizedDifference(["SR_B3", "SR_B6"]).rename("MNDWI")
        sr = img.select("SR_B5").divide(img.select("SR_B4")).rename("SR")
        ratio54 = img.select("SR_B6").divide(img.select("SR_B5")).rename("R54")
        ratio35 = img.select("SR_B4").divide(img.select("SR_B6")).rename("R35")
        gcvi = img.expression(
            "(NIR / GREEN) - 1",
            {"NIR": img.select("SR_B5"), "GREEN": img.select("SR_B3")},
        ).rename("GCVI")
        return img.addBands([ndvi, ndmi, mndwi, sr, ratio54, ratio35, gcvi])

    @staticmethod
    def set_date_str(image):
        """
        Adds a human-readable date string property ("date_str") to the image.

        Args:
            image (ee.Image): Input image with system:time_start metadata.

        Returns:
            ee.Image: Image with "date_str" property added.
        """
        date_str = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        return image.set("date_str", date_str)

    def get_processed_image(self) -> ee.Image:
        """
        Retrieves a median composite of Landsat 8 images within the specified date range,
        with cloud masking and index bands added. Also adds SRTM elevation data.

        Returns:
            ee.Image: A single image with selected bands and indices, clipped to the AOI.
        """
        l8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(self.config.start_date, self.config.end_date)
            .map(self.mask_clouds)
            .map(self.add_indices)
        )

        srtm = ee.Image("USGS/SRTMGL1_003").clip(self.aoi).rename("SRTM")
        return l8.median().addBands(srtm).clip(self.aoi).select(self.config.bands)


class SampleManager:
    """
    Manages the extraction and splitting of training samples from a landsat image
    based on vector training data.

    Attributes:
        image (ee.Image): The processed image with selected bands and indices.
        config (Config): Configuration object containing paths and parameters.
        class_attribute (str): Name of the attribute in the training shapefile used as the class label.
    """

    def __init__(self, image: ee.Image, config: Config, class_attribute):
        """
        Initializes the SampleManager with the input image, config, and class attribute.

        Args:
            image (ee.Image): The image to sample from (should be clipped and processed).
            config (Config): Configuration object with training shapefile path.
            class_attribute (str): The attribute name in the training shapefile used for class labels.
        """
        self.image = image
        self.config = config
        self.class_attribute = class_attribute

    def sample_and_split(
        self, split: float = 0.7
    ) -> Tuple[ee.FeatureCollection, ee.FeatureCollection]:
        """
        Samples pixel values from the image using the training shapefile and splits
        the result into training and testing datasets based on a random split.

        Args:
            split (float): Proportion of samples to include in the training set.
                           Remaining samples are assigned to the testing set.
                           Must be between 0 and 1. Default is 0.7.

        Returns:
            Tuple[ee.FeatureCollection, ee.FeatureCollection]: A tuple containing:
                - training samples (ee.FeatureCollection)
                - testing samples (ee.FeatureCollection)
        """
        gdf = gpd.read_file(self.config.training_shapefile_path)
        training_fc = geemap.gdf_to_ee(gdf)
        samples = self.image.sampleRegions(
            collection=training_fc, properties=[self.class_attribute], scale=30
        ).randomColumn("random")

        train = samples.filter(ee.Filter.lt("random", split))
        test = samples.filter(ee.Filter.gte("random", split))

        return train, test


class ModelTrainer:
    """
    A class for training and saving a Random Forest classifier using Earth Engine data.

    Attributes:
        bands (List[str]): The list of spectral band names used as input features.
        output_path (str): The directory path where the trained model metadata will be saved.
        class_attribute (str): The name of the class attribute in the training data.
        classifier (ee.Classifier): The trained Earth Engine classifier (initialized after training).
    """

    def __init__(self, bands: List[str], output_path: str, class_attribute):
        """
        Initializes the ModelTrainer with input features, output path, and class attribute.

        Args:
            bands (List[str]): List of feature band names to use for training.
            output_path (str): Directory path to save the trained model metadata.
            class_attribute (str): The attribute name used as the target class in the training dataset.
        """
        self.bands = bands
        self.output_path = output_path
        self.classifier = None
        self.class_attribute = class_attribute

    def train(self, training_data: ee.FeatureCollection):
        """
        Trains a Random Forest classifier using the provided training data.

        Args:
            training_data (ee.FeatureCollection): Feature collection containing labeled training samples.

        Returns:
            ee.Classifier: A trained Earth Engine Random Forest classifier.
        """
        self.classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=100, variablesPerSplit=5
        ).train(
            features=training_data.select(self.bands + [self.class_attribute]),
            classProperty=self.class_attribute,
            inputProperties=self.bands,
        )
        return self.classifier

    def save_metadata(self, classifier):
        """
        Saves the trained classifier object metadata locally as a pickle file.

        Args:
            classifier (ee.Classifier): The trained classifier to be saved.
        """
        joblib.dump(classifier, f"{self.output_path}/mangrove_model.pkl")
        print(f"Model saved to {self.output_path}")


class ModelApplier:
    """
    Applies a trained classifier model to Landsat-8 imagery over a user-defined AOI
    and computes classified area and mean NDVI for each image in a given time range.

    Attributes:
        model (ee.Classifier): The trained classifier model loaded from disk.
        bands (List[str]): List of band names used for classification.
        aoi (ee.FeatureCollection): Area of interest converted to Earth Engine object.
        start_date (str): Start date of the image collection (format: 'YYYY-MM-DD').
        end_date (str): End date of the image collection (format: 'YYYY-MM-DD').
        cloud_cover (int): Maximum cloud cover percentage allowed in imagery.
    """

    def __init__(
        self,
        model_path: str,
        aoi_path: str,
        start_date: str,
        end_date: str,
        cloud_cover: int,
        bands: List,
    ):
        """
        Initializes the ModelApplier class.

        Args:
            model_path (str): Path to the trained model file (e.g., joblib file).
            aoi_path (str): Path to the AOI shapefile/GeoJSON.
            start_date (str): Start date for filtering the image collection.
            end_date (str): End date for filtering the image collection.
            cloud_cover (int): Cloud cover threshold (0–100) for filtering images.
            bands (List[str]): List of band names required by the classifier.
        """
        self.model = joblib.load(model_path)
        self.bands = bands
        self.aoi = geemap.gdf_to_ee(gpd.read_file(aoi_path))
        self.start_date = start_date
        self.end_date = end_date
        self.cloud_cover = cloud_cover

    def _process_image(self, image: ee.Image) -> dict:
        """
        Processes a single image: classifies the image, computes the area of class 1,
        and calculates mean NDVI for the classified region.

        Args:
            image (ee.Image): Earth Engine image object to process.

        Returns:
            dict: Dictionary containing:
                - 'date' (str): Image acquisition date.
                - 'area_km2' (float): Classified area (class 1) in square kilometers.
                - 'mean_ndvi' (float): Mean NDVI value for the classified area.
        """
        image_date = (
            ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        )
        srtm = ee.Image("USGS/SRTMGL1_003").rename("SRTM").clip(self.aoi)
        image = image.addBands(srtm).select(self.bands).clip(self.aoi)
        image = image.clip(self.aoi).reproject(crs="EPSG:4326", scale=30)

        # Classify using classifier (must re-train to inject into API call)
        classifier = self.model
        classified = image.select(self.bands).classify(classifier)

        pixel_count = classified.connectedPixelCount(100, False)
        count_mask = pixel_count.gt(1)
        class_mask = classified.eq(1)
        classed = classified.updateMask(count_mask).updateMask(class_mask)

        area = (
            classed.multiply(ee.Image.pixelArea())
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=self.aoi.geometry(),
                scale=100,
                maxPixels=1e13,
                tileScale=16,
            )
            .get("classification")
        )

        area_km2 = ee.Number(area).divide(1e6).multiply(100).round().divide(100)

        ndvi = image.select("NDVI")
        masked_ndvi = ndvi.updateMask(classed.eq(1))

        mean_ndvi = masked_ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=self.aoi.geometry(),
            scale=30,
            maxPixels=1e13,
            tileScale=16,
        ).get("NDVI")

        return {
            "date": image_date,
            "area_km2": area_km2.getInfo(),
            "mean_ndvi": mean_ndvi.getInfo(),
        }

    def run_parallel(self) -> pd.DataFrame:
        """
        Runs the classification process on all valid images in the filtered collection
        and computes metrics in parallel using Dask.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - 'date' (str): Date of each image.
                - 'area_km2' (float): Area of class 1 in square kilometers.
                - 'mean_ndvi' (float): Mean NDVI for the classified area.
        """
        l8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(self.start_date, self.end_date)
            .filterMetadata("CLOUD_COVER", "less_than", self.cloud_cover)
            .filterBounds(self.aoi.geometry())
            .map(DataProcessor.mask_clouds)
            .map(DataProcessor.add_indices)
            .map(DataProcessor.set_date_str)
        )
        l8 = l8.distinct("date_str")
        image_list = l8.toList(l8.size())
        size = l8.size().getInfo()

        tasks = [
            delayed(self._process_image)(ee.Image(image_list.get(i)))
            for i in range(size)
        ]
        results = compute(*tasks)

        return pd.DataFrame(results)
