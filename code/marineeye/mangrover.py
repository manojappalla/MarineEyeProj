import ee
import geemap
import geopandas as gpd
import joblib
from typing import List, Tuple
import pandas as pd
from dask import delayed, compute


class Config:
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
    def __init__(self, config: Config):
        self.config = config
        self.aoi = self._load_aoi()

    def _load_aoi(self) -> ee.FeatureCollection:
        gdf = gpd.read_file(self.config.aoi_shapefile_path)
        return geemap.gdf_to_ee(gdf)

    @staticmethod
    def mask_clouds(image: ee.Image) -> ee.Image:
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
        date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        return image.set('date_str', date_str)

    def get_processed_image(self) -> ee.Image:
        l8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(self.config.start_date, self.config.end_date)
            .map(self.mask_clouds)
            .map(self.add_indices)
        )

        srtm = ee.Image("USGS/SRTMGL1_003").clip(self.aoi).rename("SRTM")
        return l8.median().addBands(srtm).clip(self.aoi).select(self.config.bands)


class SampleManager:
    def __init__(self, image: ee.Image, config: Config, class_attribute):
        self.image = image
        self.config = config
        self.class_attribute = class_attribute

    def sample_and_split(
        self, split: float = 0.7
    ) -> Tuple[ee.FeatureCollection, ee.FeatureCollection]:
        gdf = gpd.read_file(self.config.training_shapefile_path)
        training_fc = geemap.gdf_to_ee(gdf)
        samples = self.image.sampleRegions(
            collection=training_fc, properties=[self.class_attribute], scale=30
        ).randomColumn("random")

        train = samples.filter(ee.Filter.lt("random", split))
        test = samples.filter(ee.Filter.gte("random", split))

        return train, test


class ModelTrainer:
    def __init__(self, bands: List[str], output_path: str, class_attribute):
        self.bands = bands
        self.output_path = output_path
        self.classifier = None
        self.class_attribute = class_attribute

    def train(self, training_data: ee.FeatureCollection):
        self.classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=100, variablesPerSplit=5
        ).train(
            features=training_data.select(self.bands + [self.class_attribute]),
            classProperty=self.class_attribute,
            inputProperties=self.bands,
        )
        return self.classifier

    def save_metadata(self, classifier):
        joblib.dump(classifier, f"{self.output_path}/mangrove_model.pkl")
        print(f"Model saved to {self.output_path}")


class ModelApplier:
    def __init__(
        self,
        model_path: str,
        aoi_path: str,
        start_date: str,
        end_date: str,
        cloud_cover: int,
        bands: List,
    ):
        self.model = joblib.load(model_path)
        self.bands = bands
        self.aoi = geemap.gdf_to_ee(gpd.read_file(aoi_path))
        self.start_date = start_date
        self.end_date = end_date
        self.cloud_cover = cloud_cover

    def _process_image(self, image: ee.Image) -> dict:
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
        l8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(self.start_date, self.end_date)
            .filterMetadata("CLOUD_COVER", "less_than", self.cloud_cover)
            .filterBounds(self.aoi.geometry())
            .map(DataProcessor.mask_clouds)
            .map(DataProcessor.add_indices)
            .map(DataProcessor.set_date_str)
        )
        l8 = l8.distinct('date_str')
        image_list = l8.toList(l8.size())
        size = l8.size().getInfo()

        tasks = [
            delayed(self._process_image)(ee.Image(image_list.get(i)))
            for i in range(size)
        ]
        results = compute(*tasks)

        return pd.DataFrame(results)
