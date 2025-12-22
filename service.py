from typing import Dict, Any

import numpy as np
import pandas as pd
import bentoml
from pydantic import BaseModel, Field, ConfigDict

# --- Modèle verrouillé (évite les surprises avec :latest)
MODEL_TAG = "seattle_energy_gb:b5ptlkg7dgbmwaam"

# Constantes métier
CURRENT_YEAR = 2025
DOWNTOWN_LAT = 47.6062
DOWNTOWN_LON = -122.3321


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2) + (np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    return float(r * c)


class BuildingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year_built: int = Field(..., ge=1800, le=CURRENT_YEAR)

    building_type: str = Field(..., min_length=1)
    primary_property_type: str = Field(..., min_length=1)
    neighborhood: str = Field(..., min_length=1)

    zip_code: int = Field(..., ge=0)
    ListOfAllPropertyUseTypes: str = Field(..., min_length=1)
    largest_property_use_type: str = Field(..., min_length=1)

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

    property_gfa_total: float = Field(..., gt=0)
    property_gfa_parking: float = Field(..., ge=0)

    number_of_floors: int = Field(..., gt=0)
    number_of_buildings: int = Field(..., gt=0)

    usage_count: int = Field(..., ge=1)


@bentoml.service(name="SeattleEnergyGBService")
class SeattleEnergyGBService:
    def __init__(self) -> None:
        # Chargement pipeline complet (preprocess + regressor)
        self.model = bentoml.sklearn.load_model(MODEL_TAG)
        self.model_ref = bentoml.sklearn.get(MODEL_TAG)

        # Features attendues (ordre canonique)
        self.feature_names = self.model_ref.custom_objects.get("feature_names")
        if not self.feature_names:
            raise RuntimeError(
                "custom_objects['feature_names'] manquant. "
                "Sauvegarder le modèle avec custom_objects={'feature_names': X_api.columns.tolist()}."
            )

        self.int_cols = {"zip_code", "number_of_floors", "number_of_buildings", "usage_count"}
        self.float_cols = {
            "latitude",
            "longitude",
            "property_gfa_total",
            "property_gfa_parking",
            "building_age",
            "parking_ratio",
            "floor_density",
            "distance_to_downtown_km",
        }
        self.cat_cols = {
            "building_type",
            "primary_property_type",
            "neighborhood",
            "ListOfAllPropertyUseTypes",
            "largest_property_use_type",
        }

    @bentoml.api
    def predict(self, building: BuildingInput) -> Dict[str, Any]:
        data = building.model_dump()

        # Feature engineering
        building_age = CURRENT_YEAR - data["year_built"]
        parking_ratio = data["property_gfa_parking"] / data["property_gfa_total"]
        floor_density = data["number_of_floors"] / data["property_gfa_total"]
        distance_to_downtown_km = haversine_km(
            data["latitude"], data["longitude"], DOWNTOWN_LAT, DOWNTOWN_LON
        )

        engineered_features = {
            "building_age": float(building_age),
            "parking_ratio": float(parking_ratio),
            "floor_density": float(floor_density),
            "distance_to_downtown_km": float(distance_to_downtown_km),
        }

        model_data = {**data, **engineered_features}
        model_data.pop("year_built", None)

        # DataFrame strictement aligné
        row = {col: model_data.get(col, None) for col in self.feature_names}
        df = pd.DataFrame([row], columns=self.feature_names).replace({None: np.nan})

        # Typage robuste
        for col in df.columns:
            if col in self.int_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif col in self.float_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif col in self.cat_cols:
                df[col] = df[col].astype("string")

        y_pred = float(self.model.predict(df)[0])

        return {
            "prediction": y_pred,
            "unit": "kBtu/sf (SiteEUIWN)",
            "message": "Prédiction de consommation énergétique annuelle",
            "model_tag": str(self.model_ref.tag),
            "inputs_interpreted": {**data, **engineered_features},
        }
