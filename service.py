from typing import Dict, Any

import bentoml
from pydantic import BaseModel, Field
import pandas as pd

CURRENT_YEAR = 2025


# --------- Schéma d'entrée (Pydantic) ---------
class BuildingInput(BaseModel):
    year_built: int = Field(..., ge=1800, le=CURRENT_YEAR)
    primary_property_type: str = Field(..., description="ex: Office, Hotel, etc.")
    neighborhood: str | None = None
    property_gfa_total: float = Field(..., gt=0)
    number_of_floors: int = Field(..., gt=0)
    number_of_buildings: int = Field(..., gt=0)


# --------- Service BentoML basé sur le XGBoost optimisé ---------
@bentoml.service(name="SeattleEnergyXGBService")
class SeattleEnergyXGBService:
    """
    Service HTTP BentoML qui sert le pipeline :
    (preprocessor + XGBRegressor) sauvegardé sous 'seattle_energy_xgb'.
    """

    def __init__(self) -> None:
        # Récupération du modèle depuis le model store
        model_ref = bentoml.sklearn.get("seattle_energy_xgb:latest")

        # Chargement du pipeline sklearn (preprocessor + XGBRegressor)
        self.model = bentoml.sklearn.load_model(model_ref)

        # Colonnes d'entrée (dans l'ordre utilisé à l'entraînement)
        self.feature_names = model_ref.custom_objects["feature_names"]

    @bentoml.api
    def predict(self, building: BuildingInput) -> Dict[str, Any]:
        """
        Endpoint /predict
        - Reçoit un JSON conforme à BuildingInput
        - Reconstitue un DataFrame aligné sur les features du pipeline
        - Retourne la prédiction de SiteEUIWN
        """

        # 1. Pydantic -> dict
        data = building.model_dump()

        # 2. Réordonne les features comme à l'entraînement
        row = [data.get(col) for col in self.feature_names]
        df = pd.DataFrame([row], columns=self.feature_names)

        # 3. Forçage des types comme dans le notebook
        numeric_cols_int = ["year_built", "number_of_floors", "number_of_buildings"]
        numeric_cols_float = ["property_gfa_total"]
        cat_cols = ["primary_property_type", "neighborhood"]

        df[numeric_cols_int] = df[numeric_cols_int].astype("int64")
        df[numeric_cols_float] = df[numeric_cols_float].astype("float64")
        for c in cat_cols:
            df[c] = df[c].astype("string")

        # 4. Prédiction avec le pipeline XGBoost optimisé
        y_pred = float(self.model.predict(df)[0])

        return {
            "prediction": y_pred,
            "unit": "kBtu/sf (SiteEUIWN)",
            "message": "Prédiction de consommation d'énergie annuelle (XGBoost optimisé)",
            "inputs_interpreted": data,
        }
