from typing import Dict, Any

import bentoml
from pydantic import BaseModel, Field
import pandas as pd

CURRENT_YEAR = 2025


class BuildingInput(BaseModel):
    year_built: int = Field(..., ge=1800, le=CURRENT_YEAR)
    primary_property_type: str = Field(..., description="ex: Office, Hotel, etc.")
    neighborhood: str | None = None
    property_gfa_total: float = Field(..., gt=0)
    number_of_floors: int = Field(..., gt=0)
    number_of_buildings: int = Field(..., gt=0)


@bentoml.service(name="SeattleEnergyGBService")
class SeattleEnergyGBService:
    """
    Service HTTP BentoML qui sert le pipeline :
    (preprocessor + GradientBoostingRegressor) sauvegardé sous 'seattle_energy_gb'.
    """

    def __init__(self) -> None:
        #  stocker la référence en attribut pour pouvoir l’utiliser dans predict()
        self.model_ref = bentoml.sklearn.get("seattle_energy_gb:latest")
        self.model = bentoml.sklearn.load_model(self.model_ref)
        self.feature_names = self.model_ref.custom_objects["feature_names"]

    @bentoml.api
    def predict(self, building: BuildingInput) -> Dict[str, Any]:
        data = building.model_dump()

        #construit un dict par colonne attendue
        row = {col: data.get(col) for col in self.feature_names}
        df = pd.DataFrame([row], columns=self.feature_names)

        numeric_cols_int = ["year_built", "number_of_floors", "number_of_buildings"]
        numeric_cols_float = ["property_gfa_total"]
        cat_cols = ["primary_property_type", "neighborhood"]

        df[numeric_cols_int] = df[numeric_cols_int].astype("int64")
        df[numeric_cols_float] = df[numeric_cols_float].astype("float64")
        for c in cat_cols:
            df[c] = df[c].astype("string")

        y_pred = float(self.model.predict(df)[0])

        return {
            "prediction": y_pred,
            "unit": "kBtu/sf (SiteEUIWN)",
            "message": "Prédiction de consommation d'énergie annuelle (Gradient Boosting optimisé)",
            "inputs_interpreted": data,
            "model_tag": str(self.model_ref.tag),  
        }
