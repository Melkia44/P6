from typing import Dict, Any
import bentoml
from pydantic import BaseModel, Field
import pandas as pd

CURRENT_YEAR = 2025


# --------- Schéma d’entrée Pydantic ---------
class BuildingInput(BaseModel):
    year_built: int = Field(..., ge=1800, le=CURRENT_YEAR)
    primary_property_type: str
    neighborhood: str
    property_gfa_total: float = Field(..., gt=0)
    number_of_floors: int = Field(..., gt=0)
    number_of_buildings: int = Field(..., gt=0)


# --------- Charger le modèle XGBoost optimisé ---------
xgb_ref = bentoml.sklearn.get("seattle_energy_xgb:latest")
xgb_model = bentoml.sklearn.load_model(xgb_ref.tag)

# Récupérer les features utilisées à l'entraînement
feature_names = xgb_ref.custom_objects["feature_names"]


# --------- Définition du service ---------
@bentoml.service(name="SeattleEnergyXGBService", traffic={"timeout": 300})
class SeattleEnergyXGBService:

    @bentoml.api(input=BuildingInput, output=dict)
    def predict(self, data: BuildingInput) -> Dict[str, Any]:

        # Convertir les données en DataFrame aligné avec l'entraînement
        input_dict = data.model_dump()
        row = [input_dict.get(col) for col in feature_names]

        df = pd.DataFrame([row], columns=feature_names)

        # Prédiction via le pipeline XGBoost
        y_pred = float(xgb_model.predict(df)[0])

        return {
            "prediction": y_pred,
            "unit": "kBtu/sf (SiteEUIWN)",
            "inputs": input_dict,
            "model_version": str(xgb_ref.tag),
        }
