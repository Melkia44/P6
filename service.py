import bentoml
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any

# -------------------------------------------------------------------
#  Constantes métier
# -------------------------------------------------------------------

CURRENT_YEAR = 2025
BUILDING_TYPE_VALUE = "NonResidential"  # P6 : uniquement bâtiments non résidentiels
MODEL_TAG = "seattle_energy_predictor:latest"

# Mapping entre les champs exposés à l'utilisateur
# et les noms de colonnes réels dans le X_train / dataset
COLUMN_MAPPING = {
    "year_built": "YearBuilt",
    "primary_property_type": "PrimaryPropertyType",
    "neighborhood": "Neighborhood",
    "property_gfa_total": "PropertyGFATotal",
    "number_of_floors": "NumberofFloors",
    "number_of_buildings": "NumberofBuildings",
}

# -------------------------------------------------------------------
#  Schéma d'entrée (Pydantic v2)
# -------------------------------------------------------------------

class BuildingInput(BaseModel):
    year_built: int = Field(
        ...,
        ge=1800,
        le=CURRENT_YEAR,
        description="Année de construction du bâtiment",
    )
    primary_property_type: str = Field(
        ...,
        description="Type principal de propriété (ex: Office, Hotel, K-12 School...)",
    )
    neighborhood: Optional[str] = Field(
        default=None,
        description="Quartier de Seattle (ex: DOWNTOWN, BALLARD...). Facultatif.",
    )
    property_gfa_total: float = Field(
        ...,
        gt=0,
        description="Surface totale du bâtiment (PropertyGFATotal) en ft²",
    )
    number_of_floors: int = Field(
        ...,
        gt=0,
        description="Nombre d'étages",
    )
    number_of_buildings: int = Field(
        ...,
        gt=0,
        description="Nombre de bâtiments sur le site",
    )

    @field_validator("property_gfa_total")
    @classmethod
    def gfa_sensible(cls, v: float) -> float:
        if v > 2_000_000:
            raise ValueError("property_gfa_total semble trop élevé pour un bâtiment standard.")
        return v

    @field_validator("year_built")
    @classmethod
    def year_sensible(cls, v: int) -> int:
        if v < 1850:
            raise ValueError("year_built est trop ancien pour être réaliste dans ce contexte.")
        return v


# -------------------------------------------------------------------
#  Service BentoML
# -------------------------------------------------------------------

@bentoml.service
class SeattleEnergyService:
    def __init__(self) -> None:
        # Chargement du pipeline sklearn
        self.model = bentoml.sklearn.load_model(MODEL_TAG)
        # Récupération des meta (feature_names) depuis le Model Store
        model_ref = bentoml.models.get(MODEL_TAG)
        self.feature_names = model_ref.custom_objects["feature_names"]

    @bentoml.api
    def predict(self, building: BuildingInput) -> Dict[str, Any]:
        """
        Endpoint /predict
        - Valide les données via Pydantic (BuildingInput)
        - Map les champs utilisateur -> colonnes du modèle
        - Ajoute BuildingType = 'NonResidential'
        - Réaligne les features dans l'ordre de X_train
        - Retourne la prédiction sous forme de dict JSON-sérialisable
        """

        # 1) Données validées par Pydantic
        input_data = building.model_dump(exclude_none=True)

        # 2) Construction du dict au format attendu par le modèle
        data_for_model: Dict[str, Any] = {}

        for input_field, value in input_data.items():
            if input_field in COLUMN_MAPPING:
                col_name = COLUMN_MAPPING[input_field]
                data_for_model[col_name] = value

        # 3) Ajout du type de bâtiment (constant dans P6)
        data_for_model["BuildingType"] = BUILDING_TYPE_VALUE

        # 4) DataFrame une ligne
        df = pd.DataFrame([data_for_model])

        # 5) Réalignement des colonnes
        df_model = df.reindex(columns=self.feature_names, fill_value=0)

        # 6) Prédiction
        y_pred = self.model.predict(df_model)[0]

        # 7) Réponse JSON
        return {
            "prediction": float(y_pred),
            "unit": "kBtu/sf",
            "message": "Prévision de SiteEUI (consommation énergétique normalisée).",
            "inputs_interpreted": data_for_model,
        }
