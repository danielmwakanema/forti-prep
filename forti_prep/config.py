from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict


class SimpleParameterConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    netcdf_name: str
    units: str


class DimensionConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: str
    y: str


class ParameterConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversion_method: Optional[str]
    variables: Dict[str, SimpleParameterConfiguration]


class Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area: str

    dimensions: DimensionConfiguration

    parameters: Dict[str, ParameterConfiguration]
