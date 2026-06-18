from typing import Dict, Optional

from pydantic import BaseModel


class SimpleParameterConfiguration(BaseModel):
    netcdf_name: str
    units: str


class ParameterConfiguration(BaseModel):
    conversion_method: Optional[str]
    variables: Dict[str, SimpleParameterConfiguration]


class Configuration(BaseModel):
    area: str
    parameters: Dict[str, ParameterConfiguration]
