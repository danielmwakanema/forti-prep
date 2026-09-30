from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class SimpleParameterConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    netcdf_name: str
    units: str
    source_units: Optional[str] = None


class TimeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str = "time"
    variable: Optional[str] = None

    def resolved_variable(self) -> str:
        return self.variable or self.dimension


class DimensionConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: str
    y: str
    time: TimeConfiguration = Field(default_factory=TimeConfiguration)


class CoordinateConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: str
    longitude: str


class ParameterConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversion_method: Optional[str]
    variables: Dict[str, SimpleParameterConfiguration]
    dim: Optional[str] = None
    index: int = 0
    hours: Optional[int] = None
    timezone: Optional[str] = None


class Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area: str

    dimensions: DimensionConfiguration
    coordinates: Optional[CoordinateConfiguration] = None

    parameters: Dict[str, ParameterConfiguration]
