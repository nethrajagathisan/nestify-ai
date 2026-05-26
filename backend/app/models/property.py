from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PropertyType(str, Enum):
    APARTMENT = "apartment"
    VILLA = "villa"
    PLOT = "plot"
    COMMERCIAL = "commercial"


class BHK(int, Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


class City(str, Enum):
    BANGALORE = "Bangalore"
    CHENNAI = "Chennai"
    HYDERABAD = "Hyderabad"
    MUMBAI = "Mumbai"
    PUNE = "Pune"
    DELHI = "Delhi"
    COIMBATORE = "Coimbatore"


class Property(BaseModel):
    id: str
    title: str
    city: City
    locality: str
    bhk: BHK
    price_lakhs: float = Field(..., gt=0)
    area_sqft: int = Field(..., gt=0)
    property_type: PropertyType
    amenities: list[str] = Field(default_factory=list)
    description: str
    year_built: Optional[int] = None
    parking: bool = False

    @field_validator("year_built")
    @classmethod
    def validate_year_built(cls, v: Optional[int]) -> Optional[int]:
        if v is not None:
            current_year = 2026
            if v < 1900 or v > current_year:
                raise ValueError(f"year_built must be between 1900 and {current_year}")
        return v


class PropertySearchRequest(BaseModel):
    query: str
    city: Optional[City] = None
    min_price_lakhs: Optional[float] = Field(default=None, ge=0)
    max_price_lakhs: Optional[float] = Field(default=None, ge=0)
    bhk: Optional[BHK] = None
    property_type: Optional[PropertyType] = None
    top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_price_range(cls, values):
        if values.min_price_lakhs is not None and values.max_price_lakhs is not None:
            if values.min_price_lakhs > values.max_price_lakhs:
                raise ValueError("min_price_lakhs must be less than or equal to max_price_lakhs")
        return values


class PropertySearchResponse(BaseModel):
    results: list[Property]
    llm_summary: str
    total_found: int
