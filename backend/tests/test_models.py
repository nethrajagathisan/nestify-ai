import pytest
from pydantic import ValidationError

from backend.app.models.property import (
    Property,
    PropertyType,
    BHK,
    City,
    PropertySearchRequest,
    PropertySearchResponse,
)


class TestPropertyType:
    def test_property_type_enum_values(self):
        assert PropertyType.APARTMENT.value == "apartment"
        assert PropertyType.VILLA.value == "villa"
        assert PropertyType.PLOT.value == "plot"
        assert PropertyType.COMMERCIAL.value == "commercial"


class TestBHK:
    def test_bhk_enum_values(self):
        assert BHK.ONE.value == 1
        assert BHK.TWO.value == 2
        assert BHK.THREE.value == 3
        assert BHK.FOUR.value == 4


class TestProperty:
    def test_property_with_negative_price_raises_error(self):
        with pytest.raises(ValidationError) as exc_info:
            Property(
                id="test_1",
                title="Test Property",
                city=City.BANGALORE,
                locality="Whitefield",
                bhk=BHK.TWO,
                price_lakhs=-50.0,
                area_sqft=1200,
                property_type=PropertyType.APARTMENT,
                description="Test description",
            )
        assert "price_lakhs" in str(exc_info.value).lower()

    def test_property_with_negative_area_raises_error(self):
        with pytest.raises(ValidationError) as exc_info:
            Property(
                id="test_1",
                title="Test Property",
                city=City.BANGALORE,
                locality="Whitefield",
                bhk=BHK.TWO,
                price_lakhs=50.0,
                area_sqft=-1200,
                property_type=PropertyType.APARTMENT,
                description="Test description",
            )
        assert "area_sqft" in str(exc_info.value).lower()

    def test_property_with_invalid_year_built_raises_error(self):
        with pytest.raises(ValidationError) as exc_info:
            Property(
                id="test_1",
                title="Test Property",
                city=City.BANGALORE,
                locality="Whitefield",
                bhk=BHK.TWO,
                price_lakhs=50.0,
                area_sqft=1200,
                property_type=PropertyType.APARTMENT,
                description="Test description",
                year_built=1800,
            )
        assert "year_built" in str(exc_info.value).lower()

    def test_property_model_dump_serializes_correctly(self):
        prop = Property(
            id="test_1",
            title="Test Property",
            city=City.BANGALORE,
            locality="Whitefield",
            bhk=BHK.TWO,
            price_lakhs=50.0,
            area_sqft=1200,
            property_type=PropertyType.APARTMENT,
            description="Test description",
            amenities=["gym", "pool"],
            year_built=2020,
            parking=True,
        )
        dumped = prop.model_dump()
        
        assert dumped["id"] == "test_1"
        assert dumped["title"] == "Test Property"
        assert dumped["city"] == City.BANGALORE
        assert dumped["bhk"] == BHK.TWO
        assert dumped["price_lakhs"] == 50.0
        assert dumped["area_sqft"] == 1200
        assert dumped["property_type"] == PropertyType.APARTMENT
        assert dumped["amenities"] == ["gym", "pool"]
        assert dumped["year_built"] == 2020
        assert dumped["parking"] is True


class TestPropertySearchRequest:
    def test_invalid_top_k_greater_than_20_raises_error(self):
        with pytest.raises(ValidationError) as exc_info:
            PropertySearchRequest(
                query="Test query",
                top_k=25,
            )
        assert "top_k" in str(exc_info.value).lower()

    def test_invalid_top_k_less_than_1_raises_error(self):
        with pytest.raises(ValidationError) as exc_info:
            PropertySearchRequest(
                query="Test query",
                top_k=0,
            )
        assert "top_k" in str(exc_info.value).lower()

    def test_min_price_greater_than_max_price_raises_error(self):
        with pytest.raises(ValidationError) as exc_info:
            PropertySearchRequest(
                query="Test query",
                min_price_lakhs=100,
                max_price_lakhs=50,
            )
        assert "min_price_lakhs" in str(exc_info.value).lower()

    def test_valid_property_search_request(self):
        request = PropertySearchRequest(
            query="3BHK in Bangalore",
            city=City.BANGALORE,
            bhk=BHK.THREE,
            min_price_lakhs=30,
            max_price_lakhs=80,
            property_type=PropertyType.APARTMENT,
            top_k=10,
        )
        assert request.query == "3BHK in Bangalore"
        assert request.city == City.BANGALORE
        assert request.bhk == BHK.THREE
        assert request.min_price_lakhs == 30
        assert request.max_price_lakhs == 80
        assert request.property_type == PropertyType.APARTMENT
        assert request.top_k == 10


class TestPropertySearchResponse:
    def test_property_search_response_structure(self):
        prop = Property(
            id="test_1",
            title="Test Property",
            city=City.BANGALORE,
            locality="Whitefield",
            bhk=BHK.TWO,
            price_lakhs=50.0,
            area_sqft=1200,
            property_type=PropertyType.APARTMENT,
            description="Test description",
        )
        
        response = PropertySearchResponse(
            results=[prop],
            llm_summary="Test summary",
            total_found=1,
        )
        
        assert len(response.results) == 1
        assert response.results[0].id == "test_1"
        assert response.llm_summary == "Test summary"
        assert response.total_found == 1
