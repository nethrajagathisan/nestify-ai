"""Tests for Pydantic models."""

import pytest
from backend.app.models.property import (
    Property,
    PropertySearchRequest,
    PropertySearchResponse,
    City,
    BHK,
    PropertyType,
)


class TestPropertyModel:
    def test_valid_property(self):
        prop = Property(
            id="prop_001",
            title="Modern 2BHK Apartment in Koramangala, Bangalore",
            city=City.BANGALORE,
            locality="Koramangala",
            bhk=BHK.TWO,
            price_lakhs=85.5,
            area_sqft=1200,
            property_type=PropertyType.APARTMENT,
            amenities=["gym", "parking", "security"],
            description="Great property in IT corridor.",
            year_built=2020,
            parking=True,
        )
        assert prop.id == "prop_001"
        assert prop.price_lakhs == 85.5
        assert prop.area_sqft == 1200

    def test_year_built_validation(self):
        with pytest.raises(ValueError, match="year_built must be between"):
            Property(
                id="prop_002",
                title="Old Property",
                city=City.CHENNAI,
                locality="Adyar",
                bhk=BHK.THREE,
                price_lakhs=120.0,
                area_sqft=1500,
                property_type=PropertyType.VILLA,
                amenities=[],
                description="Test.",
                year_built=1850,
            )

    def test_price_must_be_positive(self):
        with pytest.raises(ValueError):
            Property(
                id="prop_003",
                title="Bad Price",
                city=City.MUMBAI,
                locality="Bandra",
                bhk=BHK.ONE,
                price_lakhs=0,
                area_sqft=500,
                property_type=PropertyType.APARTMENT,
                amenities=[],
                description="Test.",
            )

    def test_area_must_be_positive(self):
        with pytest.raises(ValueError):
            Property(
                id="prop_004",
                title="Bad Area",
                city=City.PUNE,
                locality="Koregaon Park",
                bhk=BHK.TWO,
                price_lakhs=50.0,
                area_sqft=-100,
                property_type=PropertyType.APARTMENT,
                amenities=[],
                description="Test.",
            )

    def test_optional_year_built(self):
        prop = Property(
            id="prop_005",
            title="No Year",
            city=City.HYDERABAD,
            locality="Gachibowli",
            bhk=BHK.FOUR,
            price_lakhs=200.0,
            area_sqft=2500,
            property_type=PropertyType.VILLA,
            amenities=["pool", "gym"],
            description="Test.",
        )
        assert prop.year_built is None


class TestPropertySearchRequest:
    def test_valid_request(self):
        req = PropertySearchRequest(query="2BHK apartment in Bangalore")
        assert req.query == "2BHK apartment in Bangalore"
        assert req.top_k == 5

    def test_request_with_filters(self):
        req = PropertySearchRequest(
            query="budget home",
            city=City.CHENNAI,
            min_price_lakhs=30.0,
            max_price_lakhs=80.0,
            bhk=BHK.TWO,
            property_type=PropertyType.APARTMENT,
            top_k=10,
        )
        assert req.city == City.CHENNAI
        assert req.min_price_lakhs == 30.0
        assert req.max_price_lakhs == 80.0

    def test_price_range_validation(self):
        with pytest.raises(ValueError, match="min_price_lakhs must be less than"):
            PropertySearchRequest(
                query="invalid",
                min_price_lakhs=100.0,
                max_price_lakhs=50.0,
            )

    def test_top_k_bounds(self):
        with pytest.raises(ValueError):
            PropertySearchRequest(query="test", top_k=0)
        with pytest.raises(ValueError):
            PropertySearchRequest(query="test", top_k=25)


class TestPropertySearchResponse:
    def test_response(self):
        prop = Property(
            id="prop_001",
            title="Test",
            city=City.BANGALORE,
            locality="Test",
            bhk=BHK.TWO,
            price_lakhs=50.0,
            area_sqft=1000,
            property_type=PropertyType.APARTMENT,
            amenities=[],
            description="Test.",
        )
        resp = PropertySearchResponse(
            results=[prop],
            llm_summary="Found 1 result",
            total_found=1,
        )
        assert resp.total_found == 1
        assert len(resp.results) == 1
