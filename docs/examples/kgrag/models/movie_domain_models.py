"""Domain-specific entity models for the movie domain example.

This module demonstrates how to extend the base Entity class for a specific domain.
Users can follow this pattern to create domain-specific entities for their own domains.

Note:
    This example assumes mellea-contribs is installed. Install with:
    pip install mellea-contribs[kg]
"""

from typing import Optional

from pydantic import Field

from mellea_contribs.kg.models import Entity


class MovieEntity(Entity):
    """Movie-specific entity.

    Extends the base Entity class with movie-domain properties.
    All extraction and storage fields are inherited from Entity.
    """

    release_year: Optional[int] = Field(default=None, description="Year the movie was released")
    director: Optional[str] = Field(default=None, description="Director(s) of the movie")
    box_office: Optional[float] = Field(
        default=None, description="Box office earnings in millions USD"
    )
    language: Optional[str] = Field(default=None, description="Primary language of the movie")
    rating: Optional[float] = Field(default=None, description="IMDb/review rating 0-10")


class PersonEntity(Entity):
    """Person-specific entity (actor, director, producer, etc.).

    Extends the base Entity class with person-domain properties.
    """

    birth_year: Optional[int] = Field(default=None, description="Birth year")
    nationality: Optional[str] = Field(default=None, description="Nationality/country")
    profession: Optional[str] = Field(
        default=None, description="Primary profession (actor, director, producer, etc.)"
    )


class AwardEntity(Entity):
    """Award-specific entity (Academy Award, Golden Globe, etc.).

    Extends the base Entity class with award-domain properties.
    """

    ceremony_number: Optional[int] = Field(
        default=None, description="Ceremony/edition number (e.g., 96th Academy Awards)"
    )
    award_type: Optional[str] = Field(default=None, description="Type of award (e.g., Best Picture)")
    award_year: Optional[int] = Field(default=None, description="Year of the award ceremony")


__all__ = [
    "MovieEntity",
    "PersonEntity",
    "AwardEntity",
]
