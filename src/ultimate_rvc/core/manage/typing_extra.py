"""
Module which defines extra types for used by modules in the
ultimate_rvc.core.manage package.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel

# Voice model management


class ModelTagName(StrEnum):
    """Names of valid voice model tags."""

    ENGLISH = "English"
    JAPANESE = "Japanese"
    OTHER_LANGUAGE = "Other Language"
    ANIME = "Anime"
    VTUBER = "Vtuber"
    REAL_PERSON = "Real person"
    GAME_CHARACTER = "Game character"


class ModelTagMetaData(BaseModel):
    """
    Metadata for a voice model tag.

    Attributes
    ----------
    name : ModelTagName
        The name of the tag.
    description : str
        The description of the tag.

    """

    name: ModelTagName
    description: str


class ModelMetaData(BaseModel):
    """
    Metadata for a voice model.

    Attributes
    ----------
    name : str
        The name of the voice model.
    description : str
        A description of the voice model.
    tags : list[ModelTagName]
        The tags associated with the voice model.
    credit : str
        Who created the voice model.
    added : str
        The date the voice model was created.
    url : str
        An URL pointing to a location where the voice model can be
        downloaded.

    """

    name: str
    description: str
    tags: list[ModelTagName]
    credit: str
    added: str
    url: str


class ModelMetaDataTable(BaseModel):
    """
    Table with metadata for a set of voice models.

    Attributes
    ----------
    tags : list[ModelTagMetaData]
        Metadata for the tags associated with the given set of voice
        models.
    models : list[ModelMetaData]
        Metadata for the given set of voice models.

    """

    tags: list[ModelTagMetaData]
    models: list[ModelMetaData]


ModelMetaDataPredicate = Callable[[ModelMetaData], bool]

ModelMetaDataList = list[list[str | list[ModelTagName]]]
