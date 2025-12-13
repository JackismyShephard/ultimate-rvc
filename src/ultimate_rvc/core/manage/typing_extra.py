"""
Module which defines extra types used by modules in the
ultimate_rvc.core.manage package.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, RootModel

from ultimate_rvc.typing_extra import TrainingSampleRate


class VoiceModelTagName(StrEnum):
    """Names of valid voice model tags."""

    ENGLISH = "English"
    JAPANESE = "Japanese"
    OTHER_LANGUAGE = "Other Language"
    ANIME = "Anime"
    VTUBER = "Vtuber"
    REAL_PERSON = "Real person"
    GAME_CHARACTER = "Game character"


class VoiceModelTagMetaData(BaseModel):
    """
    Metadata for a voice model tag.

    Attributes
    ----------
    name : ModelTagName
        The name of the tag.
    description : str
        The description of the tag.

    """

    name: VoiceModelTagName
    description: str


class VoiceModelMetaData(BaseModel):
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
    tags: list[VoiceModelTagName]
    credit: str
    added: str
    url: str


class VoiceModelMetaDataTable(BaseModel):
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

    tags: list[VoiceModelTagMetaData]
    models: list[VoiceModelMetaData]


VoiceModelMetaDataPredicate = Callable[[VoiceModelMetaData], bool]

VoiceModelMetaDataList = list[list[str | list[VoiceModelTagName]]]


class PretrainedPaths(BaseModel):
    """
    Paths to the generator and discriminator for a pretrained
    model with a given name and sample rate.
    """

    G: str
    D: str


class PretrainedModelMetaData(RootModel[dict[TrainingSampleRate, PretrainedPaths]]):
    """
    Metadata for a pretrained model with a given name.

    Attributes
    ----------
    root : dict[TrainingSampleRate, PretrainedPaths]
        Mapping from sample rate to paths to the generator and
        discriminator for the pretrained model with the given name
        at the given sample rate.

    """

    root: dict[TrainingSampleRate, PretrainedPaths]

    def __getitem__(self, item: TrainingSampleRate) -> PretrainedPaths:
        """
        Get the paths to the generator and discriminator for the
        pretrained model at the given sample rate.

        Parameters
        ----------
        item : TrainingSampleRate
            The sample rate for which to get paths to the generator
            and discriminator for the pretrained model.

        Returns
        -------
        PretrainedPaths
            The paths to the generator and discriminator for the
            pretrained model at the given sample rate.

        """
        return self.root[item]

    def keys(self) -> list[TrainingSampleRate]:
        """
        Get the sample rates for which generator and discriminator
        paths are available for the pretrained model.

        Returns
        -------
        list[TrainingSampleRate]
            The sample rates for which paths are available for the
            pretrained model.

        """
        return sorted(self.root.keys())


class PretrainedModelMetaDataTable(RootModel[dict[str, PretrainedModelMetaData]]):
    """
    Table with metadata for pretrained models available online.

    Attributes
    ----------
    root : dict[str, PretrainedModelMetaData]
        Mapping from the names of pretrained models to metadata for
        those models.

    """

    root: dict[str, PretrainedModelMetaData]

    @property
    def names(self) -> list[str]:
        """
        Get the names of all pretrained models available online.

        Returns
        -------
        list[str]
            The names of all pretrained models available online.

        """
        return self.keys()

    @property
    def default_name(self) -> str | None:
        """
        Get the name of a default pretrained model, if at least one
        pretrained model is available online.

        Returns
        -------
        str | None
            The name of a default pretrained model, or None if no
            pretrained models are available online.

        """
        titan_name = "Titan"
        return titan_name if titan_name in self.names else next(iter(self.names), None)

    @property
    def default_sample_rates(self) -> list[TrainingSampleRate]:
        """
        Get the sample rates for which instances of the default
        pretrained model are available online.

        Returns
        -------
        list[TrainingSampleRate]
            The sample rates for which instances of the default
            pretrained model are available online.

        """
        return self.get_sample_rates(self.default_name) if self.default_name else []

    @property
    def default_sample_rate(self) -> TrainingSampleRate | None:
        """
        Get the first sample rate for which an instance of the default
        pretrained model is available online.

        Returns
        -------
        TrainingSampleRate | None
            The first sample rate for which an instance of the default
            pretrained model is available online, or if no instances are
            available online.

        """
        return next(iter(self.default_sample_rates), None)

    def get_sample_rates(self, name: str) -> list[TrainingSampleRate]:
        """
        Get the sample rates for which instances of the pretrained
        model with the provided name are available online.

        Parameters
        ----------
        name : str
            The name of the pretrained model for which to get available
            sample rates.

        Returns
        -------
        list[TrainingSampleRate]
            The sample rates for which there are instances of the
            pretrained model with the provided name available online.

        """
        return self[name].keys() if name in self.names else []

    def get_sample_rates_with_default(
        self,
        name: str,
    ) -> tuple[TrainingSampleRate | None, list[TrainingSampleRate]]:
        """
        Get the sample rates for which instances of the pretrained
        model with the provided name are available online, and the
        default sample rate for that model.

        Parameters
        ----------
        name : str
            The name of the pretrained model for which to get available
            sample rates.

        Returns
        -------
        tuple[TrainingSampleRate | None, list[TrainingSampleRate]]
            A tuple containing the default sample rate for the
            pretrained model with the provided name, and a list of all
            sample rates for which there are instances of that model
            available online.

        """
        sample_rates = self.get_sample_rates(name)
        return next(iter(sample_rates), None), sample_rates

    def __getitem__(self, item: str) -> PretrainedModelMetaData:
        """
        Get the metadata for the pretrained model with the given name.

        Parameters
        ----------
        item : str
            The name of the pretrained model for which to get metadata.

        Returns
        -------
        PretrainedModelMetaData
            The metadata for the pretrained model with the given name.

        """
        return self.root[item]

    def keys(self) -> list[str]:
        """
        Get the names of all pretrained models available online.

        Returns
        -------
        list[str]
            The names of all pretrained models available online.

        """
        return sorted(self.root.keys())
