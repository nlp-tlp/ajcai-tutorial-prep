""" A simple heuristic-based RE model. Can be used as an alternative
to Flair, but is only designed for co-occurrence relations on MWOs."""

import os
import json
import pickle as pkl
from typing import List

from abc import ABC, abstractmethod


class REModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def inference(self, row: list) -> str:
        pass

    @abstractmethod
    def train(self, re_datasets_path: str):
        pass

    @abstractmethod
    def load(self, model_path: str):
        pass

    @abstractmethod
    def save(self, model_path: str):
        pass


class SimpleMWOREModel(REModel):

    """The SimpleMWORE model. Creates a relation between each Item
    and every other entity appearing in the work order.
    """

    model_name: str = "Simple MWO"

    def __init__(self):
        super(SimpleMWOREModel, self).__init__()
        self._chunked_frequency_dict = None

    def train(self, _, __):
        """The simple MWO model does not require training, and thus this
        function is just here so that we don't have to write special code
        just to run this model.
        """
        pass

    def load(self, _):
        """Same as 'train', nothing to do here."""
        pass

    def save(self, _):
        """Same as above."""
        pass

    def inference(self, row: list) -> str:
        """Run the inference over the given row.

        The Simple MWO RE model create relationships between Items
        and all other entity types in the given Redcoat Document.

        It will thus return a relation when Entity 1 is an item.

        Args:
            doc (RedcoatDocument): The Redcoat document.

        Returns:
            List[dict]: List of relations.
        """
        rel_type = "O"
        if row[2] == "Item":
            return f"HAS_{row[3].upper()}"
        return rel_type
