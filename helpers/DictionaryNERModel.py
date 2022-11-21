""" A dictionary-based NER model. Can be used as an alternative
to Flair, which is cumbersome to run and install."""

import os
import json
import pickle as pkl
from abc import ABC, abstractmethod


class NERModel(ABC):

    """Abstract base class for the NER Model."""

    def __init__(self):
        pass

    @abstractmethod
    def train(self, conll_datasets_path: str):
        pass

    @abstractmethod
    def inference(self, raw_sents: list):
        pass

    @abstractmethod
    def load(self, model_path):
        pass

    @abstractmethod
    def save(self, model_path):
        pass


class DictionaryNERModel(NERModel):

    """The dictionary-based NER model. Labels sentences based on
    the most frequent label assigned to each phrase as per the
    training dataset.

    Attributes:
        _chunked_frequency_dict (dict): A dict to keep track of the
          most frequent label for each phrase, for each phrase length.
    """

    model_name: str = "Dictionary-based"

    def __init__(self):
        super(DictionaryNERModel, self).__init__()
        self._chunked_frequency_dict = None

    def train(self, datasets_path: os.path, trained_model_path: os.path):
        """ "Train" the dictionary model on the given conll datasets.
        Note it isn't actually training... just building a dictionary
        from the training/dev datasets and using it as a means to
        heuristically tag the test sents.

        Args:
            datasets_path (os.path): The folder containing the
              train, dev and text CONLL-formatted datasets.
            trained_model_path (os.path): The folder to store the
              'trained model' i.e. the freq dict etc.
        """
        print("Building dictionary...")
        conll_dataset = self._load_conll_data(datasets_path)
        frequency_dict = self._build_frequency_dict(conll_dataset)

        self._chunked_frequency_dict = self._chunk_frequency_dict(
            frequency_dict
        )

        self.save(trained_model_path)

    def load(self, model_path):
        """Load the chunked frequency dict from the given folder.

        Args:
            model_path (str): The filename containing the chunked
              frequency dict (pickle file).

        Raises:
            ValueError: If the model.pkl file is missing, i.e.
              model has not been trained.
        """
        trained_model_path = os.path.join(model_path, "model.pkl")
        if not os.path.exists(trained_model_path) or not os.path.exists(
            model_path
        ):
            raise ValueError(
                "The KGC Model has not yet been trained (the model.pkl"
                " file is missing)."
            )

        with open(trained_model_path, "rb") as f:
            self._chunked_frequency_dict = pkl.load(f)

    def save(self, model_path: os.path):
        """Save the chunked frequency dict inside the given folder.

        Args:
            model_path (os.path): The folder to save the chunked
              frequency dict (pickle file).
        """
        with open(os.path.join(model_path, "model.pkl"), "wb") as f:
            pkl.dump(self._chunked_frequency_dict, f)

    def inference(self, raw_sents: list) -> list:
        """Run the inference on a given list of short texts.

        Raises:
            ValueError: If the model has not yet been trained.

        Args:
            raw_sents (list): The list of raw sents to run the inference on.

        Returns:
            list: The list of documents with predictions.
        """
        if self._chunked_frequency_dict is None:
            raise ValueError(
                "This Dictionary model is not trained yet. "
                "Please run the train function before proceeding."
            )

        preds = []

        min_words = min(self._chunked_frequency_dict.keys())
        max_words = max(self._chunked_frequency_dict.keys()) + 1
        cfd = self._chunked_frequency_dict

        for sent in raw_sents:
            tokens = sent
            labels = ["O"] * len(sent)

            for i, t in enumerate(tokens):

                # If label already predicted (by a larger term),
                # move on.
                if labels[i] != "O":
                    continue

                # Go through each number of words (in reverse order).
                # Check each chunk of words to see whether they are in
                # the cfd. If so, set the labels accordingly.
                for j in reversed(range(min_words, max_words)):
                    if j not in cfd:
                        continue
                    if (i + j) > len(tokens):
                        continue
                    token_str = " ".join(tokens[i : i + j])

                    if token_str in cfd[j]:
                        base_class = cfd[j][token_str]
                        labels[i] = "B-" + base_class
                        for x in range((i + 1), (i + j)):
                            labels[x] = "I-" + base_class
                            i += 1
                            # Skip ahead so the labels are not overwritten

            # Create a ConllDocument from these tokens and labels and append.
            doc = {"tokens": tokens, "labels": labels}
            preds.append(doc)

        return preds

    def _load_conll_data(self, datasets_path: str) -> dict:
        """Load the CONLL-formatted data from the given folder.
        Only loads train and dev, as loading test would give it an unfair
        advantage vs other models.

        Args:
            datasets_path (str): The folder containing the three
              CONLL-formatted files (train.txt, dev.txt, test.txt)

        Returns:
            list: A list of all documents in the training and dev sets. Each
            doc is represented as a dict, i.e.
            {tokens: [list of tokens], labels: [list of labels])}.
        """
        conll_dataset = []
        for ds_name in ["train", "dev"]:
            ds = _load_conll_dataset(
                os.path.join(datasets_path, f"{ds_name}.txt")
            )
            for doc in ds:
                conll_dataset.append(doc)
        return conll_dataset

    def _chunk_frequency_dict(self, frequency_dict: dict) -> dict:
        """Chunk the frequency dict, i.e. split it into a dict where each
        key is the number of words in each phrase, and then each item in that
        key is the most commonly occurring label for that word, e.g.
        1: {
          "pump": "Item"
        },
        2: {
          "big pump": "Item"
        }

        Args:
            frequency_dict (dict): The non-chunked frequency dict.
        """
        _chunked_frequency_dict = {}
        for (phrase, label_freqs) in frequency_dict.items():
            num_words = len(phrase.split(" "))
            label = max(label_freqs, key=label_freqs.get)
            if num_words not in _chunked_frequency_dict:
                _chunked_frequency_dict[num_words] = {}
            _chunked_frequency_dict[num_words].update({phrase: label})

        return _chunked_frequency_dict

    def _build_frequency_dict(self, conll_dataset: list):
        """Build a dictionary of the frequency of each token mapping to
        each label in the given Redcoat dataset.

        Args:
            conll_dataset (list): The Conll dataset to build the
              dict from.

        Returns:
            dict: A dict mapping each entity mention to a dict of {type:
              frequency}.
        """
        frequency_dict = {}

        for doc in conll_dataset:
            phrase_labels = _get_phrase_labels(doc)
            for (phrase, label) in phrase_labels:
                phrase_str = " ".join(phrase)

                if phrase_str not in frequency_dict:
                    frequency_dict[phrase_str] = {}
                if label not in frequency_dict[phrase_str]:
                    frequency_dict[phrase_str][label] = 0
                frequency_dict[phrase_str][label] += 1

        return frequency_dict


def _get_phrase_labels(doc: dict):
    """Return a list of (phrases, labels) for each mention
    in a doc (which is a dict of {"tokens": [tokens in the doc], "labels":
    [labels of the doc]}.
    Each phrase is a list of words of that label, i.e.
    [['centrifugal', 'pump'], 'Item']

    Returns:
        list: A list of (phrase, labels).
    """
    phrase_labels = []
    current_phrase = []
    for i, (token, label) in enumerate(zip(doc["tokens"], doc["labels"])):

        if label.startswith("B-"):
            if len(current_phrase) > 0:
                phrase_labels.append((current_phrase, current_label))
            current_phrase = [token]
            current_label = label[2:]
        elif label.startswith("I-"):
            current_phrase.append(token)
        elif label == "O":
            if len(current_phrase) > 0:
                phrase_labels.append((current_phrase, current_label))
            current_phrase = []
            current_label = None
        if (
            i == len(doc["tokens"]) - 1
            and label != "O"
            and len(current_phrase) > 0
        ):
            phrase_labels.append((current_phrase, current_label))

    return phrase_labels


def _to_conll_document(s: str):
    """Create a ConllDocument from a string as it appears
    in a Conll-formatted file.

    Args:
        s (str): A string, separated by newlines, where each
        line is a token, then a comma and space, then a label.

    Returns:
        dict: A dict of tokens and labels.
    """
    tokens, labels = [], []
    for line in s.split("\n"):
        if len(line.strip()) == 0:
            continue
        token, label = line.split()

        tokens.append(token)
        labels.append(label)
    return {"tokens": tokens, "labels": labels}


def _load_conll_dataset(filename: str) -> list:
    """Load a list of documents from the given CONLL-formatted dataset.

    Args:
        filename (str): The filename to load from.

    Returns:
        list: A list of documents, where each document is a dict of tokens and
        labels.
    """
    documents = []
    with open(filename, "r") as f:
        docs = f.read().split("\n\n")
        for d in docs:
            if len(d) == 0:
                continue
            document = _to_conll_document(d)
            documents.append(document)
    print(f"Loaded {len(documents)} documents from {filename}.")
    return documents
