""" A lexicon-based normaliser. Normalises sentences by replacing any ngrams 
(sequences of 1 or more words) with their replacement as per a predefined
lexicon."""

from csv import DictReader
import itertools


class LexiconNormaliser:
    """A lexicon-based normaliser.

    Args:
        lexicon_file: The filename of the lexicon.

    """

    def __init__(self, lexicon_file, max_ngram_size=3):

        lexicon_data = _load_csv(lexicon_file)
        self.max_ngram_size = max_ngram_size

        # Convert the loaded csv into a dictionary mapping incorrect form ->
        # correct form
        self.lexicon = {}
        for row in lexicon_data:
            self.lexicon[row["key"]] = row["value"]

    def normalise(self, sentence: str):
        """
        Normalise the given sentence via the lexicon.

        Args:
            sentence(str): The sentence to normalise.

        Returns:
            str: The normalised sentence.
        """
        words = sentence.split()
        ngrams = self._get_ngrams(words)

        # Reversing ngrams ensures the larger ngrams are normalised first.
        for ngram in reversed(ngrams):
            if ngram in self.lexicon:
                sentence = sentence.replace(ngram, self.lexicon[ngram])

        return sentence

    def _get_ngrams(self, sentence):
        """
        Given a sentence, return a list of all combinations of ngrams
        up to a certain size.

        Args:
            sentence: A list of words, e.g. ["fix", "broken", "pump"].

        Returns:
            ngrams: A list of ngrams containing up to max_ngram_size words.
                    For example, given the input ["fix", "broken", "pump"],
                    return ["fix", "broken", "pump", "fix broken",
                            "broken pump", "fix broken pump"]

        """
        ngrams = []
        for n in range(self.max_ngram_size):
            for c in itertools.combinations(sentence, n + 1):
                ngrams.append(" ".join(c))
        return ngrams


# A simple function to read in a csv file and return a list,
# where each element in the list is a dictionary of {heading : value}
def _load_csv(filename):
    data = []
    with open(filename, "r") as f:
        reader = DictReader(f)
        for row in reader:
            data.append(row)
    return data
