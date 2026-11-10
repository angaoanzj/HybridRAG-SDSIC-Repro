"""Frozen contextual cue matching for five controlled vocabulary treatments."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from wordnet_lexicon import WordNetLexicon

OLD_CUES = frozenset("""directed director written writer author authored born birthplace spouse married husband wife father mother son daughter founded founder created creator produced producer composed composer""".split())

@lru_cache(maxsize=1)
def resources():
    path = Path(os.environ["HYBRIDRAG_CUE_FILE"])
    data = path.read_bytes()
    expected = os.environ["HYBRIDRAG_CUE_SHA256"]
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Cue configuration hash differs")
    config = json.loads(data)
    if config["status"] != "mined_without_task_labels" or not config["cues"]:
        raise ValueError("Unfinished vocabulary")
    directory = Path(os.environ["HYBRIDRAG_WORDNET_DIR"])
    for name, expected_hash in config["wordnet_file_hashes"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("WordNet resource changed")
    return config, WordNetLexicon(directory)

class CueMatcher:
    def __init__(self, mode):
        if mode not in ("old24", "none", "unsupervised", "morphology", "synonyms"):
            raise ValueError("Unknown vocabulary treatment")
        self.mode = mode
        config, lexicon = resources()
        self.lexicon = lexicon
        self.original = frozenset(config["cues"])
        self.keys = {tuple(item) for item in config["morph_keys"]}
        if mode == "synonyms":
            self.keys.update(tuple(item) for item in config["synonym_keys"])

    @lru_cache(maxsize=300000)
    def __contains__(self, token):
        if self.mode == "none":
            return False
        if self.mode == "old24":
            return token in OLD_CUES
        if token in self.original:
            return True
        if self.mode == "unsupervised":
            return False
        return bool(self.lexicon.keys(token) & self.keys)

@lru_cache(maxsize=5)
def get_matcher(mode):
    return CueMatcher(mode)
