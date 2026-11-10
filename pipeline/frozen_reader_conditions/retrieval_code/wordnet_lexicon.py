"""Small deterministic reader for WordNet 3.0 noun and verb dictionary files."""
from functools import lru_cache
from pathlib import Path
import re

TOKEN = re.compile(r"[a-z0-9]+", re.I)
SUBSTITUTIONS = {
    "n": (("s", ""), ("ses", "s"), ("ves", "f"), ("xes", "x"),
          ("zes", "z"), ("ches", "ch"), ("shes", "sh"), ("men", "man"), ("ies", "y")),
    "v": (("s", ""), ("ies", "y"), ("es", "e"), ("es", ""),
          ("ed", "e"), ("ed", ""), ("ing", "e"), ("ing", "")),
}

class WordNetLexicon:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.senses = {}
        self.by_lemma = {"n": {}, "v": {}}
        self.exceptions = {"n": {}, "v": {}}
        for pos, name in (("n", "noun"), ("v", "verb")):
            with (self.directory / f"data.{name}").open() as stream:
                for line in stream:
                    if not line.strip() or line.startswith(" "):
                        continue
                    data, _, gloss = line.partition("|")
                    fields = data.split()
                    count = int(fields[3], 16)
                    lemmas = [fields[4 + 2 * i].lower() for i in range(count)]
                    key = pos + ":" + fields[0]
                    self.senses[key] = {"pos": pos, "lemmas": lemmas, "gloss": gloss.strip()}
                    for lemma in lemmas:
                        self.by_lemma[pos].setdefault(lemma, []).append(key)
            with (self.directory / f"{name}.exc").open() as stream:
                for line in stream:
                    fields = line.split()
                    if fields:
                        self.exceptions[pos][fields[0]] = fields[1:]

    @lru_cache(maxsize=300000)
    def forms(self, token, pos):
        candidates = [token]
        if token in self.exceptions[pos]:
            candidates.extend(self.exceptions[pos][token])
        else:
            for ending, replacement in SUBSTITUTIONS[pos]:
                if token.endswith(ending):
                    candidates.append(token[:-len(ending)] + replacement)
        return tuple(dict.fromkeys(word for word in candidates if word in self.by_lemma[pos]))

    def keys(self, token):
        return {(lemma, pos) for pos in ("n", "v") for lemma in self.forms(token, pos)}

    def sense_keys(self, token):
        return sorted({key for lemma, pos in self.keys(token) for key in self.by_lemma[pos][lemma]})
