import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.processors import TemplateProcessing

class KlyroTokenizer:
    def __init__(self, vocab_size=16000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.special_tokens = {
            "pad": "[PAD]",
            "unk": "[UNK]",
            "bos": "[BOS]",
            "eos": "[EOS]",
        }

    def train(self, files_iterator):
        """Train tokenizer on an iterator of text strings."""
        # Initialize a BPE model
        self.tokenizer = Tokenizer(models.BPE(unk_token=self.special_tokens["unk"]))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.special_tokens['bos']} $A {self.special_tokens['eos']}",
            pair=f"{self.special_tokens['bos']} $A {self.special_tokens['eos']} {self.special_tokens['bos']} $B {self.special_tokens['eos']}",
            special_tokens=[
                (self.special_tokens["bos"], self.tokenizer.get_vocab().get(self.special_tokens["bos"], 0)),
                (self.special_tokens["eos"], self.tokenizer.get_vocab().get(self.special_tokens["eos"], 0)),
            ],
        )

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(self.special_tokens.values()),
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        self.tokenizer.train_from_iterator(files_iterator, trainer=trainer)

    def save(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        self.tokenizer.save(os.path.join(directory, "tokenizer.json"))
        # Save special tokens mapping for convenience
        with open(os.path.join(directory, "special_tokens.json"), "w", encoding="utf-8") as f:
            json.dump(self.special_tokens, f, ensure_ascii=False, indent=2)

    def load(self, directory):
        self.tokenizer = Tokenizer.from_file(os.path.join(directory, "tokenizer.json"))
        with open(os.path.join(directory, "special_tokens.json"), "r", encoding="utf-8") as f:
            self.special_tokens = json.load(f)

    def encode(self, text):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded.")
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded.")
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size() if self.tokenizer else self.vocab_size

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, idx):
        return self.tokenizer.id_to_token(idx)