import unittest
import os
import tempfile
from src.tokenizer import KlyroTokenizer

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = KlyroTokenizer(vocab_size=100)
        self.temp_dir = tempfile.mkdtemp()

    def test_encode_decode(self):
        # Train on dummy data
        dummy_data = ["hello world", "foo bar", "test"]
        self.tokenizer.train(iter(dummy_data))
        ids = self.tokenizer.encode("hello world")
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))
        text = self.tokenizer.decode(ids)
        self.assertIsInstance(text, str)

    def test_save_load(self):
        dummy_data = ["hello world", "foo bar"]
        self.tokenizer.train(iter(dummy_data))
        save_path = os.path.join(self.temp_dir, "tokenizer")
        self.tokenizer.save(save_path)
        # Load new tokenizer
        new_tok = KlyroTokenizer(vocab_size=100)
        new_tok.load(save_path)
        ids1 = self.tokenizer.encode("hello world")
        ids2 = new_tok.encode("hello world")
        self.assertEqual(ids1, ids2)

    def test_special_tokens(self):
        self.assertIn("pad", self.tokenizer.special_tokens)
        self.assertIn("unk", self.tokenizer.special_tokens)
        self.assertIn("bos", self.tokenizer.special_tokens)
        self.assertIn("eos", self.tokenizer.special_tokens)

if __name__ == '__main__':
    unittest.main()