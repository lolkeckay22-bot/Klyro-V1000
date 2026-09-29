import unittest
import torch
from src.model import KlyroConfig, KlyroForCausalLM

class TestModel(unittest.TestCase):
    def test_model_creation(self):
        config = KlyroConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256, max_position_embeddings=128)
        model = KlyroForCausalLM(config)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertGreater(model.num_parameters(), 0)

    def test_forward_pass(self):
        config = KlyroConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256, max_position_embeddings=128)
        model = KlyroForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        outputs = model(input_ids)
        self.assertIn('logits', outputs)
        self.assertEqual(outputs['logits'].shape, (2, 10, config.vocab_size))

if __name__ == '__main__':
    unittest.main()