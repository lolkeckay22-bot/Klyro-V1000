import unittest
import os
import tempfile
import torch
import torch.nn as nn
from src.checkpoint import save_checkpoint, load_latest_checkpoint, get_all_checkpoints
from src.model import KlyroConfig, KlyroForCausalLM

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    def forward(self, x):
        return self.linear(x)

class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model = DummyModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5)
        self.config = {"dummy": 1}

    def test_save_and_load(self):
        path = save_checkpoint(self.model, self.optimizer, self.scheduler, step=100, config=self.config, checkpoint_dir=self.temp_dir)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "latest_checkpoint.pt")))
        loaded = load_latest_checkpoint(self.temp_dir, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler)
        self.assertIsNotNone(loaded)
        model, opt, sched, step, config = loaded
        self.assertEqual(step, 100)
        self.assertEqual(config["dummy"], 1)

    def test_get_all_checkpoints(self):
        save_checkpoint(self.model, self.optimizer, self.scheduler, step=1, config=self.config, checkpoint_dir=self.temp_dir)
        save_checkpoint(self.model, self.optimizer, self.scheduler, step=5, config=self.config, checkpoint_dir=self.temp_dir)
        save_checkpoint(self.model, self.optimizer, self.scheduler, step=12, config=self.config, checkpoint_dir=self.temp_dir)
        checkpoints = get_all_checkpoints(self.temp_dir)
        self.assertEqual(len(checkpoints), 3)
        # Ensure sorted by step
        steps = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in checkpoints]
        self.assertEqual(steps, [1, 5, 12])

if __name__ == '__main__':
    unittest.main()