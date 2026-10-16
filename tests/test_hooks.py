"""SYNTHETIC CPU numerical fixtures, never evidence about LoopVL neurons.

The toy model has six uses of a shared two-layer L stack, two uses of a
different H stack, biased SwiGLU projections and a first-prediction readout.
"""

from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image
try:
    import torch
except ImportError:
    raise unittest.SkipTest("CPU hook fixtures require optional PyTorch; reports and data do not")

from reuseinterp.capture import ProductHooks, pool_tokens
from reuseinterp.data import generate_dataset
from reuseinterp.data.alignment import object_token_weights
from reuseinterp.interventions import ProductIntervention
from reuseinterp.pipeline import capture_example


class SyntheticMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 6, bias=True, dtype=torch.float64)
        self.up_proj = torch.nn.Linear(4, 6, bias=True, dtype=torch.float64)
        self.down_proj = torch.nn.Linear(6, 4, bias=True, dtype=torch.float64)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SyntheticBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(4, dtype=torch.float64)
        self.mlp = SyntheticMLP()

    def forward(self, x):
        return x + .1 * self.mlp(self.norm(x))


class SyntheticStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([SyntheticBlock(), SyntheticBlock()])

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x


class SyntheticRecurrent(torch.nn.Module):
    def __init__(self, cycles=2):
        super().__init__()
        self.L_module = SyntheticStack()
        self.H_module = SyntheticStack()
        self.cycles = cycles

    def forward(self, inputs_embeds):
        state = inputs_embeds
        for _ in range(self.cycles):
            for _ in range(3):
                state = self.L_module(state)
            state = self.H_module(state)
        return state


class SyntheticAdapter:
    def __init__(self):
        torch.manual_seed(41)
        self.recurrent = SyntheticRecurrent().eval()
        self.model = SimpleNamespace(config=SimpleNamespace(vision=SimpleNamespace(dynamic_resolution=True)))
        self.head = torch.nn.Linear(4, 9, dtype=torch.float64).eval()
        self.inputs = torch.randn(1, 25, 4, dtype=torch.float64)

    def prepare(self, image_path, prompt):
        # Two header +20 actual visual +3 instruction positions. No answer input.
        return {"grid_heights": torch.tensor([5]), "grid_widths": torch.tensor([4]),
                "merge_sizes": torch.tensor([1]), "visual_token_counts": torch.tensor([20]),
                "header_length": 2, "input_ids": torch.arange(25)[None]}

    def forward(self, batch):
        with torch.no_grad():
            return self.head(self.recurrent(self.inputs)[:, -1]).float()


class SyntheticHookContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.recurrent = SyntheticRecurrent().eval()
        self.inputs = torch.randn(1, 5, 4, dtype=torch.float64)
        self.channels = {"0": [0, 2], "1": [1, 3]}

    def run_forward(self, inputs=None):
        with torch.no_grad():
            return self.recurrent(self.inputs if inputs is None else inputs)

    def test_capture_is_exact_and_reconstructs_biased_swiglu(self):
        baseline = self.run_forward()
        observed = defaultdict(list)
        observers = []
        for layer, block in enumerate(self.recurrent.L_module.layers):
            def observe(module, args, layer=layer):
                observed[layer].append(args[0].detach().clone())
            observers.append(block.mlp.down_proj.register_forward_pre_hook(observe))
        try:
            with ProductHooks(self.recurrent, self.channels, "synthetic", [0, 2, 4], reconstruct=True) as hooks:
                captured = self.run_forward()
        finally:
            for handle in observers:
                handle.remove()
        torch.testing.assert_close(baseline, captured, rtol=0, atol=0)
        self.assertEqual(hooks.counts, {"L": 6, "H": 2})
        self.assertEqual(hooks.layer_counts, {0: 6, 1: 6})
        self.assertEqual(len(hooks.rows), 12)
        self.assertEqual(len(hooks.reconstruction), 24)
        self.assertTrue(all(row["max_abs"] == 0 for row in hooks.reconstruction))
        for row in hooks.rows:
            layer, invocation = row["physical_layer"], row["module_invocation"]
            expected = observed[layer][invocation - 1][0, [0, 2, 4]][:, self.channels[str(layer)]].float().numpy()
            np.testing.assert_array_equal(row["activations"], expected)
            self.assertEqual(row["outer_cycle"], (invocation - 1) // 3)
        for layer in (0, 1):
            identities = {(row["object_id"], row["storage_ptr"], row["hash"])
                          for row in hooks.parameter_visits if row["layer"] == layer}
            self.assertEqual(len(identities), 1)
        self.assertNotEqual(hooks.parameter_visits[0]["storage_ptr"], hooks.parameter_visits[1]["storage_ptr"])

    def test_every_generated_or_forced_prefix_forward_resets_counts(self):
        with ProductHooks(self.recurrent, self.channels, "synthetic_prefixes", [0, 2, 4]) as hooks:
            for prefix_length in range(3):
                continuation = torch.full((1, prefix_length, 4), .15, dtype=torch.float64)
                self.run_forward(torch.cat((self.inputs, continuation), dim=1))
        self.assertEqual(len(hooks.rows), 36)
        for forward in range(3):
            rows = [row for row in hooks.rows if row["forward_id"] == forward]
            self.assertEqual({row["decode_step"] for row in rows}, {forward})
            for layer in (0, 1):
                self.assertEqual([row["module_invocation"] for row in rows if row["physical_layer"] == layer], list(range(1, 7)))
        self.assertEqual(hooks.counts, {"L": 6, "H": 2})

    def test_self_replacement_and_zero_dose_are_exact(self):
        baseline = self.run_forward()
        for self_replace in (False, True):
            intervention = ProductIntervention(1, 4, 2, (1, 3), dose=0, self_replace=self_replace)
            with ProductHooks(self.recurrent, self.channels, "synthetic_noop", [0, 2, 4], intervention) as hooks:
                actual = self.run_forward()
            torch.testing.assert_close(baseline, actual, rtol=0, atol=0)
            self.assertEqual(hooks.changed, [{"forward_id": 0, "layer": 1, "invocation": 4, "indices": []}])

    def test_positive_and_negative_doses_are_local_at_application(self):
        for dose in (-.25, .25):
            before, after = defaultdict(list), defaultdict(list)
            handles = []
            for layer, block in enumerate(self.recurrent.L_module.layers):
                def record_before(module, args, layer=layer):
                    before[layer].append(args[0].detach().clone())
                handles.append(block.mlp.down_proj.register_forward_pre_hook(record_before))
            edit = ProductIntervention(1, 4, 2, (1, 3), dose=dose)
            try:
                with ProductHooks(self.recurrent, self.channels, "synthetic_dose", [0, 2, 4], edit) as hooks:
                    # Registered after intervention: observes the actual tensor
                    # entering each projection, including its local replacement.
                    for layer, block in enumerate(self.recurrent.L_module.layers):
                        def record_after(module, args, layer=layer):
                            after[layer].append(args[0].detach().clone())
                        handles.append(block.mlp.down_proj.register_forward_pre_hook(record_after))
                    self.run_forward()
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(len(hooks.changed), 1)
            self.assertEqual(hooks.changed[0]["indices"], [[0, 1, 2], [0, 3, 2]])
            for layer in (0, 1):
                for invocation in range(6):
                    expected = torch.zeros_like(before[layer][invocation])
                    if layer == 1 and invocation == 3:
                        expected[0, [1, 3], 2] = dose
                    torch.testing.assert_close(after[layer][invocation] - before[layer][invocation], expected, rtol=0, atol=2e-16)
            # Later recurrence can change; locality concerns the edit operator.

    def test_incomplete_forward_fails_and_removes_hooks(self):
        self.recurrent.cycles = 1
        with self.assertRaisesRegex(ValueError, "Incomplete recurrence"):
            with ProductHooks(self.recurrent, self.channels, "synthetic_bad", [0]) as hooks:
                self.run_forward()
        self.assertTrue(all(not block.mlp.down_proj._forward_pre_hooks for block in self.recurrent.L_module.layers))
        self.assertFalse(self.recurrent._forward_pre_hooks)
        self.recurrent.cycles = 2
        self.run_forward()

    def test_capture_pipeline_saves_raw_tokens_pools_and_area_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "data"
            generate_dataset(dataset, {"smoke": 1}, image_size=96)
            example = json.loads((dataset / "examples.jsonl").read_text().splitlines()[0])
            adapter = SyntheticAdapter()
            artifact = Path(tmp) / "synthetic_capture.npz"
            row = capture_example(adapter, example, dataset, self.channels, [0, 1], artifact, save_raw=True)
            weights = object_token_weights(dataset / example["mask_path"], 5, 4, source_image_size=(96, 96))
            self.assertEqual(row["mask_alignment"], weights.coverage)
            self.assertEqual(row["mask_alignment"]["source_image_size_width_height"], [96, 96])
            with np.load(artifact, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["token_indices"], list(range(2, 22)) + [24])
                for layer in self.channels:
                    raw = arrays["diagnostic_real_tokens_layer_" + layer]
                    pooled = arrays["products_layer_" + layer]
                    self.assertEqual(raw.shape, (6, 21, 2))
                    self.assertEqual(pooled.shape, (6, len(weights) + 1, 2))
                    expected = np.stack([np.stack([pool_tokens(v[:-1], weights[label]) for label in sorted(weights)] + [v[-1]]) for v in raw])
                    np.testing.assert_allclose(pooled, expected, rtol=0, atol=0)
                np.testing.assert_allclose(arrays["first_prediction_logits"], adapter.forward(None).numpy(), rtol=0, atol=0)
                self.assertEqual(arrays["states_layer_0_input"].shape, (6, len(weights) + 1, 4))
            adapter.model.config.vision.dynamic_resolution = False
            with self.assertRaisesRegex(ValueError, "letterbox"):
                capture_example(adapter, example, dataset, self.channels, [], artifact)
            adapter.model.config.vision.dynamic_resolution = True
            Image.new("L", (48, 96), 1).save(dataset / example["mask_path"])
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                capture_example(adapter, example, dataset, self.channels, [], artifact)


if __name__ == "__main__":
    unittest.main()
