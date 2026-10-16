"""Validated invocation counters and physical post-product hooks.

This module doesn't import torch until hooks are used. Captures selected real
token activations before pooling; a pooled vector is never an intervention site.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .model_adapter import tensor_hash


@dataclass(frozen=True)
class Trace:
    forward_id: int
    decode_step: int
    module_kind: str
    module_invocation: int
    outer_cycle: int
    physical_layer: int
    hook_name: str
    neuron_index: int
    token_indices: tuple
    token_role: str
    example_id: str


def sample_neurons(dimensions, seed=1729, per_layer=32):
    rng = np.random.default_rng(seed)
    return {str(layer): sorted(rng.choice(width, per_layer, replace=False).tolist())
            for layer, width in enumerate(dimensions)}


class ProductHooks:
    def __init__(self, recurrent, channels, example_id, tokens, intervention=None, reconstruct=False):
        self.r = recurrent
        self.channels = channels
        self.example_id = example_id
        self.tokens = list(tokens)
        self.intervention = intervention
        self.reconstruct = reconstruct
        self.handles, self.rows, self.parameter_visits = [], [], []
        self.forward_id, self.decode_step = -1, -1
        self.counts = {"L": 0, "H": 0}
        self.layer_counts = {}
        self.changed = []
        self.reconstruction = []
        self.last_product = {}
        self.last_input = {}

    def __enter__(self):
        import torch
        def reset(module, inputs, kwargs):
            if self.forward_id >= 0:
                self.assert_counts()
            self.forward_id += 1
            self.decode_step += 1
            self.counts = {"L": 0, "H": 0}
            self.layer_counts = {}
        self.handles.append(self.r.register_forward_pre_hook(reset, with_kwargs=True))
        for kind in ("L", "H"):
            stack = getattr(self.r, kind + "_module")
            def stack_hook(module, inputs, kind=kind):
                self.counts[kind] += 1
            self.handles.append(stack.register_forward_pre_hook(stack_hook))
        for layer_key, indices in self.channels.items():
            layer = int(layer_key)
            mlp = self.r.L_module.layers[layer].mlp
            if self.reconstruct:
                def mlp_in(module, args, layer=layer):
                    self.last_input[layer] = args[0].detach()
                self.handles.append(mlp.register_forward_pre_hook(mlp_in))
            def product(module, args, layer=layer, indices=indices, mlp=mlp):
                activation = args[0]
                invocation = self.counts["L"]
                if not 1 <= invocation <= 6 or activation.ndim != 3 or activation.shape[0] != 1:
                    raise ValueError("Unsupported invocation/batch shape")
                self.layer_counts[layer] = self.layer_counts.get(layer, 0) + 1
                if max(self.tokens) >= activation.shape[1] or min(self.tokens) < 0:
                    raise ValueError("Token alignment outside current sequence")
                if self.reconstruct:
                    recomputed = mlp.act_fn(mlp.gate_proj(self.last_input[layer])) * mlp.up_proj(self.last_input[layer])
                    err = float((recomputed - activation).abs().max())
                    self.reconstruction.append({"kind": "product", "layer": layer, "invocation": invocation, "max_abs": err})
                    self.last_product[layer] = activation.detach()
                self.parameter_visits.append({"layer": layer, "invocation": invocation,
                                              "object_id": id(module.weight),
                                              "storage_ptr": module.weight.untyped_storage().data_ptr(),
                                              "hash": tensor_hash(module.weight) if self.reconstruct else None,
                                              "mlp_parameters": {n: {"object_id": id(p),
                                                                      "storage_ptr": p.untyped_storage().data_ptr(),
                                                                      "hash": tensor_hash(p) if self.reconstruct else None}
                                                                 for n, p in mlp.named_parameters()}})
                selected = activation[0, self.tokens][:, indices].detach().float().cpu().numpy()
                self.rows.append({"example_id": self.example_id, "forward_id": self.forward_id,
                                  "decode_step": self.decode_step, "module_kind": "L", "module_invocation": invocation,
                                  "outer_cycle": (invocation - 1) // 3, "physical_layer": layer,
                                  "hook_name": "mlp.down_proj.input", "neuron_indices": list(indices),
                                  "token_indices": self.tokens, "token_role": "explicit_positions_before_stratum_pooling",
                                  "activations": selected.tolist()})
                if self.intervention and self.intervention.matches(layer, invocation):
                    replacement, changed = self.intervention.apply(activation)
                    self.changed.append({"forward_id": self.forward_id, "layer": layer,
                                         "invocation": invocation, "indices": changed})
                    return (replacement,) + args[1:]
            self.handles.append(mlp.down_proj.register_forward_pre_hook(product))
            if self.reconstruct:
                def projected(module, args, out, layer=layer):
                    expected = torch.nn.functional.linear(self.last_product.pop(layer), module.weight, module.bias)
                    self.reconstruction.append({"kind": "projection", "layer": layer,
                                                "invocation": self.counts["L"], "max_abs": float((expected - out).abs().max())})
                    self.last_input.pop(layer, None)
                self.handles.append(mlp.down_proj.register_forward_hook(projected))
        return self

    def assert_counts(self):
        if self.counts != {"L": 6, "H": 2} or any(self.layer_counts.get(int(k)) != 6 for k in self.channels):
            raise ValueError("Incomplete recurrence on forward: " + str(self.counts))

    def __exit__(self, exc_type, *args):
        try:
            if exc_type is None and self.forward_id >= 0:
                self.assert_counts()
        finally:
            for handle in self.handles:
                handle.remove()


def pool_tokens(values, weights):
    values, weights = np.asarray(values), np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) != len(values) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid object pooling mask")
    return (values * (weights / weights.sum())[:, None]).sum(axis=0)
