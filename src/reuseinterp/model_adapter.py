"""Explicit official LoopVL loader. Imported only by scheduled GPU commands.

Paths below are verified against the saved upstream source; runtime structural
checks precede hook installation. Loaded-weight checks remain GPU work.
"""
import importlib.metadata
import inspect
import os
from pathlib import Path
import subprocess
import sys
from .io import file_hash, digest

CODE_REVISION = "6c14bed1d62fbecb559d5902459c6093c38da405"
CHECKPOINT_REVISION = "92aa28bc678f523f8641fdb2f9e5159f614d1c02"


def tensor_hash(tensor):
    import torch
    return __import__("hashlib").sha256(
        tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


class LoopVLAdapter:
    def __init__(self, config):
        import torch
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
            raise RuntimeError("Model inference requires a scheduled Slurm CUDA allocation")
        self.device = torch.device("cuda:0")  # index within Slurm's CUDA_VISIBLE_DEVICES
        self.root = Path(os.environ.get("REUSE_LOOPVL_DIR", config.get("loopvl_dir", "vendor/LoopVL"))).resolve()
        revision = subprocess.check_output(["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True).strip()
        if revision != config.get("code_revision", CODE_REVISION):
            raise ValueError("Official source revision mismatch")
        tracked = subprocess.check_output(["git", "-C", str(self.root), "diff", "HEAD", "--", "runtime"], text=True)
        if tracked:
            raise ValueError("Official runtime has local changes")
        self.model_dir = Path(os.environ.get("REUSE_MODEL_DIR", config.get("model_dir", "vendor/LoopVL/model"))).resolve()
        required = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "generation_config.json"]
        if not all((self.model_dir / f).is_file() for f in required):
            raise FileNotFoundError("Complete pinned checkpoint must be staged before GPU jobs")
        expected_hashes = config.get("checkpoint_hashes", {})
        if not expected_hashes or "model.safetensors" not in expected_hashes:
            raise ValueError("Configuration must pin checkpoint byte hashes before loading")
        self.checkpoint_hashes = {f: file_hash(self.model_dir / f) for f in required}
        for name, expected in expected_hashes.items():
            if file_hash(self.model_dir / name) != expected:
                raise ValueError("Checkpoint checksum mismatch: " + name)
        sys.path.insert(0, str(self.root / "runtime"))
        from modeling_loopvl import LoopVLForConditionalGeneration, load_tokenizer_and_processor
        self.model = LoopVLForConditionalGeneration.from_pretrained(self.model_dir, device=self.device)
        self.tokenizer, self.processor = load_tokenizer_and_processor(self.model_dir, self.model.config)
        self.recurrent = self.model.recurrent_model
        self.audit = self.audit_loaded()

    def audit_loaded(self):
        import torch
        r = self.recurrent
        if (r.config.H_cycles, r.config.L_cycles) != (2, 3):
            raise ValueError("Unvalidated recurrence schedule")
        if len(r.L_module.layers) != 16 or len(r.H_module.layers) != 16:
            raise ValueError("Unvalidated physical stack depth")
        sites = []
        for kind in ("L", "H"):
            for layer, block in enumerate(getattr(r, kind + "_module").layers):
                mlp = block.mlp
                for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
                    if not hasattr(mlp, name):
                        raise ValueError("Loaded MLP contract differs from audited source")
                if not isinstance(mlp.down_proj, torch.nn.Linear):
                    raise ValueError("Post-product site is not an audited linear projection")
                parameters = {n: {"object_id": id(p), "storage_ptr": p.untyped_storage().data_ptr(),
                                  "hash": tensor_hash(p), "shape": list(p.shape), "dtype": str(p.dtype)}
                              for n, p in mlp.named_parameters()}
                sites.append({"module_kind": kind, "physical_layer": layer,
                              "hook_name": "mlp.down_proj.input", "parameters": parameters,
                              "mlp_forward_source": inspect.getsource(type(mlp).forward)})
        return {"status": "loaded_structure_verified", "schedule": ["L", "L", "L", "H", "L", "L", "L", "H"],
                "code_revision": CODE_REVISION, "checkpoint_revision": CHECKPOINT_REVISION,
                "checkpoint_hashes": self.checkpoint_hashes, "sites": sites,
                "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "Pillow")},
                "precision": {name: str(next(module.parameters()).dtype) for name, module in
                              [("language", self.model.language_model), ("projector", self.model.projector),
                               ("visual_gate", r.visual_gate), ("visual_anchor", r.visual_anchor)]}}

    def prepare(self, image_path, question):
        from PIL import Image
        from hrm_penguin.inference import build_inference_batch
        # API receives a question, never a reference answer or task program.
        with Image.open(image_path) as image:
            return build_inference_batch(self.tokenizer, self.processor, image.convert("RGB"),
                                         question, self.model.config, self.device)

    def forward(self, batch, continuation=()):
        """Official recomputation path at prompt or forced answer-prefix only."""
        import torch
        from hrm_penguin.recurrent.positions import build_compact_positions
        ids = batch["input_ids"]
        if continuation:
            ids = torch.cat([ids, torch.tensor([continuation], device=ids.device)], dim=1)
        header, instruction = int(batch["header_length"]), int(batch["instruction_length"])
        count = int(batch["visual_token_counts"][0])
        positions, rope, visual, mask = build_compact_positions(
            torch.tensor([header], device=ids.device), torch.tensor([instruction], device=ids.device),
            torch.tensor([len(continuation)], device=ids.device), visual_token_counts=batch["visual_token_counts"],
            grid_heights=batch["grid_heights"], grid_widths=batch["grid_widths"],
            grid_h=self.model.config.vision.grid_height, grid_w=self.model.config.vision.grid_width,
            rope_mode=self.model.config.recurrent_visual.rope_mode)
        embeds = self.model.language_model.get_input_embeddings()(ids)
        visual_embeds = self.model.encode_images(batch["pixel_values"], batch["grid_sizes"], batch["merge_sizes"])
        embeds = self.model._replace_visual_tokens(embeds, visual_embeds, visual)
        types = torch.zeros_like(ids)
        types[:, :header + count + instruction] = 1
        with torch.no_grad():
            state = self.recurrent(inputs_embeds=embeds, attention_mask=torch.ones_like(ids),
                                   token_type_ids=types, position_ids=positions, rope_position_ids=rope,
                                   visual_mask=visual, instruction_mask=mask, use_cache=False)
            return self.model.language_model.lm_head(state.last_hidden_state[:, -1]).float()

    def generate_official(self, batch, budget=8):
        from hrm_penguin.generation import greedy_generate
        return greedy_generate(self.model, batch, max_new_tokens=budget,
                               eos_token_id=self.tokenizer.eos_token_id)[0].tolist()

    def official_with_logits(self, batch, budget=8):
        """Independent official entry point; record its actual first logits."""
        first = []
        def capture(module, args, output):
            if not first:
                first.append(output.detach().float().cpu().numpy())
        handle = self.model.language_model.lm_head.register_forward_hook(capture)
        try:
            answer = self.generate_official(batch, budget)
        finally:
            handle.remove()
        if len(first) != 1:
            raise ValueError("Official output head did not fire")
        return answer, first[0]

    def generate(self, batch, budget=8):
        result = []
        for _ in range(budget):
            token = int(self.forward(batch, result).argmax(-1).item())
            result.append(token)
            if token == self.tokenizer.eos_token_id:
                break
        return result

    def score(self, batch, candidates):
        """Full conditional answer-string scores; no answer in first-prediction capture."""
        scores = {}
        for answer in candidates:
            tokens = self.tokenizer.encode(answer, add_special_tokens=False)
            if not tokens:
                raise ValueError("Empty candidate tokenization")
            total = 0.0
            for step, token in enumerate(tokens):
                total += float(self.forward(batch, tokens[:step]).log_softmax(-1)[0, token])
            scores[answer] = total
        return scores
