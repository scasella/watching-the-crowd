"""Running the trainable tail (layers 34, 35, final norm, frozen lm_head) on cached layer-34 inputs.

The replay mirrors Qwen3Model.forward in transformers 5.18 exactly: the same causal-mask call,
the same rotary call on a [1, L] position grid, each decoder layer called with the same keyword
arguments, the norm over the whole sequence, and lm_head on the last position only
(logits_to_keep=1). Replay geometry matters: a crowded example is replayed inside a dense
[32, L] batch (target in row 0), a solo example as [1, L], so every GEMM in the tail sees the
same M as in deployment.

Library module (no CLI): imported by 02-07.
"""

import torch

from wcrowd_common import TAIL_START


def tail_modules(model):
    return [model.model.layers[TAIL_START], model.model.layers[TAIL_START + 1], model.model.norm]


def trainable_named_params(model):
    keep = (f"model.layers.{TAIL_START}.", f"model.layers.{TAIL_START + 1}.", "model.norm.")
    return {n: p for n, p in model.named_parameters() if n.startswith(keep)}


def tail_forward(model, h, modules=None, all_positions=False):
    """h: [B, L, H] bf16 residual stream entering layer 34. Returns logits [B, 1, V] (or the
    final-normed hidden [B, L, H] when all_positions=True, so a caller can choose rows)."""
    from transformers.masking_utils import create_causal_mask

    layer_a, layer_b, norm = modules or tail_modules(model)
    position_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    mask = create_causal_mask(
        config=model.config, inputs_embeds=h, attention_mask=None, past_key_values=None,
        position_ids=position_ids,
    )
    pe = model.model.rotary_emb(h, position_ids)
    for layer in (layer_a, layer_b):
        h = layer(h, attention_mask=mask, position_embeddings=pe, position_ids=position_ids,
                  past_key_values=None, use_cache=False)
        if isinstance(h, tuple):
            h = h[0]
    h = norm(h)
    if all_positions:
        return h
    return model.lm_head(h[:, -1:, :])


def crowd_batch(h_target, companions=None, batch=32):
    """Place a target's crowded layer-34 input in row 0 of a dense [batch, L, H] tensor.

    Companion rows do not affect row 0 (verified in Gate 1 and by the Phase 2 parity check), so
    by default they are copies of the target."""
    if companions is None:
        return h_target.unsqueeze(0).repeat(batch, 1, 1)
    return torch.cat([h_target.unsqueeze(0), companions[: batch - 1]], 0)


class Layer34Capture:
    """Read-only pre-hook: stores row `row` of the residual stream entering layer 34."""

    def __init__(self, model, rows=(0,)):
        self.rows = rows
        self.value = None
        self.handle = model.model.layers[TAIL_START].register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        self.value = args[0][list(self.rows)].detach().clone()

    def remove(self):
        self.handle.remove()


def log_odds(logits, pos_id, neg_id):
    lp = logits.double().log_softmax(-1)
    return lp[..., pos_id] - lp[..., neg_id]
