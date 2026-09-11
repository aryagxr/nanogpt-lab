from .sdpa import CausalSelfAttention


ATTENTIONS = {"sdpa": CausalSelfAttention}


def build_attention(dim, config, linear):
    config = dict(config)
    return ATTENTIONS[config.pop("name")](dim, linear=linear, **config)
