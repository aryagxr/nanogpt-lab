from .sdpa import CausalSelfAttention


ATTENTIONS = {"sdpa": CausalSelfAttention}


def build_attention(dim, config, position, linear):
    config = dict(config)
    return ATTENTIONS[config.pop("name")](dim, linear=linear, position=position, **config)
