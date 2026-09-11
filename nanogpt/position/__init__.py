from .rope import RoPE


POSITIONS = {"rope": RoPE}


def build_position(head_dim, config):
    config = dict(config)
    return POSITIONS[config.pop("name")](head_dim, **config)
