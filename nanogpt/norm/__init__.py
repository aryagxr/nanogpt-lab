from .rmsnorm import RMSnorm


NORMS = {"rmsnorm": RMSnorm}


def build_norm(dim, config):
    config = dict(config)
    return NORMS[config.pop("name")](dim, **config)
