from .standard import StandardResidual


RESIDUALS = {"standard": StandardResidual}


def build_residual(dim, num_layers, config):
    config = dict(config)
    return RESIDUALS[config.pop("name")](dim, num_layers, **config)
