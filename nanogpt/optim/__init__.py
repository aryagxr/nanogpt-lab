from .muonh import build_optimizers as build_muonh


OPTIMIZERS = {"muonh": build_muonh}


def build_optimizers(model, config):
    optimizers = OPTIMIZERS[config["name"]](model, config)
    params = [p for opt in optimizers for group in opt.param_groups for p in group["params"]]
    assert len(params) == len(set(params))
    assert set(params) == set(model.parameters())
    return optimizers
