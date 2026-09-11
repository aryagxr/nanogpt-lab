from .flat_cooldown import FlatCooldown


SCHEDULES = {"flat_cooldown": FlatCooldown}


def build_schedule(optimizers, train_steps, config):
    config = dict(config)
    return SCHEDULES[config.pop("name")](optimizers, train_steps, **config)
