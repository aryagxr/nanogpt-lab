class FlatCooldown:
    def __init__(self, optimizers, train_steps, cooldown_fraction):
        self.optimizers = optimizers
        self.train_steps = train_steps
        self.cooldown_fraction = cooldown_fraction
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["initial_lr"] = group["lr"]

    def step(self, step):
        progress = step / self.train_steps
        assert 0 <= progress < 1
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                cooldown_frac = self.cooldown_fraction[group["name"]]
                if progress < 1 - cooldown_frac:
                    eta = 1.0
                else:
                    eta = (1 - progress) / cooldown_frac
                group["lr"] = group["initial_lr"] * eta
