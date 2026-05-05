"""DDIM sampler for LatentSync."""

import mlx.core as mx


class DDIMSampler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
    ):
        if beta_schedule == "scaled_linear":
            betas = mx.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        elif beta_schedule == "linear":
            betas = mx.linspace(beta_start, beta_end, num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        self.alphas_cumprod = mx.cumprod(alphas)
        self.num_train_timesteps = num_train_timesteps

        self._timesteps = None

    def set_timesteps(self, num_inference_steps: int):
        step_ratio = self.num_train_timesteps // num_inference_steps
        self._timesteps = list(range(0, self.num_train_timesteps, step_ratio))[::-1]

    @property
    def timesteps(self):
        return self._timesteps

    def scale_model_input(self, sample, timestep):
        return sample

    @property
    def init_noise_sigma(self):
        return 1.0

    def step(self, noise_pred, timestep, sample, eta=0.0):
        prev_timestep = timestep - self.num_train_timesteps // len(self._timesteps)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[max(prev_timestep, 0)] if prev_timestep >= 0 else mx.array(1.0)

        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - mx.sqrt(beta_prod_t) * noise_pred) / mx.sqrt(alpha_prod_t)

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        if eta > 0:
            std_dev_t = eta * mx.sqrt(variance)
        else:
            std_dev_t = mx.array(0.0)

        pred_sample_direction = mx.sqrt(1 - alpha_prod_t_prev - std_dev_t ** 2) * noise_pred
        prev_sample = mx.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        if eta > 0:
            noise = mx.random.normal(prev_sample.shape)
            prev_sample = prev_sample + std_dev_t * noise

        return prev_sample
