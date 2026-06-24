
import torch


class PnPFlow:
    def __init__(self, model, device, noise_type, lr_pnp, alpha,
                 num_samples, steps_pnp, gamma_style):
        self.model = model.to(device)
        self.device = device
        self.noise_type = noise_type
        self.lr_pnp = lr_pnp
        self.alpha = alpha
        self.num_samples = num_samples
        self.steps_pnp = steps_pnp
        self.gamma_style = gamma_style

    def model_forward(self, x, t):
        return self.model(x, t)

    def learning_rate_strat(self, lr, t):
        t = t.view(-1, 1, 1, 1)
        gamma_styles = {
            "1_minus_t": lambda lr_, t_: lr_ * (1 - t_),
            "sqrt_1_minus_t": lambda lr_, t_: lr_ * torch.sqrt(1 - t_),
            "constant": lambda lr_, t_: lr_,
            "alpha_1_minus_t": lambda lr_, t_: lr_ * (1 - t_)**self.alpha,
        }
        if self.gamma_style not in gamma_styles:
            raise ValueError(f"Unsupported gamma_style: {self.gamma_style}")
        return gamma_styles[self.gamma_style](lr, t)

    def grad_datafit(self, x, y, H, H_adj, sigma_noise):
        if self.noise_type == "gaussian":
            return H_adj(H(x) - y) / (sigma_noise**2)
        if self.noise_type == "laplace":
            return H_adj(
                2 * torch.heaviside(H(x)-y, torch.zeros_like(H(x))) - 1
            ) / sigma_noise
        raise ValueError("Noise type not supported")

    def interpolation_step(self, x, t):
        return t * x + torch.randn_like(x) * (1 - t)

    def denoiser(self, x, t):
        v = self.model_forward(x, t)
        return x + (1 - t.view(-1, 1, 1, 1)) * v

    @staticmethod
    def snapshot_iterations(steps):
        target_t_values = [0.0, 0.2, 0.4, 0.6, 0.8, 0.99]

        selected = []

        for target_t in target_t_values:
            iteration = int(round(target_t * steps))

            iteration = max(0, min(iteration, steps - 1))

            selected.append(iteration + 1)

        return selected

    def reconstruct(self, clean_img, degradation, sigma_noise, seed):
        H = degradation.H
        H_adj = degradation.H_adj

        if self.noise_type == "gaussian":
            torch.manual_seed(seed)
            noisy_img = H(clean_img.clone().to(self.device))
            noisy_img += torch.randn_like(noisy_img) * sigma_noise
            lr = sigma_noise**2 * self.lr_pnp
        elif self.noise_type == "laplace":
            torch.manual_seed(seed)
            noisy_img = H(clean_img.clone().to(self.device))
            distribution = torch.distributions.laplace.Laplace(
                torch.zeros_like(noisy_img),
                sigma_noise * torch.ones_like(noisy_img),
            )
            noisy_img += distribution.sample().to(self.device)
            lr = sigma_noise * self.lr_pnp
        else:
            raise ValueError("Noise type not supported")

        x = H_adj(torch.ones_like(noisy_img)).to(self.device)
        delta = 1 / self.steps_pnp
        wanted = set(self.snapshot_iterations(self.steps_pnp))
        snapshots = []

        with torch.no_grad():
            for iteration in range(int(self.steps_pnp)):
                t = torch.ones(len(x), device=self.device) * delta * iteration
                lr_t = self.learning_rate_strat(lr, t)

                # Gradient/reprojection step.
                z = x - lr_t * self.grad_datafit(
                    x, noisy_img, H, H_adj, sigma_noise
                )

                # Interpolation and denoising steps.
                x_new = torch.zeros_like(x)
                for _ in range(self.num_samples):
                    z_tilde = self.interpolation_step(
                        z, t.view(-1, 1, 1, 1)
                    )
                    x_new += self.denoiser(z_tilde, t)
                x = x_new / self.num_samples

                completed_iteration = iteration + 1
                if completed_iteration in wanted:
                    snapshots.append((
                        completed_iteration,
                        float(t[0].item()),
                        x.detach().clone(),
                    ))

        return noisy_img.detach(), x.detach(), snapshots
