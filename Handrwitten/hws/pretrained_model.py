"""Canonical Alex Graves (2013) Peephole LSTM Pretrained Handwriting Synthesis Model.

Provides full inference, tokenizer, and coordinate un-normalization/baseline alignment
for the reference Graves 2013 baseline model trained on the IAM Online Handwriting Database.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hws.preprocess import normalize_scale, offsets_to_coords

logger = logging.getLogger(__name__)


# ── Canonical Graves Architecture (Peephole LSTM + SoftWindow + MDN) ──────────


class PeepholeLSTM(jit.ScriptModule):
    """LSTM with peephole connections as specified in Graves (2013)."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ci = nn.Parameter(torch.Tensor(hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.W_cf = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.W_co = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.W_xh = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size * 4))
        self.init_weights()

    def get_initial_state(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        h0 = torch.zeros(batch_size, self.hidden_size)
        c0 = torch.zeros_like(h0)
        return h0, c0

    def init_weights(self):
        sd = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-sd, sd)

    @jit.script_method
    def forward(self, x: Tensor, states: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        bs, seq_sz, _ = x.size()
        hidden_seq = torch.jit.annotate(List[Tensor], [])

        h_t, c_t = states

        for t in range(seq_sz):
            x_t = x[:, t, :]
            x_with_h = torch.cat([x_t, h_t], dim=1)
            z = x_with_h @ self.W_xh

            z_i, z_f, z_c, z_o = z.chunk(4, 1)

            i_t = torch.sigmoid(z_i + c_t * self.W_ci + self.b_i)
            f_t = torch.sigmoid(z_f + c_t * self.W_cf + self.b_f)
            c_t = f_t * c_t + i_t * torch.tanh(z_c + self.b_c)
            o_t = torch.sigmoid(z_o + c_t * self.W_co + self.b_o)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t)

        hidden_seq = torch.stack(hidden_seq, dim=1)
        return hidden_seq, (h_t, c_t)


class SoftWindow(jit.ScriptModule):
    """Monotonic Gaussian soft-window attention."""

    def __init__(self, input_size: int, num_components: int):
        super().__init__()
        self.alpha = nn.Linear(input_size, num_components)
        self.beta = nn.Linear(input_size, num_components)
        self.k = nn.Linear(input_size, num_components)

    @jit.script_method
    def forward(self, x: Tensor, c: Tensor, prev_k: Tensor):
        x = x[:, 0]
        alpha = torch.exp(self.alpha(x))
        beta = torch.exp(self.beta(x))
        k_new = prev_k + torch.exp(self.k(x))

        batch_size, num_chars, _ = c.shape
        alpha = alpha.unsqueeze(2).repeat(1, 1, num_chars)
        beta = beta.unsqueeze(2).repeat(1, 1, num_chars)
        k = k_new.unsqueeze(2).repeat(1, 1, num_chars)
        u = torch.arange(num_chars, device=alpha.device)

        densities = alpha * torch.exp(-beta * (k - u) ** 2)
        phi = densities.sum(dim=1).unsqueeze(1)
        return phi, k_new

    @staticmethod
    def matmul_3d(phi: Tensor, c: Tensor) -> Tensor:
        return torch.bmm(phi, c)


class MixtureDensityLayer(nn.Module):
    """Bivariate Gaussian Mixture Density Network."""

    def __init__(self, input_size: int, num_components: int, bias: float | None = None):
        super().__init__()
        self.num_components = num_components
        self.pi = nn.Linear(input_size, num_components)
        self.mu = nn.Linear(input_size, num_components * 2)
        self.sd = nn.Linear(input_size, num_components * 2)
        self.ro = nn.Linear(input_size, num_components)
        self.eos = nn.Linear(input_size, 1)
        self.bias = bias

    def forward(self, x: Tensor):
        pi_hat = self.pi(x)
        sd_hat = self.sd(x)

        if self.bias is None or self.bias == 0:
            pi = F.softmax(pi_hat, dim=-1)
            sd = torch.exp(sd_hat)
        else:
            pi = F.softmax(pi_hat * (1 + self.bias), dim=-1)
            sd = torch.exp(sd_hat - self.bias)

        mu = self.mu(x)
        ro = torch.tanh(self.ro(x))
        eos = torch.sigmoid(self.eos(x))
        return pi, mu, sd, ro, eos


def sample_from_bivariate_mixture(mu1: Tensor, mu2: Tensor, sd1: Tensor, sd2: Tensor, ro: Tensor) -> tuple[float, float]:
    cov_x_y = ro * sd1 * sd2
    sigma = torch.tensor([[sd1 ** 2, cov_x_y], [cov_x_y, sd2 ** 2]])
    loc = torch.tensor([mu1.item(), mu2.item()])
    gmm = torch.distributions.MultivariateNormal(loc, sigma)
    v = gmm.sample()
    return float(v[0].item()), float(v[1].item())


def get_mean_prediction(output, device: torch.device, stochastic: bool) -> Tensor:
    pi, mu, sd, ro, eos = output
    num_components = len(pi)

    if stochastic:
        component = torch.multinomial(pi, 1).item()
    else:
        component = pi.cpu().argmax().item()

    mu1 = mu[component]
    mu2 = mu[component + num_components]
    sd1 = sd[component]
    sd2 = sd[component + num_components]
    component_ro = ro[component]

    x, y = mu1.item(), mu2.item()
    if stochastic:
        try:
            x, y = sample_from_bivariate_mixture(mu1, mu2, sd1, sd2, component_ro)
        except Exception:
            pass

    eos_flag = 1.0 if eos.item() > 0.5 else 0.0
    return torch.tensor([x, y, eos_flag], device=device)


class SynthesisNetwork(jit.ScriptModule):
    """Alex Graves (2013) 3-layer LSTM with SoftWindow attention."""

    @classmethod
    def get_default_model(cls, alphabet_size: int, device: torch.device, bias: float | None = None):
        return cls(3, 400, alphabet_size, device, bias=bias)

    def __init__(self, input_size: int, hidden_size: int, alphabet_size: int, device: torch.device,
                 gaussian_components: int = 10, output_mixtures: int = 20, bias: float | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.alphabet_size = alphabet_size
        self.device = device
        self.gaussian_components = gaussian_components

        self.lstm1 = PeepholeLSTM(input_size + alphabet_size, hidden_size)
        self.window = SoftWindow(hidden_size, gaussian_components)
        self.lstm2 = PeepholeLSTM(input_size + hidden_size + alphabet_size, hidden_size)
        self.lstm3 = PeepholeLSTM(input_size + hidden_size + alphabet_size, hidden_size)
        self.mixture = MixtureDensityLayer(hidden_size * 3, output_mixtures, bias)

    def get_initial_states(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        h0 = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros_like(h0)
        return h0, c0

    def get_all_initial_states(self, batch_size: int):
        return (
            self.get_initial_states(batch_size),
            self.get_initial_states(batch_size),
            self.get_initial_states(batch_size),
        )

    def get_initial_input(self) -> Tensor:
        return torch.zeros(1, 3, device=self.device)

    def get_initial_window(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, 1, self.alphabet_size, device=self.device)

    def compute_mixture(self, x: Tensor, h1: Tensor, w1: Tensor,
                        hidden2: Tuple[Tensor, Tensor], hidden3: Tuple[Tensor, Tensor]):
        inputs = torch.cat([x, h1, w1], dim=-1)
        h2, hidden2 = self.lstm2(inputs, hidden2)
        inputs = torch.cat([x, h2, w1], dim=-1)
        h3, hidden3 = self.lstm3(inputs, hidden3)
        inputs = torch.cat([h1, h2, h3], dim=-1)
        return self.mixture(inputs), hidden2, hidden3

    def sample_means_with_attention(self, context: Tensor, steps: int = 1200, stochastic: bool = True):
        c = context.to(self.device)
        batch_size, u, _ = c.shape
        assert batch_size == 1

        x = self.get_initial_input().unsqueeze(0)
        w = self.get_initial_window(batch_size)
        k = torch.zeros(batch_size, self.gaussian_components, device=self.device, dtype=torch.float32)
        hidden1, hidden2, hidden3 = self.get_all_initial_states(batch_size)

        outputs = []
        attention_weights = []

        for t in range(steps):
            x_with_w = torch.cat([x, w], dim=-1)
            h1, hidden1 = self.lstm1(x_with_w, hidden1)

            phi, k = self.window(h1, c, k)
            attention_weights.append(phi.squeeze(0))

            w = self.window.matmul_3d(phi, c)
            mixture, hidden2, hidden3 = self.compute_mixture(x, h1, w, hidden2, hidden3)

            pi, mu, sd, ro, eos = mixture
            single_step = (pi[0, 0], mu[0, 0], sd[0, 0], ro[0, 0], eos[0, 0])
            x_temp = get_mean_prediction(single_step, self.device, stochastic=stochastic).unsqueeze(0)

            # End of string check
            last_phi = phi[0, 0, u - 1].item()
            is_eos = x_temp[0, 2].item() > 0.5
            if last_phi > 0.8 or (phi[0, 0].argmax().item() == u - 1 and is_eos):
                x_temp[0, 2] = 1.0
                outputs.append(x_temp)
                break

            outputs.append(x_temp)
            x = x_temp.unsqueeze(0)

        return torch.cat(outputs, dim=0), torch.cat(attention_weights, dim=0)


# ── Tokenizer matching IAM Checkpoint ─────────────────────────────────────────


class IAMTokenizer:
    def __init__(self, charset: str):
        self._charset = str(charset)
        self._chr_to_int = {ch: i + 1 for i, ch in enumerate(self._charset)}

    @property
    def size(self) -> int:
        return len(self._charset) + 1

    def tokenize(self, s: str) -> list[int]:
        return [self._chr_to_int.get(ch, 0) for ch in s]

    def encode(self, text: str) -> torch.Tensor:
        eye = torch.eye(self.size)
        tokens = self.tokenize(text)
        return eye[tokens].unsqueeze(0)


# ── High-Level Synthesis Wrapper ──────────────────────────────────────────────


class PretrainedGravesSynthesis:
    """End-to-end synthesizer wrapping the canonical Graves 2013 checkpoint."""

    def __init__(self, model_dir: Path | str, device: torch.device | None = None, bias: float = 0.0):
        model_dir = Path(model_dir)
        meta_file = model_dir / "meta.json" if (model_dir / "meta.json").is_file() else model_dir.parent / "meta.json"
        model_file = model_dir / "model.pt" if (model_dir / "model.pt").is_file() else model_dir

        if not meta_file.is_file():
            raise FileNotFoundError(f"meta.json not found in {model_dir}")
        if not model_file.is_file():
            raise FileNotFoundError(f"model.pt not found in {model_dir}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.mu = torch.tensor(meta["mu"], dtype=torch.float32)
        self.sd = torch.tensor(meta["std"], dtype=torch.float32)
        self.tokenizer = IAMTokenizer(meta["charset"])

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.bias = bias

        self.model = SynthesisNetwork.get_default_model(self.tokenizer.size, device, bias=bias).to(device)
        state_dict = torch.load(model_file, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def generate(
        self,
        text: str,
        bias: float | None = None,
        max_steps: int = 1200,
        stochastic: bool = True,
        seed: int | None = None,
        temp: float = 0.35,
        max_search_seeds: int = 12,
    ) -> np.ndarray:
        """Synthesize text into scaled, baseline-anchored, smooth (N, 4) stroke coordinates."""
        from hws.preprocess import denoise

        if not text.strip():
            return np.zeros((0, 4), dtype=np.float32)

        # Encode with trailing sentinel space
        clean_text = text.strip()
        c = self.tokenizer.encode(clean_text + " ").to(self.device)
        u = c.shape[1]

        seeds_to_try = [seed] if seed is not None else list(range(max_search_seeds))
        best_coords = None
        best_score = -1

        min_expected_pts = max(80, len(clean_text) * 10)
        max_expected_pts = max(500, len(clean_text) * 38)

        for s in seeds_to_try:
            torch.manual_seed(s)
            np.random.seed(s)

            x = self.model.get_initial_input().unsqueeze(0)
            w = self.model.get_initial_window(1)
            k = torch.zeros(1, self.model.gaussian_components, device=self.device, dtype=torch.float32)
            hidden1, hidden2, hidden3 = self.model.get_all_initial_states(1)

            outputs = []
            char_counts = np.zeros(u, dtype=int)

            with torch.no_grad():
                for t in range(max_steps):
                    x_with_w = torch.cat([x, w], dim=-1)
                    h1, hidden1 = self.model.lstm1(x_with_w, hidden1)
                    phi, k = self.model.window(h1, c, k)
                    w = self.model.window.matmul_3d(phi, c)
                    mixture, hidden2, hidden3 = self.model.compute_mixture(x, h1, w, hidden2, hidden3)

                    curr_char = phi[0, 0].argmax().item()
                    char_counts[curr_char] += 1

                    pi, mu_out, sd_out, ro, eos = mixture
                    comp = torch.multinomial(pi[0, 0], 1).item() if stochastic else pi[0, 0].argmax().item()
                    num_c = len(pi[0, 0])
                    mx = mu_out[0, 0, comp]
                    my = mu_out[0, 0, comp + num_c]
                    sx = sd_out[0, 0, comp] * temp
                    sy = sd_out[0, 0, comp + num_c] * temp
                    r = ro[0, 0, comp]
                    cov = r * sx * sy
                    sigma = torch.tensor([[sx**2, cov], [cov, sy**2]])
                    gmm = torch.distributions.MultivariateNormal(torch.tensor([mx.item(), my.item()]), sigma)
                    v = gmm.sample() if stochastic else torch.tensor([mx.item(), my.item()])
                    e = 1.0 if eos[0, 0].item() > 0.5 else 0.0
                    x_temp = torch.tensor([v[0].item(), v[1].item(), e], device=self.device).unsqueeze(0)

                    last_phi = phi[0, 0, u - 1].item()
                    is_eos = x_temp[0, 2].item() > 0.5
                    if last_phi > 0.8 or (curr_char == u - 1 and is_eos):
                        x_temp[0, 2] = 1.0
                        outputs.append(x_temp)
                        break

                    outputs.append(x_temp)
                    x = x_temp.unsqueeze(0)

            # Check coverage
            valid_chars = char_counts[:len(clean_text)]
            min_c = int(valid_chars.min()) if len(valid_chars) > 0 else 0
            total_pts = len(outputs)

            deltas = (torch.cat(outputs, dim=0).cpu() * self.sd + self.mu).numpy()
            N = len(deltas)
            raw_offsets = np.zeros((N, 4), dtype=np.float32)
            raw_offsets[:, 0] = deltas[:, 0]
            raw_offsets[:, 1] = -deltas[:, 1]
            raw_offsets[:, 2] = deltas[:, 2]
            raw_offsets[:, 3] = 0.02

            scaled_offsets, _ = normalize_scale(raw_offsets, target_median=1.0)
            coords = offsets_to_coords(scaled_offsets)
            coords = denoise(coords, window=7, polyorder=3)

            base = float(np.percentile(coords[:, 1], 20))
            coords[:, 1] -= base
            # Natural handwriting scale for ruled notebooks
            coords[:, :2] *= 1.7

            w_span = coords[:, 0].max() - coords[:, 0].min()
            h_span = coords[:, 1].max() - coords[:, 1].min()
            aspect = w_span / (h_span + 1e-5)

            if seed is not None:
                return coords

            # Quality criteria: all characters attended, valid aspect ratio, length proportional to text
            if min_c >= 4 and aspect > 1.8 and min_expected_pts <= total_pts <= max_expected_pts:
                return coords

            if min_c > best_score:
                best_score = min_c
                best_coords = coords

        return best_coords if best_coords is not None else coords
