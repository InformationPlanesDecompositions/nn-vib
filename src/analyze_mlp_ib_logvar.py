#!/usr/bin/env python3
import os
import re
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from msc import CIFAR10Dataset, FashionMnistIdxDataset, get_device

mlp_input_shape = 784
cnn_input_shape = (3, 32, 32)
output_shape = 10
mlp_batch_size = 128
cnn_batch_size = 512

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
fashion_data_dir = os.path.join(repo_root, "data", "mnist_fashion")
cifar_data_dir = os.path.join(repo_root, "data", "CIFAR-10")
mlp_model_root = os.path.abspath(os.path.join(repo_root, "..", "SaveStatsWeights", "Mlp"))
cnn_model_root = os.path.abspath(os.path.join(repo_root, "..", "SaveStatsWeights", "Cnn"))
mlp_run_name_pattern = re.compile(
    r"^vib_mlp_(\d+)_(\d+)_(\d+)_([0-9.eE+-]+)_([0-9.eE+-]+)_(\d+)_(\d+)$"
)
cnn_run_name_pattern = re.compile(
    r"^vib_cnn_(\d+)_(\d+)_(\d+)_(\d+)_([0-9.eE+-]+)_([0-9.eE+-]+)_(\d+)_(\d+)$"
)

class VIBMLPLogvar(nn.Module):
    def __init__(
        self,
        z_dim: int,
        input_shape: int,
        hidden1: int,
        hidden2: int,
        output_shape: int,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden1)
        self.fc_mu_logvar = nn.Linear(hidden1, 2 * z_dim)
        self.fc2 = nn.Linear(z_dim, hidden2)
        self.fc_decode = nn.Linear(hidden2, output_shape)

    def forward(self, x: torch.Tensor):
        x_flat = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x_flat))
        _, logvar = torch.chunk(self.fc_mu_logvar(h), 2, dim=1)
        return torch.clamp(logvar, min=-10.0, max=2.0)

class VIBCNNLogvar(nn.Module):
    def __init__(
        self,
        z_dim: int,
        input_shape: tuple[int, int, int],
        hidden1: int,
        hidden2: int,
        decoder_hidden: int,
        output_shape: int,
    ):
        super().__init__()
        channels, _, _ = input_shape

        self.conv1 = nn.Conv2d(channels, hidden1, kernel_size=5)
        self.conv2 = nn.Conv2d(hidden1, hidden2, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_dim = self._forward_features(dummy).shape[1]

        self.fc1 = nn.Linear(flat_dim, 120)
        self.fc_mu_logvar = nn.Linear(120, 2 * z_dim)
        self.fc2 = nn.Linear(z_dim, decoder_hidden)
        self.fc_decode = nn.Linear(decoder_hidden, output_shape)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor):
        h = torch.tanh(self.fc1(self._forward_features(x)))
        _, logvar = torch.chunk(self.fc_mu_logvar(h), 2, dim=1)
        return torch.clamp(logvar, min=-10.0, max=2.0)

def parse_mlp_run_dir(run_dir: str) -> dict[str, object] | None:
    run_name = os.path.basename(run_dir.rstrip(os.sep))
    match = mlp_run_name_pattern.match(run_name)
    if match is None:
        return None

    hidden1, hidden2, z_dim, beta, lr, epochs, seed = match.groups()
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "hidden1": int(hidden1),
        "hidden2": int(hidden2),
        "z_dim": int(z_dim),
        "beta": float(beta),
        "lr": float(lr),
        "epochs": int(epochs),
        "seed": int(seed),
    }

def parse_cnn_run_dir(run_dir: str) -> dict[str, object] | None:
    run_name = os.path.basename(run_dir.rstrip(os.sep))
    match = cnn_run_name_pattern.match(run_name)
    if match is None:
        return None

    hidden1, hidden2, decoder_hidden, z_dim, beta, lr, epochs, seed = match.groups()
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "hidden1": int(hidden1),
        "hidden2": int(hidden2),
        "decoder_hidden": int(decoder_hidden),
        "z_dim": int(z_dim),
        "beta": float(beta),
        "lr": float(lr),
        "epochs": int(epochs),
        "seed": int(seed),
    }

def mlp_config_key(run: dict[str, object]) -> tuple[int, int, int, float, float, int]:
    return (
        run["hidden1"],
        run["hidden2"],
        run["z_dim"],
        run["beta"],
        run["lr"],
        run["epochs"],
    )

def cnn_config_key(run: dict[str, object]) -> tuple[int, int, int, int, float, float, int]:
    return (
        run["hidden1"],
        run["hidden2"],
        run["decoder_hidden"],
        run["z_dim"],
        run["beta"],
        run["lr"],
        run["epochs"],
    )

def mlp_config_label(key: tuple[int, int, int, float, float, int]) -> str:
    hidden1, hidden2, z_dim, beta, lr, epochs = key
    return f"hidden1={hidden1} hidden2={hidden2} z_dim={z_dim} beta={beta:g} lr={lr:g} epochs={epochs}"

def cnn_config_label(key: tuple[int, int, int, int, float, float, int]) -> str:
    hidden1, hidden2, decoder_hidden, z_dim, beta, lr, epochs = key
    return (
        f"hidden1={hidden1} hidden2={hidden2} decoder_hidden={decoder_hidden} "
        f"z_dim={z_dim} beta={beta:g} lr={lr:g} epochs={epochs}"
    )

def mlp_model_config_key(key: tuple[int, int, int, float, float, int]) -> tuple[int, int, int, float, int]:
    hidden1, hidden2, z_dim, _, lr, epochs = key
    return (hidden1, hidden2, z_dim, lr, epochs)

def cnn_model_config_key(key: tuple[int, int, int, int, float, float, int]) -> tuple[int, int, int, int, float, int]:
    hidden1, hidden2, decoder_hidden, z_dim, _, lr, epochs = key
    return (hidden1, hidden2, decoder_hidden, z_dim, lr, epochs)

def mlp_model_config_label(key: tuple[int, int, int, float, int]) -> str:
    hidden1, hidden2, z_dim, lr, epochs = key
    return f"hidden1={hidden1} hidden2={hidden2} z_dim={z_dim} lr={lr:g} epochs={epochs}"

def cnn_model_config_label(key: tuple[int, int, int, int, float, int]) -> str:
    hidden1, hidden2, decoder_hidden, z_dim, lr, epochs = key
    return f"hidden1={hidden1} hidden2={hidden2} decoder_hidden={decoder_hidden} z_dim={z_dim} lr={lr:g} epochs={epochs}"

def grouped_mlp_runs() -> list[
    tuple[tuple[int, int, int, float, float, int], list[dict[str, object]]]
]:
    groups = {}
    for run_name in os.listdir(mlp_model_root):
        run_dir = os.path.join(mlp_model_root, run_name)
        if not os.path.isdir(run_dir):
            continue

        run = parse_mlp_run_dir(run_dir)
        if run is None:
            continue

        groups.setdefault(mlp_config_key(run), []).append(run)

    items = []
    for key, runs in groups.items():
        items.append((key, sorted(runs, key=lambda run: run["seed"])))
    return sorted(items, key=lambda item: item[0])

def grouped_cnn_runs() -> list[
    tuple[tuple[int, int, int, int, float, float, int], list[dict[str, object]]]
]:
    groups = {}
    for run_name in os.listdir(cnn_model_root):
        run_dir = os.path.join(cnn_model_root, run_name)
        if not os.path.isdir(run_dir):
            continue

        run = parse_cnn_run_dir(run_dir)
        if run is None:
            continue

        groups.setdefault(cnn_config_key(run), []).append(run)

    items = []
    for key, runs in groups.items():
        items.append((key, sorted(runs, key=lambda run: run["seed"])))
    return sorted(items, key=lambda item: item[0])

def pth_path_for_run(run_dir: str) -> str:
    pth_files = sorted(name for name in os.listdir(run_dir) if name.endswith(".pth"))
    if not pth_files:
        raise FileNotFoundError(f"no .pth file found in {run_dir}")
    return os.path.join(run_dir, pth_files[0])

def collect_logvars(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> torch.Tensor:
    model.eval()
    logvars = []

    with torch.no_grad():
        for X, _ in loader:
            logvars.append(model(X.to(device)).cpu())

    return torch.cat(logvars, dim=0)

if __name__ == "__main__":
    torch.set_printoptions(profile="full", linewidth=200)
    device = get_device()
    mlp_loader = DataLoader(
        FashionMnistIdxDataset(fashion_data_dir, train=False),
        batch_size=mlp_batch_size,
        shuffle=False,
    )
    cnn_loader = DataLoader(
        CIFAR10Dataset(cifar_data_dir, train=False, transform=CIFAR10Dataset.test_transform(cifar_data_dir)),
        batch_size=cnn_batch_size,
        shuffle=False,
    )

    mlp_mean_logvar_summary = {}
    cnn_mean_logvar_summary = {}

    print("\n# MLP averaged logvars")
    for key, runs in grouped_mlp_runs():
        averaged_logvars = None

        for run in runs:
            model = VIBMLPLogvar(
                run["z_dim"],
                mlp_input_shape,
                run["hidden1"],
                run["hidden2"],
                output_shape,
            ).to(device)
            model.load_state_dict(
                torch.load(pth_path_for_run(run["run_dir"]), map_location="cpu")
            )

            logvars = collect_logvars(model, mlp_loader, device)
            if averaged_logvars is None:
                averaged_logvars = logvars
            else:
                averaged_logvars += logvars

        averaged_logvars /= len(runs)

        print(f"\n# MLP {mlp_config_label(key)}")
        print(f"# seeds averaged: {[run['seed'] for run in runs]}")
        if len(runs) != 4:
            print(f"# WARNING: expected 4 seeds, found {len(runs)}")
        print(f"# averaged logvar shape: {tuple(averaged_logvars.shape)}")
        print(averaged_logvars)

        model_key = mlp_model_config_key(key)
        beta = key[3]
        mean_logvar_per_latent = averaged_logvars.mean(dim=0)
        mean_logvar_all = averaged_logvars.mean()
        mean_variance_all = averaged_logvars.exp().mean()
        mlp_mean_logvar_summary.setdefault(model_key, []).append(
            (beta, mean_logvar_per_latent, mean_logvar_all, mean_variance_all)
        )

    print("\n# CNN averaged logvars")
    for key, runs in grouped_cnn_runs():
        averaged_logvars = None

        for run in runs:
            model = VIBCNNLogvar(
                run["z_dim"],
                cnn_input_shape,
                run["hidden1"],
                run["hidden2"],
                run["decoder_hidden"],
                output_shape,
            ).to(device)
            model.load_state_dict(
                torch.load(pth_path_for_run(run["run_dir"]), map_location="cpu")
            )

            logvars = collect_logvars(model, cnn_loader, device)
            if averaged_logvars is None:
                averaged_logvars = logvars
            else:
                averaged_logvars += logvars

        averaged_logvars /= len(runs)

        print(f"\n# CNN {cnn_config_label(key)}")
        print(f"# seeds averaged: {[run['seed'] for run in runs]}")
        if len(runs) != 4:
            print(f"# WARNING: expected 4 seeds, found {len(runs)}")
        print(f"# averaged logvar shape: {tuple(averaged_logvars.shape)}")
        print(averaged_logvars)

        model_key = cnn_model_config_key(key)
        beta = key[4]
        mean_logvar_per_latent = averaged_logvars.mean(dim=0)
        mean_logvar_all = averaged_logvars.mean()
        mean_variance_all = averaged_logvars.exp().mean()
        cnn_mean_logvar_summary.setdefault(model_key, []).append(
            (beta, mean_logvar_per_latent, mean_logvar_all, mean_variance_all)
        )

    print("\n# MLP mean logvar summary")
    for model_key in sorted(mlp_mean_logvar_summary):
        print(f"\nmodel config: {mlp_model_config_label(model_key)}")
        for beta, mean_logvar_per_latent, mean_logvar_all, mean_variance_all in sorted(
            mlp_mean_logvar_summary[model_key], key=lambda item: item[0]
        ):
            print(f"  beta={beta:g}, mean_variance_all={mean_variance_all.item():.4f}")
            #print(f"    mean_logvar_all={mean_logvar_all.item()}")
            #print(f"    mean_logvar_per_latent={mean_logvar_per_latent.tolist()}")

    print("\n# CNN mean logvar summary")
    for model_key in sorted(cnn_mean_logvar_summary):
        print(f"\nmodel config: {cnn_model_config_label(model_key)}")
        for beta, mean_logvar_per_latent, mean_logvar_all, mean_variance_all in sorted(
            cnn_mean_logvar_summary[model_key], key=lambda item: item[0]
        ):
            print(f"  beta={beta:g}, mean_variance_all={mean_variance_all.item():.4f}")
            #print(f"    mean_logvar_all={mean_logvar_all.item()}")
            #print(f"    mean_logvar_per_latent={mean_logvar_per_latent.tolist()}")
