from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


def set_seed(seed: int = SEED) -> None:
    """실험 재현성을 위해 Python, NumPy, PyTorch 난수를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_matplotlib() -> None:
    """Windows의 맑은 고딕을 등록해 PNG에서 한글이 깨지지 않게 한다."""
    font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    if font_path.exists():
        mpl.font_manager.fontManager.addfont(str(font_path))
    mpl.rcParams["font.family"] = "Malgun Gothic"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.dpi"] = 130


class MLP(nn.Module):
    """활성화 함수를 교체할 수 있는 간단한 MLP."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Iterable[int],
        num_classes: int,
        activation: str = "relu",
        small_init: bool = False,
        init_std: float = 0.01,
        hidden_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.activation_name = activation
        sizes = [input_size, *hidden_sizes, num_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )
        if small_init:
            self.apply_small_init(init_std=init_std, hidden_bias=hidden_bias)

    def apply_small_init(self, init_std: float, hidden_bias: float) -> None:
        """실험 B에서 작은 초기값과 음수 bias로 Dead ReLU 상황을 유도한다."""
        for idx, layer in enumerate(self.layers):
            nn.init.normal_(layer.weight, mean=0.0, std=init_std)
            if idx < len(self.layers) - 1:
                nn.init.constant_(layer.bias, hidden_bias)
            else:
                nn.init.zeros_(layer.bias)

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return F.relu(x)
        if self.activation_name == "leaky_relu":
            return F.leaky_relu(x, negative_slope=0.05)
        if self.activation_name == "sigmoid":
            return torch.sigmoid(x)
        raise ValueError(f"Unknown activation: {self.activation_name}")

    def forward(
        self, x: torch.Tensor, return_activations: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        activations: List[torch.Tensor] = []
        for layer in self.layers[:-1]:
            x = self.activate(layer(x))
            activations.append(x)
        logits = self.layers[-1](x)
        if return_activations:
            return logits, activations
        return logits


@dataclass
class RunResult:
    name: str
    history: pd.DataFrame
    final_test_acc: float
    min_test_loss: float
    convergence_epoch: int
    converged: bool
    grad_by_epoch: pd.DataFrame
    activation_summary: pd.DataFrame | None = None
    dead_matrix: np.ndarray | None = None
    activation_snapshots: Dict[int, List[np.ndarray]] | None = None
    model: MLP | None = None


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def load_digits_data() -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Digits 데이터셋: 8x8 회색조 숫자 이미지, 10-class 분류."""
    digits = load_digits()
    x = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=SEED, stratify=y
    )
    train_loader = make_loader(x_train, y_train, batch_size=128, shuffle=True, seed=SEED)
    test_loader = make_loader(x_test, y_test, batch_size=256, shuffle=False, seed=SEED)
    return (
        train_loader,
        test_loader,
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )


def load_moons_data() -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """make_moons 데이터셋: 2D 비선형 분류 문제."""
    x, y = make_moons(n_samples=2500, noise=0.25, random_state=SEED)
    x = StandardScaler().fit_transform(x).astype(np.float32)
    y = y.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=SEED, stratify=y
    )
    train_loader = make_loader(x_train, y_train, batch_size=128, shuffle=True, seed=SEED)
    test_loader = make_loader(x_test, y_test, batch_size=256, shuffle=False, seed=SEED)
    return (
        train_loader,
        test_loader,
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_name: str,
    loss_fn: nn.Module,
    num_classes: int,
) -> Tuple[float, float]:
    """평가는 반드시 model.eval()과 torch.no_grad()로 수행한다."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs)
            if loss_name == "mse":
                probs = torch.softmax(logits, dim=1)
                targets_oh = F.one_hot(targets, num_classes=num_classes).float()
                loss = loss_fn(probs, targets_oh)
                preds = probs.argmax(dim=1)
            else:
                loss = loss_fn(logits, targets)
                preds = logits.argmax(dim=1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += (preds == targets).sum().item()
            total_count += inputs.size(0)
    return total_loss / total_count, total_correct / total_count


def layer_grad_norms(model: MLP) -> Dict[str, float]:
    """각 Linear layer weight gradient의 L2 norm을 기록한다."""
    norms = {}
    for idx, layer in enumerate(model.layers):
        if layer.weight.grad is None:
            norms[f"layer_{idx + 1}"] = 0.0
        else:
            norms[f"layer_{idx + 1}"] = float(layer.weight.grad.detach().norm().cpu())
    return norms


def train_model(
    *,
    name: str,
    model: MLP,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    num_classes: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    snapshot_epochs: Iterable[int] = (),
    snapshot_inputs: torch.Tensor | None = None,
) -> RunResult:
    """직접 작성한 epoch 학습 루프. 손실함수/optimizer는 외부에서 명시적으로 주입한다."""
    loss_fn: nn.Module = nn.MSELoss() if loss_name == "mse" else nn.CrossEntropyLoss()
    rows: List[Dict[str, float]] = []
    grad_rows: List[Dict[str, float]] = []
    snapshots: Dict[int, List[np.ndarray]] = {}
    snapshot_set = set(snapshot_epochs)

    if 0 in snapshot_set and snapshot_inputs is not None:
        snapshots[0] = collect_activation_arrays(model, snapshot_inputs)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0
        epoch_grad_totals: Dict[str, float] = {}
        grad_batches = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)

            # 실험 A의 MSE 조건: 출력 logits에 softmax를 명시적으로 적용한 뒤 one-hot target과 비교한다.
            if loss_name == "mse":
                probs = torch.softmax(logits, dim=1)
                targets_oh = F.one_hot(targets, num_classes=num_classes).float()
                loss = loss_fn(probs, targets_oh)
                preds = probs.argmax(dim=1)
            else:
                loss = loss_fn(logits, targets)
                preds = logits.argmax(dim=1)

            loss.backward()
            batch_grad_norms = layer_grad_norms(model)
            for key, value in batch_grad_norms.items():
                epoch_grad_totals[key] = epoch_grad_totals.get(key, 0.0) + value
            grad_batches += 1
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (preds == targets).sum().item()
            train_count += inputs.size(0)

        if scheduler is not None:
            scheduler.step()

        test_loss, test_acc = evaluate(model, test_loader, loss_name, loss_fn, num_classes)
        current_lr = optimizer.param_groups[0]["lr"]
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / train_count,
                "train_acc": train_correct / train_count,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": current_lr,
            }
        )
        grad_row = {"epoch": epoch}
        grad_row.update({k: v / max(grad_batches, 1) for k, v in epoch_grad_totals.items()})
        grad_rows.append(grad_row)

        if epoch in snapshot_set and snapshot_inputs is not None:
            snapshots[epoch] = collect_activation_arrays(model, snapshot_inputs)

    history = pd.DataFrame(rows)
    grad_df = pd.DataFrame(grad_rows)
    final_acc = float(history["test_acc"].iloc[-1])
    min_loss = float(history["test_loss"].min())
    # 실패한 run의 낮은 최종 정확도를 기준으로 "수렴"했다고 착각하지 않도록
    # 최소 80% test accuracy에 도달한 경우만 수렴으로 인정한다.
    convergence_floor = 0.80
    target_acc = max(convergence_floor, final_acc * 0.95)
    reached = history.loc[history["test_acc"] >= target_acc, "epoch"]
    converged = bool(len(reached) and final_acc >= convergence_floor)
    convergence_epoch = int(reached.iloc[0]) if converged else int(epochs)
    return RunResult(
        name=name,
        history=history,
        final_test_acc=final_acc,
        min_test_loss=min_loss,
        convergence_epoch=convergence_epoch,
        converged=converged,
        grad_by_epoch=grad_df,
        activation_snapshots=snapshots,
        model=model,
    )


def collect_activation_arrays(model: MLP, inputs: torch.Tensor) -> List[np.ndarray]:
    """중간 layer activation을 no_grad로 수집한다."""
    model.eval()
    with torch.no_grad():
        _, activations = model(inputs, return_activations=True)
    return [a.detach().cpu().numpy() for a in activations]


def summarize_activations(model: MLP, inputs: torch.Tensor) -> Tuple[pd.DataFrame, np.ndarray]:
    """Dead ReLU 비율, Sigmoid saturation 비율, activation 평균/표준편차를 계산한다."""
    acts = collect_activation_arrays(model, inputs)
    rows = []
    max_width = max(a.shape[1] for a in acts)
    dead_matrix = np.full((len(acts), max_width), np.nan)
    for idx, arr in enumerate(acts):
        near_zero_by_neuron = (np.abs(arr).mean(axis=0) <= 1e-6)
        dead_matrix[idx, : arr.shape[1]] = near_zero_by_neuron.astype(float)
        saturated = ((arr < 0.05) | (arr > 0.95)).mean()
        rows.append(
            {
                "layer": f"hidden_{idx + 1}",
                "dead_ratio_pct": near_zero_by_neuron.mean() * 100.0,
                "saturation_ratio_pct": saturated * 100.0,
                "activation_mean": arr.mean(),
                "activation_std": arr.std(),
            }
        )
    return pd.DataFrame(rows), dead_matrix


def create_optimizer(
    name: str, model: nn.Module, lr: float
) -> torch.optim.Optimizer:
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if name == "SGD+Momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(name)


def run_experiment_a() -> Dict[str, RunResult]:
    train_loader, test_loader, _, _ = load_digits_data()
    results = {}
    for loss_name, label in [("ce", "CrossEntropy"), ("mse", "MSE_with_softmax")]:
        set_seed(SEED)
        model = MLP(input_size=64, hidden_sizes=[256, 128], num_classes=10, activation="relu")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        results[label] = train_model(
            name=label,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_name=loss_name,
            optimizer=optimizer,
            epochs=40,
            num_classes=10,
        )
    return results


def run_experiment_b() -> Dict[str, RunResult]:
    train_loader, test_loader, x_test, _ = load_moons_data()
    results = {}
    snapshot_epochs = [0, 150, 300]
    for activation, label in [
        ("relu", "ReLU"),
        ("leaky_relu", "LeakyReLU"),
        ("sigmoid", "Sigmoid"),
    ]:
        set_seed(SEED)
        # 모든 활성화 함수에 동일한 작은 초기값을 사용한다.
        # ReLU 계열은 음수 bias(-0.05) 때문에 일부 뉴런이 0 출력에 고착되는 경향이 나타난다.
        model = MLP(
            input_size=2,
            hidden_sizes=[128, 64],
            num_classes=2,
            activation=activation,
            small_init=True,
            init_std=0.01,
            hidden_bias=-0.05,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        result = train_model(
            name=label,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_name="ce",
            optimizer=optimizer,
            epochs=300,
            num_classes=2,
            snapshot_epochs=snapshot_epochs,
            snapshot_inputs=x_test,
        )
        result.activation_summary, result.dead_matrix = summarize_activations(model, x_test)
        results[label] = result
    return results


def run_experiment_c() -> Dict[str, RunResult]:
    train_loader, test_loader, _, _ = load_digits_data()
    results = {}
    for optimizer_name in ["SGD", "SGD+Momentum", "Adam"]:
        for lr in [0.1, 0.01, 0.001]:
            set_seed(SEED)
            model = MLP(input_size=64, hidden_sizes=[256, 128], num_classes=10, activation="relu")
            optimizer = create_optimizer(optimizer_name, model, lr=lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            label = f"{optimizer_name}_lr{lr:g}"
            results[label] = train_model(
                name=label,
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                loss_name="ce",
                optimizer=optimizer,
                epochs=40,
                num_classes=10,
                scheduler=scheduler,
            )
    return results


def save_history_tables(
    exp_a: Dict[str, RunResult],
    exp_b: Dict[str, RunResult],
    exp_c: Dict[str, RunResult],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    a_rows = [
        {
            "loss_function": name,
            "final_accuracy_pct": result.final_test_acc * 100,
            "min_loss": result.min_test_loss,
            "convergence_epoch": result.convergence_epoch,
            "convergence_status": "converged" if result.converged else "not_converged",
            "final_grad_norm_layer1": result.grad_by_epoch["layer_1"].iloc[-1],
            "final_grad_norm_output": result.grad_by_epoch.iloc[-1].filter(like="layer_").iloc[-1],
        }
        for name, result in exp_a.items()
    ]
    a_table = pd.DataFrame(a_rows)
    a_table.to_csv(TABLE_DIR / "experiment_a_summary.csv", index=False, encoding="utf-8-sig")

    b_rows = []
    for name, result in exp_b.items():
        assert result.activation_summary is not None
        dead_avg = result.activation_summary["dead_ratio_pct"].mean()
        sat_avg = result.activation_summary["saturation_ratio_pct"].mean()
        b_rows.append(
            {
                "activation": name,
                "dead_relu_like_ratio_pct": dead_avg,
                "saturation_ratio_pct": sat_avg,
                "final_accuracy_pct": result.final_test_acc * 100,
                "convergence_epoch": result.convergence_epoch,
                "convergence_status": "converged" if result.converged else "not_converged",
            }
        )
        result.activation_summary.to_csv(
            TABLE_DIR / f"experiment_b_{name}_activation_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    b_table = pd.DataFrame(b_rows)
    b_table.to_csv(TABLE_DIR / "experiment_b_summary.csv", index=False, encoding="utf-8-sig")

    c_rows = []
    for name, result in exp_c.items():
        last_10 = result.history["test_loss"].tail(10)
        loss_std = float(last_10.std())
        if result.final_test_acc < 0.75:
            stability = "stalled"
        elif loss_std < 0.03:
            stability = "stable"
        else:
            stability = "oscillating"
        opt_name, lr_text = name.rsplit("_lr", 1)
        c_rows.append(
            {
                "optimizer": opt_name,
                "learning_rate": float(lr_text),
                "final_accuracy_pct": result.final_test_acc * 100,
                "min_loss": result.min_test_loss,
                "convergence_epoch": result.convergence_epoch,
                "convergence_status": "converged" if result.converged else "not_converged",
                "last10_loss_std": loss_std,
                "stability": stability,
            }
        )
    c_table = pd.DataFrame(c_rows).sort_values(["optimizer", "learning_rate"], ascending=[True, False])
    c_table.to_csv(TABLE_DIR / "experiment_c_summary.csv", index=False, encoding="utf-8-sig")
    return a_table, b_table, c_table


def plot_loss_acc(results: Dict[str, RunResult], title: str, filename: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, result in results.items():
        axes[0].plot(result.history["epoch"], result.history["test_loss"], label=name)
        axes[1].plot(result.history["epoch"], result.history["test_acc"] * 100, label=name)
    axes[0].set_title(f"{title} - Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test Loss")
    axes[0].grid(alpha=0.3)
    axes[1].set_title(f"{title} - Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_grad_heatmap(results: Dict[str, RunResult], title: str, filename: str) -> Path:
    labels = list(results.keys())
    max_layers = max(len(r.grad_by_epoch.filter(like="layer_").columns) for r in results.values())
    fig, axes = plt.subplots(len(labels), 1, figsize=(11, 2.2 * len(labels)), squeeze=False)
    for row, label in enumerate(labels):
        grad_cols = [f"layer_{i + 1}" for i in range(max_layers)]
        data = results[label].grad_by_epoch.reindex(columns=grad_cols).T.to_numpy(dtype=float)
        data = np.log10(data + 1e-12)
        im = axes[row, 0].imshow(data, aspect="auto", cmap="viridis")
        axes[row, 0].set_title(f"{title}: {label} gradient flow (log10 norm)")
        axes[row, 0].set_ylabel("Layer")
        axes[row, 0].set_yticks(range(max_layers), grad_cols)
        axes[row, 0].set_xlabel("Epoch")
        fig.colorbar(im, ax=axes[row, 0], fraction=0.02, pad=0.01)
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dead_heatmaps(exp_b: Dict[str, RunResult]) -> Path:
    # hidden_1은 128개, hidden_2는 64개 뉴런이므로 한 imshow 행렬에 억지로
    # padding하면 빈 영역이 "0%"처럼 보인다. 각 layer를 별도 strip으로 그려
    # 존재하지 않는 뉴런과 살아있는 뉴런을 혼동하지 않게 한다.
    fig, axes = plt.subplots(2, 3, figsize=(12, 4.8))
    im = None
    for col, (name, result) in enumerate(exp_b.items()):
        assert result.dead_matrix is not None
        assert result.activation_summary is not None
        layer_dead = result.activation_summary["dead_ratio_pct"].to_list()
        for row in range(2):
            ax = axes[row, col]
            values = result.dead_matrix[row]
            values = values[~np.isnan(values)]
            im = ax.imshow(
                values.reshape(1, -1),
                aspect="auto",
                cmap="Reds",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_yticks([])
            ax.set_xlim(-0.5, len(values) - 0.5)
            if row == 0:
                ax.set_title(f"{name}\nH1 {layer_dead[0]:.1f}%, H2 {layer_dead[1]:.1f}% dead")
            if col == 0:
                ax.set_ylabel(f"hidden_{row + 1}", rotation=0, labelpad=34, va="center")
            if row == 1:
                ax.set_xlabel("Neuron index")
            else:
                ax.set_xticklabels([])
    assert im is not None
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.015,
        label="1 = dead/near-zero neuron",
        )
    path = FIG_DIR / "experiment_b_dead_relu_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_activation_histograms(exp_b: Dict[str, RunResult]) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=False)
    for row, (name, result) in enumerate(exp_b.items()):
        assert result.activation_snapshots is not None
        for col, epoch in enumerate([0, 150, 300]):
            arrays = result.activation_snapshots[epoch]
            flat = np.concatenate([a.reshape(-1) for a in arrays])
            axes[row, col].hist(flat, bins=40, color="#3267a8", alpha=0.82)
            axes[row, col].set_title(f"{name}, epoch {epoch}")
            axes[row, col].set_xlabel("Activation value")
            axes[row, col].set_ylabel("Count")
            axes[row, col].grid(alpha=0.2)
    fig.suptitle("Experiment B: layer activation distribution over training", fontsize=14)
    fig.tight_layout()
    path = FIG_DIR / "experiment_b_activation_histograms.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_optimizer_grid(exp_c: Dict[str, RunResult]) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for row, opt_name in enumerate(["SGD", "SGD+Momentum", "Adam"]):
        for lr in [0.1, 0.01, 0.001]:
            label = f"{opt_name}_lr{lr:g}"
            result = exp_c[label]
            axes[row, 0].plot(result.history["epoch"], result.history["test_loss"], label=f"lr={lr:g}")
            axes[row, 1].plot(result.history["epoch"], result.history["test_acc"] * 100, label=f"lr={lr:g}")
        axes[row, 0].set_title(f"{opt_name}: Loss")
        axes[row, 1].set_title(f"{opt_name}: Accuracy")
        axes[row, 0].set_ylabel("Test Loss")
        axes[row, 1].set_ylabel("Accuracy (%)")
        axes[row, 0].grid(alpha=0.3)
        axes[row, 1].grid(alpha=0.3)
        axes[row, 1].legend(fontsize=8)
    axes[-1, 0].set_xlabel("Epoch")
    axes[-1, 1].set_xlabel("Epoch")
    fig.tight_layout()
    path = FIG_DIR / "experiment_c_optimizer_lr_grid.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(
    exp_a: Dict[str, RunResult],
    exp_b: Dict[str, RunResult],
    exp_c: Dict[str, RunResult],
) -> Dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "a_curve": plot_loss_acc(exp_a, "Experiment A: Loss Function", "experiment_a_loss_accuracy.png"),
        "a_grad": plot_grad_heatmap(exp_a, "Experiment A", "experiment_a_gradient_flow.png"),
        "b_curve": plot_loss_acc(exp_b, "Experiment B: Activation Function", "experiment_b_loss_accuracy.png"),
        "b_hist": plot_activation_histograms(exp_b),
        "b_dead": plot_dead_heatmaps(exp_b),
        "b_grad": plot_grad_heatmap(exp_b, "Experiment B", "experiment_b_gradient_flow.png"),
        "c_curve": plot_optimizer_grid(exp_c),
        "c_grad": plot_grad_heatmap(
            {k: v for k, v in exp_c.items() if k in ["SGD_lr0.1", "SGD+Momentum_lr0.1", "Adam_lr0.001"]},
            "Experiment C",
            "experiment_c_gradient_flow_selected.png",
        ),
    }


def main() -> None:
    set_seed(SEED)
    configure_matplotlib()
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Running experiment A: loss functions")
    exp_a = run_experiment_a()
    print("[2/5] Running experiment B: activation functions")
    exp_b = run_experiment_b()
    print("[3/5] Running experiment C: optimizers and learning rates")
    exp_c = run_experiment_c()
    print("[4/5] Saving summary tables")
    save_history_tables(exp_a, exp_b, exp_c)
    print("[5/5] Creating figures")
    make_figures(exp_a, exp_b, exp_c)
    print("Done.")
    print(f"Code: {Path(__file__).resolve()}")
    print(f"Figures: {FIG_DIR}")
    print(f"Tables: {TABLE_DIR}")


if __name__ == "__main__":
    main()
