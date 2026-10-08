from time import perf_counter
tstart_import = perf_counter()
from operator import ge
from typing import Callable
import numpy as np
from numpy.typing import NDArray

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset, TensorDataset, DataLoader, random_split)
from torch.optim import Optimizer, AdamW

try:
    from sklearn.datasets import fetch_california_housing
except ImportError:
    mssg = "This example depends on scikit-learn's california housing dataset. " \
        "This dataset is not a dependency for this project but is used in this example."
    raise ImportError(mssg)

try:
    from tqdm import tqdm
except ImportError:
    mssg = "This example depends on tqdm. This is not a dependency for this project, but is used in this example."
    raise ImportError(mssg)
tstop_import = perf_counter()
print(f"Time to import modules: {tstop_import - tstart_import} sec")

Tensor = torch.Tensor
Batch = tuple[Tensor, Tensor]
Criterion = Callable[[Tensor, Tensor], Tensor]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NFEATURES = 8


def time_wrapper(func_name: str):
    def time_wrapper_decorator(func: Callable):
        def wrapped_func(*args, **kwargs):
            tstart = perf_counter()
            res = func(*args, **kwargs)
            tstop = perf_counter()
            print(f"Time Elapsed {func_name}: {tstop - tstart} sec")
            return res
        return wrapped_func
    return time_wrapper_decorator


@time_wrapper("prepare_dataset")
def prepare_dataset(generator: torch.Generator, batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    X, y = fetch_california_housing(
        data_home=".tmp/", 
        download_if_missing=True, 
        return_X_y=True)
    y = y[:, None]
    _dset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())

    train_ratio = 0.8
    test_ratio = 1 - train_ratio
    train_dset, test_dset = random_split(
    _dset, (train_ratio, test_ratio), generator
    )

    train_dload = DataLoader(train_dset, batch_size=batch_size, shuffle=True, generator=generator)
    test_dload = DataLoader(test_dset, batch_size=batch_size, shuffle=False, generator=generator)
    return train_dload, test_dload


def prep_trainer(
    criterion: Criterion, 
    optimizer: Optimizer
) -> Callable[[nn.Module, DataLoader, DataLoader, str], tuple[float, float]]:
    def training_step(model: nn.Module, batch: Batch, device: str) -> float:
        # Prepare batch
        X, y = batch
        X, y = X.to(device), y.to(device)

        # Compute Loss
        yhat = model(X)
        loss = criterion(yhat, y)

        # Backpropagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    @torch.no_grad()
    def test_step(model: nn.Module, batch: Batch, device: str) -> float:
        # Prepare batch
        X, y = batch
        X, y = X.to(device), y.to(device)

        # Compute Loss
        yhat = model(X)
        loss = criterion(yhat, y)

        return loss.item()

    def train_loop(
        model: nn.Module, 
        train_dloader: DataLoader, 
        test_dloader: DataLoader, 
        device: str
    ) -> tuple[float, float]:
        model.train()
        train_loss = 0.0
        train_samples = 0
        for batch in tqdm(train_dloader):
            batch_loss = training_step(model, batch, device)
            batch_size = batch[0].shape[0]
            train_loss += batch_loss
            train_samples += batch_size
        train_loss /= train_samples

        model.eval()
        test_loss = 0.0
        test_samples = 0
        test_loss = 0
        for batch in tqdm(test_dloader):
            batch_loss = test_step(model, batch, device)
            batch_size = batch[0].shape[0]
            test_loss += batch_loss
            test_samples += batch_size
        test_loss /= test_samples

        return train_loss, test_loss

    return train_loop


def train_model(
    model: nn.Module, 
    num_epochs: int = 24, 
    batch_size: int = 64, 
    device: str = str(DEVICE), 
    seed: int = 1992
) -> tuple[NDArray, NDArray, NDArray]:
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    train_dload, test_dload = prepare_dataset(generator=generator, batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    train_loop = prep_trainer(criterion=criterion, optimizer=optimizer)

    epochs = np.arange(num_epochs)
    train_losses: list = []
    test_losses: list = []
    for epoch in range(num_epochs):
        train_loss, test_loss = train_loop(model, train_dload, test_dload, device)
        print(f"Epoch:{epoch}\tTrain Loss:{train_loss:0.04f}\tTest Loss:{test_loss:0.04f}\n")
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    return epochs, np.asarray(train_losses), np.asarray(test_losses)


def display_results(epochs: NDArray, train_losses: NDArray, test_losses: NDArray) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, 'bo-', label='Train Loss')
    ax.plot(epochs, test_losses, 'ro-', label='Test Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.set_title('Training Metrics')
    ax.set_yscale('log')
    ax.legend()
    return fig, ax


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features=NFEATURES, out_features=100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.max_val = 5
        self.min_val = 1.5
    
    def forward(self, X: Tensor) -> Tensor:
        return self.backbone(X)

    def get_weights(self, device: str = 'cpu') -> tuple[Tensor, Tensor, Tensor]:
        w1 = self.backbone[0].weight.detach()  # ty: ignore[call-non-callable]
        w2 = self.backbone[3].weight.detach()  # ty: ignore[call-non-callable]
        w3 = self.backbone[6].weight.detach()  # ty: ignore[call-non-callable]
        return w1.to(device), w2.to(device), w3.to(device)


if __name__ == "__main__":
    from tltorch import compress_matrix_to_mpo_budget as compression
    from tltorch.utils import get_tensorized_shape
    from tltorch import BlockTT, TensorizedTensor

    model = SimpleMLP()
    _ = train_model(model, num_epochs=24)
    model.eval()
    # fig, ax = display_results(epochs, train_loss, test_loss)
    # fig.show()

    w1, w2, w3 = model.get_weights()
    w2.detach()
    w2_num_params = w2.numel()
    rho = 0.2
    w2_norm = torch.linalg.norm(w2)**2

    tensor_shapes = get_tensorized_shape(
        in_features=w2.shape[1], 
        out_features=w2.shape[0])

    compression = time_wrapper('compression')(compression)

    block_w2, diagnostics = compression(
        matrix=w2,
        input_modes=tensor_shapes[0],
        output_modes=tensor_shapes[1],
        max_parameters = int(w2.numel() * rho), 
        max_sweeps_per_refinement = 10
    )
    block_w2_numel = sum([b2.numel() for b2 in block_w2.factors])
    block_w2_frob_norm = diagnostics['final_squared_frobenius_error']

    # Compare to existing factorization methods
    tt_w2 = TensorizedTensor.from_matrix(
        matrix=w2,
        tensorized_row_shape=tensor_shapes[1],
        tensorized_column_shape=tensor_shapes[0],
        rank=rho,
        factorization='blocktt'
    )
    tt_w2_numel = sum([b2.numel() for b2 in tt_w2.factors])
    tt_w2_frob_norm = torch.linalg.norm(tt_w2.to_matrix() - w2).item()**2
    
    print(f'W2: {w2.numel()}\n')

    print(f"DMRG Result: {print(block_w2)}")
    print(f"compression: {block_w2_numel / w2.numel()}\t rel_frob_err: {block_w2_frob_norm / w2_norm}\n")

    print(f"Baseline Result: {print(tt_w2)}")
    print(f"compression: {tt_w2_numel / w2.numel()}\t rel_frob_err: {tt_w2_frob_norm / w2_norm}\n")











