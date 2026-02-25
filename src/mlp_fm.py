#!/usr/bin/env python3
import argparse
import json
import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from msc import FashionMnistIdxDataset, get_device, plot_losses


class MLPNet(nn.Module):
    def __init__(
        self,
        input_shape: int,
        hidden1: int,
        hidden2: int,
        output_shape: int,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_decode = nn.Linear(hidden2, output_shape)

    def forward(self, x: torch.Tensor):
        x_flat = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x_flat))
        h2 = F.relu(self.fc2(h1))
        return self.fc_decode(h2)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        loss.backward()
        optimizer.step()
        bs = X.size(0)
        running_loss += loss.item() * bs
        preds = torch.argmax(logits, dim=1)
        correct += (preds == Y).sum().item()
        total += bs
    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    model.eval()
    with torch.no_grad():
        batches = list(dataloader)
        Xs, Ys = zip(*batches)
        X = torch.cat(Xs, dim=0).to(device)
        Y = torch.cat(Ys, dim=0).to(device)
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        preds = torch.argmax(logits, dim=1)
        acc = 100.0 * (preds == Y).float().mean().item()
    return loss.item(), acc


def save_dir(hidden1: int, hidden2: int, lr: float, epochs: int):
    run_name = f"mlp_fm_{hidden1}_{hidden2}_{lr}_{epochs}"
    run_dir = f"save_stats_weights/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    return run_name, f"{run_dir}/{run_name}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mlp fashion-mnist training without ib layer.")
    parser.add_argument("--hidden1", type=int, required=True, default=1024, help="size of first hidden layer")
    parser.add_argument("--hidden2", type=int, required=True, default=512, help="size of second hidden layer")
    parser.add_argument("--epochs", type=int, required=True, default=200, help="number of training epochs")
    parser.add_argument("--rnd_seed", type=bool, default=False, help="random torch seed or default of 42")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--data_dir", type=str, default="data/mnist_fashion/", help="dataset path")
    args = parser.parse_args()

    device = get_device()
    if not args.rnd_seed:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

    run_name, run_prefix = save_dir(args.hidden1, args.hidden2, args.lr, args.epochs)
    print(
        "hyperparameters:\n"
        f"\tmodel         = mlp_fm\n"
        f"\thidden1       = {args.hidden1}\n"
        f"\thidden2       = {args.hidden2}\n"
        f"\tlr            = {args.lr}\n"
        f"\tepochs        = {args.epochs}\n"
        f"\tbatch_size    = {args.batch_size}\n"
        f"\tdevice        = {device}\n"
        f"\trnd_seed      = {args.rnd_seed}\n"
        f"\tsave_dir      = {run_prefix}"
    )

    train_dataset = FashionMnistIdxDataset(args.data_dir, train=True)
    test_dataset = FashionMnistIdxDataset(args.data_dir, train=False)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    model = MLPNet(784, args.hidden1, args.hidden2, 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    print(f"# of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_losses, test_losses = [], []
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate_epoch(model, test_loader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(
            f"epoch [{epoch + 1}/{args.epochs}] train loss: {train_loss:.3f} | train acc: {train_acc:.2f}%\n"
            f"\t\t\ttest loss: {test_loss:.3f} | test acc: {test_acc:.2f}%"
        )

    torch.save(model.state_dict(), f"{run_prefix}.pth")
    with open(f"{run_prefix}_stats.json", "w") as json_file:
        json.dump(
            {
                "model": "mlp_fm",
                "hidden1": args.hidden1,
                "hidden2": args.hidden2,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "rnd_seed": args.rnd_seed,
                "test_losses": test_losses,
                "test_acc": test_acc,
            },
            json_file,
            indent=2,
        )
    plot_losses(test_losses, train_losses, run_name, run_prefix)
