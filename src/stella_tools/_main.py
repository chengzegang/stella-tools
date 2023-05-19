from datetime import datetime
from typing import List, Optional
from matplotlib import pyplot as plt
import torch
from .dp import KeyframeData
from .proj import reproj, sim3_to_so3
import typer
import os
import pandas as pd
import numpy as np
from torch import Tensor

app = typer.Typer()


@app.command()
def main(
    db_path: str,
    out_path: str = "./*_trajectory.csv",
    out_fig: Optional[str] = "./*_trajectory.png",
):
    typer.secho(
        "✨ Welcome to the Stella-VSLAM map database trajectory extractor! 🗺️",
        fg="green",
    )

    db_name = os.path.basename(db_path).split(".")[0]
    out_path = out_path.replace("*", db_name)
    out_fig = out_fig.replace("*", db_name) if out_fig is not None else None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = KeyframeData(db_path)
    trajectory: List[Tensor] = []

    curr_location = torch.as_tensor([0, 0, 0, 1]).float().view(1, 4)
    trajectory = []
    last_pose = torch.eye(4).cuda().to(device).view(1, 4, 4)
    timestamps: List[float] = []
    with typer.progressbar(data, label="propagating positions") as pbar:
        for _, (ts, curr_pose, *_) in enumerate(pbar):
            timestamps.append(ts)
            curr_pose = curr_pose.to(device).float().view(1, 4, 4)

            curr_pose = torch.linalg.inv(curr_pose)
            curr_pose = sim3_to_so3(curr_pose)

            curr_location = reproj(curr_location.to(device), last_pose, curr_pose)
            curr_location = curr_location / curr_location[:, -1:]
            trajectory.append(curr_location.clone().cpu())
            last_pose = curr_pose
    timestamps = torch.as_tensor(timestamps)
    trajectory = torch.cat(trajectory)
    _plot_figure(trajectory, db_name, out_fig)
    _save_file(timestamps, trajectory, out_path)
    typer.secho("Done!", fg=typer.colors.GREEN)


def _plot_figure(traj: Tensor, db_name: str, out_path: str | None = None) -> None:
    if out_path is not None:
        traj = traj.cpu().numpy()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(traj[:, 0], traj[:, 2], traj[:, 1], color="blue")
        ax.set_aspect("equal")
        ax.set_title(f"Trajectory - {db_name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.savefig(out_path)
        plt.close(fig)
        typer.secho(f"Figure saved to {out_path}", fg=typer.colors.GREEN)


def _save_file(timestamps: Tensor, traj: Tensor, out_path: str) -> None:
    def _save_to_csv() -> None:
        print(len(timestamps), traj.shape)
        df = pd.DataFrame(
            {
                "timestamp": [datetime.fromtimestamp(ts) for ts in timestamps.tolist()],
                "x": traj[:, 0].tolist(),
                "y": traj[:, 1].tolist(),
                "z": traj[:, 2].tolist(),
            }
        )

        out_csv = os.path.splitext(out_path)[0] + ".csv"
        df.to_csv(out_csv, index=False)

    ext = os.path.splitext(out_path)[-1]
    if ext == ".npy":
        torch.stack([timestamps, traj], dim=-1)
        np.save(out_path, traj.numpy())
    elif ext == ".txt":
        torch.stack([timestamps, traj], dim=-1)
        np.savetxt(out_path, traj.numpy())
    elif ext == ".pt":
        torch.stack([timestamps, traj], dim=-1)
        torch.save(traj, out_path)
    elif ext == ".csv":
        _save_to_csv()
    else:
        typer.secho(f"unsupported file extension: {ext}", fg=typer.colors.RED)
        typer.secho("supported extensions: .npy, .csv, .txt, .pt", fg=typer.colors.RED)
        typer.secho("saving to default .csv file", fg=typer.colors.YELLOW)
        _save_to_csv()
    typer.secho(f"Trajectory saved to {out_path}", fg=typer.colors.GREEN)
