from typing import Any, Tuple
import torch
from sqlmodel import Session, create_engine, select, func
from .orm import Keyframes

import numpy as np
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.map import MapDataPipe
from torch import Tensor


@functional_datapipe("keyframe_data")
class KeyframeData(MapDataPipe):
    def __init__(self, root: str, *args, **kwargs) -> None:
        super().__init__()
        self.root = root
        self.engine = create_engine(f"sqlite:///{root}")
        with Session(self.engine) as session:
            statement = select(func.count(Keyframes.id))  # type: ignore
            self._size: int = session.exec(statement).first()
            statement = select(Keyframes.id)
            self._ids = session.exec(statement).all()

    def __reduce__(self) -> Tuple[Any, ...]:
        return self.__class__, (self.root,)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Tuple[float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.engine is None:
            self.engine = create_engine(f"sqlite:///{self.root}")
        with Session(self.engine) as session:
            statement = select(Keyframes).where(Keyframes.id == self._ids[idx])
            keyframe = session.exec(statement).all()[0]
            pose_cw = keyframe.pose_cw.copy()
            ts = keyframe.ts
            kpts: np.ndarray = keyframe.undist_keypts.copy()
            kpt_locs, kpt_sizes, kpt_angles, kpt_resp = (
                kpts["pt"],
                kpts["size"],
                kpts["angle"],
                kpts["response"],
            )
            descs = keyframe.descs.copy()

            pose_cw: Tensor = torch.from_numpy(pose_cw).view(4, 4)  # type: ignore
            kpt_locs: Tensor = torch.from_numpy(kpt_locs).view(-1, 2)  # type: ignore
            kpt_sizes: Tensor = torch.from_numpy(kpt_sizes)  # type: ignore
            kpt_angles: Tensor = torch.from_numpy(kpt_angles)  # type: ignore
            kpt_resp: Tensor = torch.from_numpy(kpt_resp)  # type: ignore
            descs: Tensor = torch.from_numpy(descs).view(-1, 32)  # type: ignore

            return ts, pose_cw, kpt_locs, kpt_sizes, kpt_angles, kpt_resp, descs
