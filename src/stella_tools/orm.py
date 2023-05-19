from sqlmodel import (
    BLOB,
    Column,
    Field,
    SQLModel,
)
from typing import Optional
import numpy as np


class Keyframes(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    src_frm_id: int
    ts: float
    __cam = Column("cam", BLOB)
    __orb_params = Column("orb_params", BLOB)
    __pose_cw = Column("pose_cw", BLOB)
    n_keypts: int
    __undist_keypts = Column("undist_keypts", BLOB, nullable=True)
    __x_rights = Column("x_rights", BLOB, nullable=True)
    __depths = Column("depths", BLOB, nullable=True)
    __descs = Column("descs", BLOB, nullable=True)

    class Config:
        extend_existing = True
        arbitrary_types_allowed = True

    class Parser:
        OPENCV_KEYPOINT_DTYPE: np.dtype = np.dtype(
            [
                ("pt", np.float32, 2),
                ("size", np.float32),
                ("angle", np.float32),
                ("response", np.float32),
                ("octave", np.int32),
                ("class_id", np.int32),
            ]
        )

        @classmethod
        def parse_pose_cw(cls, pose_cw: bytes) -> np.ndarray:
            return np.frombuffer(buffer=pose_cw, dtype=np.float64).reshape(4, 4).T

        @classmethod
        def parse_cam(cls, cam: bytes) -> str:
            return str(cam, "utf-8")

        @classmethod
        def parse_orb_params(cls, orb_params: bytes) -> np.ndarray:
            return np.frombuffer(buffer=orb_params, dtype=np.float64)

        @classmethod
        def parse_keypts(cls, keypts: bytes) -> np.ndarray:
            stct_arr = np.frombuffer(buffer=keypts, dtype=cls.OPENCV_KEYPOINT_DTYPE)
            return stct_arr

        @classmethod
        def parse_descs(cls, descs: bytes) -> np.ndarray:
            return np.frombuffer(buffer=descs, dtype=np.uint8).reshape(-1, 32)

    @property
    def pose_cw(self) -> np.ndarray:
        return self.Parser.parse_pose_cw(self.__pose_cw)  # type: ignore

    @property
    def cam(self) -> str:
        return self.Parser.parse_cam(self.__cam)  # type: ignore

    @property
    def orb_params(self) -> np.ndarray:
        return self.Parser.parse_orb_params(self.__orb_params)  # type: ignore

    @property
    def undist_keypts(self) -> np.ndarray:
        return self.Parser.parse_keypts(self.__undist_keypts)  # type: ignore

    @property
    def x_rights(self):
        return NotImplemented

    @property
    def depths(self):
        return NotImplemented

    @property
    def descs(self) -> np.ndarray:
        return self.Parser.parse_descs(self.__descs)  # type: ignore


class Landmarks(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    first_keyfrm: int
    __pose_w = Column("pose_cw", BLOB)
    ref_keyfrm: int
    n_vis: int
    n_fnd: int

    class Config:
        extend_existing = True
        arbitrary_types_allowed = True

    class Parser:
        @classmethod
        def parse_pose_w(cls, pose_cw: bytes) -> np.ndarray:
            return np.frombuffer(buffer=pose_cw, dtype=np.float64).reshape(4, 4)

    @property
    def pose_w(self) -> np.ndarray:
        return self.Parser.parse_pose_cw(self.__pose_w)  # type: ignore


class Cameras(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: bytes
    setup_type: bytes
    model_type: bytes
    color_type: bytes
    cols: int
    rows: int
    fps: float
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    k4: float
    focal_x_baseline: float
    distortion: float


class Associations(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    lm_ids: bytes
    span_parent: int
    n_spaning_children: int
    spanning_children: bytes
    n_loop_edges: int
    loop_edges: bytes
