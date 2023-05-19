from typing import List
from orm import Keyframes
from sqlalchemy.engine import Engine
from sqlmodel import Session, create_engine, select
import kuzu  # type: ignore
import numpy as np
import os
from collections import defaultdict
import shutil
import pandas as pd


# TODO: multi-process reading and cache
def _read_and_cache(sql: str, cache_dir: str, kfm_id_list: List[int]):
    kfm_id_set = set(kfm_id_list)
    engine = create_engine(f"sqlite:///{sql}")
    with Session(engine) as session:
        statement = select(Keyframes).where(Keyframes.id in kfm_id_set)
        keyframes = session.exec(statement)

        kfm_dict = defaultdict(list)
        ldm_dict = defaultdict(list)
        rel_records: List[dict] = []
        desc_id = 0

        for kfm in keyframes:
            kfm_id = np.asarray(kfm.id)
            kfm_ts = np.asarray(kfm.ts)
            kfm_pose_cw = kfm.pose_cw.flatten()

            keypts = kfm.undist_keypts

            kfm_keypts_pt = keypts["pt"].flatten()
            kfm_keypts_size = keypts["size"].flatten()
            kfm_keypts_angle = keypts["angle"].flatten()
            kfm_keypts_response = keypts["response"].flatten()
            kfm_keypts_desc = kfm.descs.reshape(-1, 32)

            kfm_dict["id"].append(kfm_id)
            kfm_dict["ts"].append(kfm_ts)
            kfm_dict["pose_cw"].append(kfm_pose_cw.astype(np.float32))

            kfm_ids = np.full(kfm.n_keypts, kfm_id)
            desc_ids = np.arange(desc_id, desc_id + kfm.n_keypts, dtype=np.int64)

            rel_dict = defaultdict(list)
            rel_dict["from"].append(kfm_ids.reshape(-1, 1))
            rel_dict["to"].append(desc_ids.reshape(-1, 1))
            rel_dict["pt"].append(kfm_keypts_pt.reshape(-1, 2).astype(np.float32))
            rel_dict["size"].append(kfm_keypts_size.reshape(-1, 1).astype(np.float32))
            rel_dict["angle"].append(kfm_keypts_angle.reshape(-1, 1).astype(np.float32))
            rel_dict["resp"].append(
                kfm_keypts_response.reshape(-1, 1).astype(np.float32)
            )
            rel_records.append(rel_dict)

            ldm_dict["id"].append(desc_ids.reshape(-1, 1))
            ldm_dict["dsc"].append(kfm_keypts_desc.reshape(-1, 32).astype(np.int16))
            desc_id += kfm.n_keypts


def cache_intermediate_files(engine: Engine, cache_dir: str):
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    with Session(engine) as session:
        statement = select(Keyframes)
        keyframes = session.exec(statement)

        kfm_dict = defaultdict(list)
        ldm_dict = defaultdict(list)
        rel_records: List[dict] = []
        desc_id = 0

        for kfm in keyframes:
            kfm_id = np.asarray(kfm.id)
            kfm_ts = np.asarray(kfm.ts)
            kfm_pose_cw = kfm.pose_cw.flatten()

            keypts = kfm.undist_keypts

            kfm_keypts_pt = keypts["pt"].flatten()
            kfm_keypts_size = keypts["size"].flatten()
            kfm_keypts_angle = keypts["angle"].flatten()
            kfm_keypts_response = keypts["response"].flatten()
            kfm_keypts_desc = kfm.descs.reshape(-1, 32)

            kfm_dict["id"].append(kfm_id)
            kfm_dict["ts"].append(kfm_ts)
            kfm_dict["pose_cw"].append(kfm_pose_cw.astype(np.float32))

            kfm_ids = np.full(kfm.n_keypts, kfm_id)
            desc_ids = np.arange(desc_id, desc_id + kfm.n_keypts, dtype=np.int64)

            rel_dict = defaultdict(list)
            rel_dict["from"].append(kfm_ids.reshape(-1, 1))
            rel_dict["to"].append(desc_ids.reshape(-1, 1))
            rel_dict["pt"].append(kfm_keypts_pt.reshape(-1, 2).astype(np.float32))
            rel_dict["size"].append(kfm_keypts_size.reshape(-1, 1).astype(np.float32))
            rel_dict["angle"].append(kfm_keypts_angle.reshape(-1, 1).astype(np.float32))
            rel_dict["resp"].append(
                kfm_keypts_response.reshape(-1, 1).astype(np.float32)
            )
            rel_records.append(rel_dict)

            ldm_dict["id"].append(desc_ids.reshape(-1, 1))
            ldm_dict["dsc"].append(kfm_keypts_desc.reshape(-1, 32).astype(np.int16))
            desc_id += kfm.n_keypts

        kfm_dir = os.path.join(cache_dir, "keyframe")
        os.makedirs(kfm_dir, exist_ok=True)
        for k, v in kfm_dict.items():
            kfm_dict[k] = np.vstack(v)  # type: ignore
            np.save(os.path.join(kfm_dir, f"node_{k}.npy"), kfm_dict[k])

        rel_dir = os.path.join(cache_dir, "detects")
        os.makedirs(rel_dir, exist_ok=True)
        for idx, rel_dict in enumerate(rel_records):  # type: ignore
            for k, v in rel_dict.items():
                rel_dict[k] = np.vstack(v)  # type: ignore
                rel_dict[k] = (
                    rel_dict[k].flatten() if rel_dict[k].shape[-1] == 1 else rel_dict[k]  # type: ignore
                )
                rel_dict[k] = rel_dict[k].tolist()  # type: ignore
            pd.DataFrame.from_dict(rel_dict).to_csv(
                os.path.join(rel_dir, f"detects_{idx}.csv"),
                header=False,
                index=False,
            )

        ldm_dir = os.path.join(cache_dir, "landmark")
        os.makedirs(ldm_dir, exist_ok=True)
        for k, v in ldm_dict.items():
            ldm_dict[k] = np.vstack(v)  # type: ignore
            np.save(os.path.join(ldm_dir, f"node_{k}.npy"), ldm_dict[k])


def create_keyframe_table(db: kuzu.Database):
    conn = kuzu.Connection(db, num_threads=os.cpu_count())
    conn.execute(
        "CREATE NODE TABLE Landmark (id INT64, dsc INT16[32], PRIMARY KEY (id));"
    )
    conn.execute(
        "CREATE NODE TABLE Keyframe (id INT64,ts DOUBLE,pose_cw FLOAT[16],PRIMARY KEY (id));"
    )
    conn.execute(
        "CREATE REL TABLE Detects (FROM Keyframe TO Landmark, pt FLOAT[2], size FLOAT, angle FLOAT, resp FLOAT);"
    )


def write_intermediate_to_kuzu(db: kuzu.Database, cache_dir: str):
    conn = kuzu.Connection(db, num_threads=2)
    keyframe_dir = os.path.join(cache_dir, "keyframe")
    conn.execute(
        f"""
            COPY Keyframe FROM ("{keyframe_dir}/node_id.npy", "{keyframe_dir}/node_ts.npy", "{keyframe_dir}/node_pose_cw.npy") BY COLUMN;
            """
    )
    landmark_dir = os.path.join(cache_dir, "landmark")
    conn.execute(
        f"""
            COPY Landmark FROM ("{landmark_dir}/node_id.npy", "{landmark_dir}/node_dsc.npy") BY COLUMN;
            """
    )
    detects_dir = os.path.join(cache_dir, "detects")
    conn.execute(
        f"""
        COPY Detects FROM "{detects_dir}/detects_*.csv";
        """
    )


def _to_kuzu(engine: Engine, db: kuzu.Database, rm_cache: bool = True):
    cache_dir = os.path.join(db.database_path, "kuzu_cache")
    create_keyframe_table(db)
    cache_intermediate_files(engine, cache_dir)
    write_intermediate_to_kuzu(db, "kuzu_cache")
    if rm_cache:
        shutil.rmtree(cache_dir)


def to_kuzu(
    stella_sqlite_db: str,
    kuzu_db: str,
    rm_cache: bool = True,
    buffer_size: int = 1024**3 * 16,
):
    engine = create_engine(f"sqlite:///{stella_sqlite_db}")
    db = kuzu.Database(kuzu_db, buffer_pool_size=buffer_size)
    _to_kuzu(engine, db, rm_cache)
