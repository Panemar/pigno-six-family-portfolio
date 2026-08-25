from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


@dataclass(frozen=True)
class CapacityMetadata:
    case_id: str
    base_case_id: str
    time_samples: int
    observation_nodes: int
    graph_nodes: int
    physical_rank: int
    residual_rank: int
    dt_s: float


class HistoricalCapacityDataset:
    """Read-only adapter for the admitted V4 full-dt capacity witness.

    The adapter exposes only causal inputs and same-grid FEM/COMSOL targets. It does
    not turn the historically exposed trajectory into validation evidence.
    """

    def __init__(self, dataset_h5: str | Path, graph_npz: str | Path):
        self.dataset_h5 = Path(dataset_h5)
        self.graph_npz = Path(graph_npz)
        if not self.dataset_h5.is_file() or not self.graph_npz.is_file():
            raise FileNotFoundError("Capacity witness dataset or graph inputs are missing")

        with h5py.File(self.dataset_h5, "r") as h5:
            self.time_s = h5["time_s"][:]
            self.global_series = h5["force/global_series"][:]
            self.observation_features = h5["force/observation_features"][:]
            self.reduced_force = h5["force/reduced_force"][:]
            self.load_node_force = h5["force/load_node_force_N"][:]
            self.load_node = h5["force/load_node_zero_based"][:]
            self.observation_node = h5["observation/graph_node_zero_based"][:]
            self.translation = h5["observation/FEM_COMSOL_delta_translation_m"][:]
            self.velocity = h5["observation/FEM_COMSOL_delta_velocity_mps"][:]
            self.q = h5["state/q_delta"][:]
            self.qdot = h5["state/qdot_delta"][:]
            self.M = h5["operator/M"][:]
            self.C = h5["operator/C"][:]
            self.K = h5["operator/K"][:]
            self.phi_graph_shape = tuple(h5["basis/phi_graph"].shape)
            self.case_id = str(h5.attrs["case_id"])
            self.base_case_id = str(h5.attrs["base_case_id"])
            self.physical_rank = int(h5.attrs["physical_rank"])
            self.total_rank = int(h5.attrs["selected_total_rank"])

        graph = np.load(self.graph_npz, allow_pickle=False)
        self.graph_coords = graph["graph_coords_m"]
        self.edge_index = graph["edge_index"]
        self.edge_attr = graph["edge_attr"]
        self.edge_frames = graph["edge_local_frame_R_local_from_global"]
        self.fixed_dof = graph["fixed_dof"]
        self.graph_node_features = np.concatenate(
            [
                graph["graph_coords_m"],
                graph["node_translation_stiffness_proxy_N_per_m_sym6"],
                graph["node_shear_stiffness_scale_N_per_m_sym6"],
                graph["node_rotation_stiffness_proxy_Nm_sym6"],
                graph["node_longitudinal_fabric_tensor_sym6"],
                graph["node_lumped_mass_proxy_kg"][:, None],
                graph["node_added_lumped_mass_kg"][:, None],
                graph["node_incident_length_m"][:, None],
            ],
            axis=1,
        )
        self.metadata = CapacityMetadata(
            case_id=self.case_id,
            base_case_id=self.base_case_id,
            time_samples=int(self.time_s.size),
            observation_nodes=int(self.observation_node.size),
            graph_nodes=int(self.graph_coords.shape[0]),
            physical_rank=self.physical_rank,
            residual_rank=self.total_rank - self.physical_rank,
            dt_s=float(np.median(np.diff(self.time_s))),
        )
        self._validate()

    def _validate(self) -> None:
        if self.metadata.case_id != "V40_A_E6_C10_1T" or self.metadata.base_case_id != "BASE_C1_0T":
            raise ValueError("Unexpected capacity/base case identity")
        if self.metadata.time_samples != 1201 or self.metadata.observation_nodes != 512:
            raise ValueError("Capacity witness grid changed")
        if not np.isclose(self.metadata.dt_s, 0.025, rtol=0.0, atol=1e-12):
            raise ValueError("Saved time step changed")
        if self.edge_index.shape != (2, 48430) or self.metadata.graph_nodes != 22164:
            raise ValueError("Active Beam graph changed")
        if self.fixed_dof.shape != (22164, 6) or int(self.fixed_dof.sum()) != 36:
            raise ValueError("Hard-BC mask changed")
        if self.phi_graph_shape != (22164 * 6, 224):
            raise ValueError("Six-DOF graph basis changed")
        numeric = (
            self.global_series,
            self.observation_features,
            self.reduced_force,
            self.load_node_force,
            self.translation,
            self.velocity,
            self.q,
            self.qdot,
            self.M,
            self.C,
            self.K,
            self.graph_node_features,
            self.edge_attr,
            self.edge_frames,
        )
        if not all(np.isfinite(item).all() for item in numeric):
            raise ValueError("Non-finite value in capacity witness")

    def observation_input(self) -> np.ndarray:
        global_broadcast = np.broadcast_to(
            self.global_series[:, None, :],
            (self.metadata.time_samples, self.metadata.observation_nodes, self.global_series.shape[1]),
        )
        time = (self.time_s / max(float(self.time_s[-1]), np.finfo(float).eps))[:, None, None]
        time_broadcast = np.broadcast_to(time, (self.metadata.time_samples, self.metadata.observation_nodes, 1))
        coords = np.broadcast_to(
            self.graph_coords[self.observation_node][None, :, :],
            (self.metadata.time_samples, self.metadata.observation_nodes, 3),
        )
        return np.concatenate([self.observation_features, global_broadcast, time_broadcast, coords], axis=-1).astype(np.float32)

    def graph_load_field(self, time_indices: np.ndarray) -> np.ndarray:
        field = np.zeros((len(time_indices), self.metadata.graph_nodes, 3), dtype=np.float32)
        field[:, self.load_node, :] = self.load_node_force[time_indices]
        return field

    def observation_basis(self) -> np.ndarray:
        """Return the six-DOF basis rows at the identical 512 observation nodes."""
        row_index = (self.observation_node[:, None] * 6 + np.arange(6)[None, :]).reshape(-1)
        with h5py.File(self.dataset_h5, "r") as h5:
            # h5py requires monotonically increasing fancy indices. Restore the
            # requested node/DOF order after one sorted read.
            order = np.argsort(row_index)
            inverse = np.empty_like(order)
            inverse[order] = np.arange(order.size)
            selected = h5["basis/phi_graph"][row_index[order], :][inverse]
        return selected.reshape(self.metadata.observation_nodes, 6, self.total_rank)

    @staticmethod
    def tensor(array: np.ndarray, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(array, device=device, dtype=dtype)


@dataclass(frozen=True)
class MicropanelMetadata:
    cases: int
    time_samples: int
    observation_nodes: int
    graph_nodes: int
    physical_rank: int
    displacement_rank_per_axis: int
    velocity_rank_per_axis: int
    dt_s: float


class HistoricalMicropanelDataset:
    """Read-only historical-panel adapter with separate physical and field coordinates."""

    def __init__(self, dataset_h5: str | Path, representation_h5: str | Path, graph_npz: str | Path):
        self.dataset_h5 = Path(dataset_h5)
        self.representation_h5 = Path(representation_h5)
        self.graph_npz = Path(graph_npz)
        for path in (self.dataset_h5, self.representation_h5, self.graph_npz):
            if not path.is_file():
                raise FileNotFoundError(path)
        decode = lambda value: value.decode("utf-8") if isinstance(value, bytes) else str(value)
        with h5py.File(self.dataset_h5, "r") as h5:
            if str(h5.attrs["status"]) not in {
                "PASS_S6_SIX_CASE_MICROPANEL_DATASET",
                "PASS_S8_BALANCED_FACTORIAL_PANEL_DATASET",
            }:
                raise ValueError("Historical panel source is not internally complete")
            self.case_id = [decode(value) for value in h5["case_id"][:]]
            self.base_case_id = [decode(value) for value in h5["base_case_id"][:]]
            self.time_s = h5["time_s"][:]
            self.static_features = h5["causal/static_features"][:]
            self.global_series = h5["force/global_series"][:]
            self.reduced_force = h5["force/reduced_force"][:]
            self.load_node_force = h5["force/load_node_force_N"][:]
            self.load_node = h5["force/load_node_zero_based"][:]
            self.observation_node = h5["observation/graph_node_zero_based"][:]
            self.translation = h5["response/delta_translation_m"][:]
            self.velocity = h5["response/delta_velocity_mps"][:]
            self.total_translation = h5["response/total_translation_m"][:]
            self.total_velocity = h5["response/total_velocity_mps"][:]
            self.q13 = h5["state/q_direct_full_dof_13"][:, :, :32]
            self.qdot13 = h5["state/qdot_direct_full_dof_13"][:, :, :32]
            self.direct_times_s = h5["state/direct_full_dof_times_s"][:]
            self.M = h5["operator/M"][:32, :32]
            self.C = h5["operator/C"][:32, :32]
            self.K = h5["operator/K"][:32, :32]
        with h5py.File(self.representation_h5, "r") as h5:
            if str(h5.attrs["status"]) not in {
                "PASS_S6_DUAL_STATE_FIELD_REPRESENTATION",
                "PASS_S8_DUAL_STATE_FIELD_REPRESENTATION",
                "PASS_S9_FOLD_LOCAL_REPRESENTATION",
            }:
                raise ValueError("Dual state/field representation is not admitted")
            if [decode(value) for value in h5["case_id"][:]] != self.case_id:
                raise ValueError("Micropanel/representation case order changed")
            self.displacement_basis = h5["displacement/basis_by_axis"][:]
            self.velocity_basis = h5["velocity/basis_by_axis"][:]
            self.displacement_coefficients = h5["displacement/coefficients"][:]
            self.velocity_coefficients = h5["velocity/coefficients"][:]
            self.physical_basis_at_observations = h5["physical/basis_at_observations_6dof"][:]
            displacement_rank = int(h5.attrs["displacement_rank_per_axis"])
            velocity_rank = int(h5.attrs["velocity_rank_per_axis"])

        graph = np.load(self.graph_npz, allow_pickle=False)
        self.graph_coords = graph["graph_coords_m"]
        self.edge_index = graph["edge_index"]
        self.edge_attr = graph["edge_attr"]
        self.edge_frames = graph["edge_local_frame_R_local_from_global"]
        self.fixed_dof = graph["fixed_dof"]
        self.graph_node_features = np.concatenate(
            [
                graph["graph_coords_m"],
                graph["node_translation_stiffness_proxy_N_per_m_sym6"],
                graph["node_shear_stiffness_scale_N_per_m_sym6"],
                graph["node_rotation_stiffness_proxy_Nm_sym6"],
                graph["node_longitudinal_fabric_tensor_sym6"],
                graph["node_lumped_mass_proxy_kg"][:, None],
                graph["node_added_lumped_mass_kg"][:, None],
                graph["node_incident_length_m"][:, None],
            ],
            axis=1,
        )
        self.direct_time_index = np.asarray(
            [[int(np.argmin(np.abs(self.time_s - value))) for value in case_times] for case_times in self.direct_times_s],
            dtype=np.int64,
        )
        self.metadata = MicropanelMetadata(
            cases=len(self.case_id),
            time_samples=int(self.time_s.size),
            observation_nodes=int(self.observation_node.size),
            graph_nodes=int(self.graph_coords.shape[0]),
            physical_rank=32,
            displacement_rank_per_axis=displacement_rank,
            velocity_rank_per_axis=velocity_rank,
            dt_s=float(np.median(np.diff(self.time_s))),
        )
        self._validate_micropanel()

    def _validate_micropanel(self) -> None:
        if self.metadata.cases not in {6, 12} or self.metadata.time_samples != 1201 or self.metadata.observation_nodes != 512:
            raise ValueError("Historical panel dimensions changed")
        if self.metadata.displacement_rank_per_axis != 64 or self.metadata.velocity_rank_per_axis != 128:
            raise ValueError("Specialized field ranks changed")
        if not np.isclose(self.metadata.dt_s, 0.025, rtol=0.0, atol=1e-12):
            raise ValueError("Saved time step changed")
        if self.edge_index.shape != (2, 48430) or self.metadata.graph_nodes != 22164:
            raise ValueError("Active Beam graph changed")
        if self.fixed_dof.shape != (22164, 6) or int(self.fixed_dof.sum()) != 36:
            raise ValueError("Hard-BC mask changed")
        numeric = (
            self.static_features, self.global_series, self.reduced_force, self.load_node_force,
            self.translation, self.velocity, self.q13, self.qdot13, self.M, self.C, self.K,
            self.displacement_basis, self.velocity_basis, self.displacement_coefficients,
            self.velocity_coefficients, self.physical_basis_at_observations,
            self.graph_node_features, self.edge_attr, self.edge_frames,
        )
        if not all(np.isfinite(item).all() for item in numeric):
            raise ValueError("Non-finite value in micropanel")
        if np.max(np.abs(self.time_s[self.direct_time_index] - self.direct_times_s)) > 1e-12:
            raise ValueError("Direct full-DOF sample-time identity failed")

    def temporal_input(self) -> np.ndarray:
        static = np.broadcast_to(
            self.static_features[:, None, :],
            (self.metadata.cases, self.metadata.time_samples, self.static_features.shape[1]),
        )
        time = np.broadcast_to(
            (self.time_s / self.time_s[-1])[None, :, None],
            (self.metadata.cases, self.metadata.time_samples, 1),
        )
        return np.concatenate([self.global_series, self.reduced_force[:, :, :32], static, time], axis=-1).astype(np.float32)


class HistoricalOOFDataset:
    """Read-only 68-trajectory adapter for leakage-safe nested grouped OOF."""

    def __init__(self, dataset_h5: str | Path, representation_h5: str | Path, graph_npz: str | Path):
        self.dataset_h5 = Path(dataset_h5)
        self.representation_h5 = Path(representation_h5)
        self.graph_npz = Path(graph_npz)
        for path in (self.dataset_h5, self.representation_h5, self.graph_npz):
            if not path.is_file():
                raise FileNotFoundError(path)
        decode = lambda value: value.decode("utf-8") if isinstance(value, bytes) else str(value)
        with h5py.File(self.dataset_h5, "r") as h5:
            if str(h5.attrs["status"]) != "PASS_S10_ORIGINAL_68CASE_DATASET_INTERNAL":
                raise ValueError("S10 historical dataset is not internally complete")
            self.case_id = [decode(value) for value in h5["case_id"][:]]
            self.base_case_id = [decode(value) for value in h5["base_case_id"][:]]
            self.time_s = h5["time_s"][:]
            self.static_features = h5["causal/static_features"][:]
            self.global_series = h5["causal/external_series"][:]
            self.reduced_force = h5["force/reduced_force"][:]
            self.load_node_force = h5["force/load_node_force_N"][:]
            self.load_node = h5["force/load_node_zero_based"][:]
            self.observation_node = h5["observation/graph_node_zero_based"][:]
            self.translation = h5["response/delta_translation_m"][:]
            self.velocity = h5["response/delta_velocity_mps"][:]
            self.total_translation = h5["response/total_translation_m"][:]
            self.total_velocity = h5["response/total_velocity_mps"][:]
            self.q13 = h5["state/q_direct_full_dof_13_or_zero"][:, :, :32]
            self.qdot13 = h5["state/qdot_direct_full_dof_13_or_zero"][:, :, :32]
            self.direct_times_s = h5["state/direct_full_dof_times_s_or_zero"][:]
            self.direct_state_available = h5["state/direct_full_dof_available"][:].astype(bool)
            self.M = h5["operator/M"][:32, :32]
            self.C = h5["operator/C"][:32, :32]
            self.K = h5["operator/K"][:32, :32]
        with h5py.File(self.representation_h5, "r") as h5:
            if str(h5.attrs["status"]) != "PASS_S10_FOLD_LOCAL_REPRESENTATION":
                raise ValueError("S10 fold-local representation is not admitted")
            if [decode(value) for value in h5["case_id"][:]] != self.case_id:
                raise ValueError("S10 dataset/representation case order changed")
            self.displacement_basis = h5["displacement/basis_by_axis"][:]
            self.velocity_basis = h5["velocity/basis_by_axis"][:]
            self.displacement_coefficients = h5["displacement/coefficients"][:]
            self.velocity_coefficients = h5["velocity/coefficients"][:]
            self.physical_basis_at_observations = h5["physical/basis_at_observations_6dof"][:]
            displacement_rank = int(h5.attrs["displacement_rank_per_axis"])
            velocity_rank = int(h5.attrs["velocity_rank_per_axis"])
        graph = np.load(self.graph_npz, allow_pickle=False)
        self.graph_coords = graph["graph_coords_m"]
        self.edge_index = graph["edge_index"]
        self.edge_attr = graph["edge_attr"]
        self.edge_frames = graph["edge_local_frame_R_local_from_global"]
        self.fixed_dof = graph["fixed_dof"]
        self.graph_node_features = np.concatenate([
            graph["graph_coords_m"], graph["node_translation_stiffness_proxy_N_per_m_sym6"],
            graph["node_shear_stiffness_scale_N_per_m_sym6"], graph["node_rotation_stiffness_proxy_Nm_sym6"],
            graph["node_longitudinal_fabric_tensor_sym6"], graph["node_lumped_mass_proxy_kg"][:, None],
            graph["node_added_lumped_mass_kg"][:, None], graph["node_incident_length_m"][:, None],
        ], axis=1)
        self.direct_time_index = np.zeros((len(self.case_id), 13), dtype=np.int64)
        for case in np.flatnonzero(self.direct_state_available):
            self.direct_time_index[case] = [int(np.argmin(np.abs(self.time_s - value))) for value in self.direct_times_s[case]]
        self.metadata = MicropanelMetadata(
            cases=len(self.case_id), time_samples=int(self.time_s.size), observation_nodes=int(self.observation_node.size),
            graph_nodes=int(self.graph_coords.shape[0]), physical_rank=32,
            displacement_rank_per_axis=displacement_rank, velocity_rank_per_axis=velocity_rank,
            dt_s=float(np.median(np.diff(self.time_s))),
        )
        if self.metadata.cases != 68 or self.metadata.time_samples != 1201 or self.metadata.observation_nodes != 512:
            raise ValueError("S10 historical OOF dimensions changed")
        if displacement_rank != 64 or velocity_rank != 128:
            raise ValueError("S10 specialized field ranks changed")
        if int(self.direct_state_available.sum()) != 12:
            raise ValueError("S10 direct-state availability contract changed")
        if not np.isclose(self.metadata.dt_s, 0.025, rtol=0.0, atol=1e-12):
            raise ValueError("S10 saved time step changed")

    def temporal_input(self) -> np.ndarray:
        static = np.broadcast_to(self.static_features[:, None, :], (68, 1201, self.static_features.shape[1]))
        time = np.broadcast_to((self.time_s / self.time_s[-1])[None, :, None], (68, 1201, 1))
        return np.concatenate([self.global_series, self.reduced_force[:, :, :32], static, time], axis=-1).astype(np.float32)
