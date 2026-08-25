#!/usr/bin/env python3
"""Inventory the immutable full-DOF sources required by the S6 micropanel gate."""

from pathlib import Path

import h5py


DATA = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\Full Data Extracción\Original_extractions_20260801\dataset_original_v1")
PATTERNS = {
    "CAPACITY_LOADED": "full_dof_state_recovery_calfit41_V40_A_E6_C10_1T_v1/original_full_dof_state_pilot.h5",
    "BASE_C1_0T": "full_dof_state_recovery_panel_BASE_C1_0T_v1/original_full_dof_state_pilot.h5",
    "BASE_C2_0T": "full_dof_state_recovery_panel_BASE_C2_0T_v1/original_full_dof_state_pilot.h5",
    "BASE_C3_0T": "full_dof_state_recovery_panel_BASE_C3_0T_v1/original_full_dof_state_pilot.h5",
    "COMBINED": "full_dof_state_recovery_panel_V40_CPLUS_E2_C5_2T_v1/original_full_dof_state_pilot.h5",
    "V52_TRAIN": "full_dof_state_recovery_panel_V52_B_E6_C10_1T_v1/original_full_dof_state_pilot.h5",
}


def main() -> None:
    for label, relative in PATTERNS.items():
        path = DATA / relative
        print(f"\n[{label}] {path} exists={path.exists()}")
        if not path.exists():
            continue
        with h5py.File(path, "r") as handle:
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"{name}\tshape={obj.shape}\tdtype={obj.dtype}")
            handle.visititems(visitor)
    compact = DATA.parent / "cases" / "V40_CPLUS_E2_C5_2T" / "compact_kinematics.h5"
    print(f"\n[COMPACT_EXAMPLE] {compact} exists={compact.exists()}")
    if compact.exists():
        with h5py.File(compact, "r") as handle:
            print(f"attrs={dict(handle.attrs)}")
            handle.visititems(
                lambda name, obj: print(f"{name}\tshape={obj.shape}\tdtype={obj.dtype}")
                if isinstance(obj, h5py.Dataset) else None
            )


if __name__ == "__main__":
    main()
