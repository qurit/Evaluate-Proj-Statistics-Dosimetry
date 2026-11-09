"""Run separate Ac-225 window/time reconstructions: python -m reconstruction.batch --help."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import pydicom

from reconstruction.Acquisition import AC225_WINDOWS, resolve_acquisition


def dicom_files(directory, modality):
    """Accept DICOM regardless of extension; ignore Windows metadata sidecars."""
    files = []
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.name.endswith(":Zone.Identifier"):
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            continue
        if getattr(ds, "Modality", None) == modality:
            files.append(str(path.resolve()))
    if not files:
        raise ValueError(f"No {modality} DICOM files found in {directory}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projections", required=True, help="Directory containing projection DICOMs")
    parser.add_argument("--ct", required=True, help="Directory containing CT slices")
    parser.add_argument("--output", required=True, help="Parent directory for new reconstruction folders")
    parser.add_argument("--times", nargs="+", type=float, required=True, help="Target seconds per projection")
    parser.add_argument("--windows", nargs="+", choices=AC225_WINDOWS, default=list(AC225_WINDOWS))
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--subsets", type=int, default=8)
    parser.add_argument("--realizations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scatter", choices=["TEW", "None"], default="TEW")
    parser.add_argument("--smooth-scatter", action="store_true")
    parser.add_argument("--normalization", choices=["raw", "original_duration"], default="raw")
    parser.add_argument("--intrinsic-resolution-140kev-cm", type=float)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan without reconstructing or writing")
    args = parser.parse_args()
    if min(args.iterations, args.subsets, args.realizations) < 1:
        parser.error("iterations, subsets and realizations must be positive")
    if len(set(args.times)) != len(args.times) or len(set(args.windows)) != len(args.windows):
        parser.error("Duplicate target times or windows are not allowed")
    projections = dicom_files(args.projections, "NM")
    ct = dicom_files(args.ct, "CT")
    datasets = [pydicom.dcmread(p, stop_before_pixels=True) for p in projections]
    jobs = []
    for window_name in args.windows:
        for seconds in args.times:
            for realization in range(args.realizations):
                name = f"OSEM_{window_name}_{seconds:g}s_r{realization}"
                params = {
                    "Recon_Name": name, "Algorithm": "OSEM", "Collimator": "SY-HE",
                    "Energy_Windows": deepcopy(AC225_WINDOWS[window_name]),
                    "Data_Reduction": {"Target_Seconds": seconds},
                    "Scatter": {"Method": args.scatter, "Smooth": args.smooth_scatter},
                    "Algorithm_HyperParameters": {"n_iters": args.iterations, "n_subsets": args.subsets},
                    "Output_Normalization": args.normalization,
                    "Seed": args.seed + realization,
                }
                if args.intrinsic_resolution_140kev_cm is not None:
                    if args.intrinsic_resolution_140kev_cm < 0:
                        parser.error("Intrinsic resolution cannot be negative")
                    params["PSF_Parameters"] = {"intrinsic_resolution_140keV": args.intrinsic_resolution_140kev_cm}
                acquisitions = [resolve_acquisition(ds, params) for ds in datasets]
                for ds in datasets:
                    vector = list(ds.EnergyWindowVector)
                    count = vector.count(params["Energy_Windows"]["Photopeak"] + 1)
                    if args.subsets > count:
                        parser.error(f"Requested subsets exceed {count} projections")
                destination = Path(args.output) / name
                if destination.exists():
                    raise FileExistsError(f"Output already exists: {destination}")
                jobs.append((params, str(destination), acquisitions))
    print(json.dumps({"projection_files": projections, "ct_slice_count": len(ct),
                      "jobs": [{"output": dest, "parameters": params, "acquisitions": acq}
                               for params, dest, acq in jobs]}, indent=2))
    if not args.dry_run:
        from reconstruction.Tools import reconstruct_study
        for params, destination, _ in jobs:
            reconstruct_study(projections, ct, params, destination)


if __name__ == "__main__":
    main()
