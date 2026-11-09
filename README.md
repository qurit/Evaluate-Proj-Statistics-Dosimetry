# Evaluate-Proj-Statistics-Dosimetry

Research tools for reduced-acquisition SPECT reconstruction and PR21 dosimetry.

## Ac-225 phantom reconstruction

Run from the repository root in an environment with PyTomography, PyTorch,
pydicom, and their dependencies installed:

```bash
python -m reconstruction.batch \
  --projections data-ac225/Ac225-CTN-NEMA/NEMA/Projections \
  --ct data-ac225/Ac225-CTN-NEMA/NEMA/CTAC \
  --output results/nema \
  --times 300 150 60 30 \
  --windows 218keV 440keV \
  --iterations 24 --subsets 8 \
  --seed 42 --dry-run
```

Remove `--dry-run` to reconstruct. Dry runs validate metadata and print the plan
without writing outputs. Durations and iteration settings above are examples,
not validated optimal reconstruction settings. Use separate input/output paths
for each phantom. Inputs may use `.IMA` or `.dcm` extensions; Windows
`:Zone.Identifier` sidecars are ignored. CT slices must already be compatible
with the projection geometry; this tool does not register images.

The two predefined groups are:

| Group | Photopeak index/bounds (keV) | Lower index/bounds | Upper index/bounds |
|---|---|---|---|
| 218keV | 0 / 196.2–239.8 | 1 / 174.4–196.2 | 2 / 239.8–261.6 |
| 440keV | 3 / 396–484 | 4 / 352–396 | 5 / 484–528 |

Indices are zero-based and boundaries are checked against each projection file.
Each group is reconstructed separately using OSEM, CT attenuation correction at
its selected energy, and the PyTomography `SY-HE` analytical PSF approximation.
`SY-HE` is the Siemens Symbia high-energy model, used as the closest library
approximation to the Intevo HE collimator. Full penetration/scatter response is
not represented by the analytical Gaussian PSF. Detector intrinsic resolution
is unspecified by default (zero in the library); supply measured/manufacturer
FWHM in cm with `--intrinsic-resolution-140kev-cm` if available.

The original duration is read from DICOM `ActualFrameDuration` in the rotation
sequence. Nominal durations within 2 ms of the recorded duration are treated as
full duration, preserving the original counts and random-generator state.
Reduced acquisitions use binomial thinning of raw integer counts independently
for the photopeak and scatter windows, followed by TEW scatter estimation.
Use `--scatter None` to disable scatter correction or `--smooth-scatter` to
smooth the estimated scatter. Other scatter methods are not implemented.
Projection angles are retained; subset count must not exceed projection count.

`--realizations N` runs N realizations per window/time with seeds
`seed + realization`. Each job starts its own generator; using the same seed
across jobs does not make exposures nested, independent, or disjoint event
samples. Separate runs on different hardware/library versions need not be
bitwise reproducible. Full-duration repeats have identical input counts.

Outputs are new folders named `OSEM_<window>_<seconds>s_r<realization>` containing:

- `reconstruction.dcm`: derived reconstructed image with DICOM rescale slope.
- `provenance.json`: source files, actual energy bounds and timings, reduction
  fraction, seed, algorithm/scatter/PSF settings, and library versions.

Existing reconstruction folders are refused. `--normalization raw` (default)
retains raw reconstructed values; `--normalization original_duration` divides
by the retention fraction for comparison at the original exposure scale.
Neither option provides activity calibration. DICOM derivation text records
simulated target exposure; exact timings and normalization are in the sidecar.
Consumers should apply the DICOM rescale slope when comparing image values.

## Python interface and existing workflows

`reconstruction.Tools.reconstruct_study(projection_files, ct_files, params,
output_path)` supports OSEM, BSREM, KEM, and the existing hybrid Monte Carlo
branch. The batch CLI intentionally uses OSEM. New parameters include:

```python
from reconstruction.Acquisition import AC225_WINDOWS

params = {
    "Recon_Name": "OSEM_218keV_60s_r0",
    "Algorithm": "OSEM",
    "Algorithm_HyperParameters": {"n_iters": 24, "n_subsets": 8},
    "Collimator": "SY-HE",
    "Energy_Windows": AC225_WINDOWS["218keV"],
    "Data_Reduction": {"Target_Seconds": 60},
    "Scatter": {"Method": "TEW", "Smooth": False},
    "Seed": 42,
    "Output_Normalization": "raw",
}
```

Custom groups can specify `Photopeak`, `Lower`, `Upper`, `Energy_keV`, and optional
`Expected_Bounds_keV` (photopeak/lower/upper order). With scatter disabled only
the photopeak index is required. `PSF_Parameters` forwards optional settings to
PyTomography's `get_psfmeta_from_scanner_params`.

The legacy `Data_Reduction.Time` fraction and angular reduction configuration
`Projections: {Angle_Reduction, Angle_Start_Index}` remain supported. Specify
only one of `Time` and `Target_Seconds`. Legacy window defaults are indices
0/1/2 at `Photopeak_Energy` (208 keV if omitted). Monte Carlo now requires
explicit `MC_Parameters.isotope_names` and `isotope_ratios`; source support must
be confirmed in the installed SIMIND backend before using Ac-225. The 140.5-keV
attenuation-map input remains the backend's reference-energy convention.

The inventory and dosimetry modules retain their original PR21 functionality.
The dosimetry filename parser is not compatible with the new batch output names;
this reconstruction workflow does not call it.

## Checks

```bash
python -m unittest discover -s tests -v
```
