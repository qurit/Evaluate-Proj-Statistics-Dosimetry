"""DICOM-driven acquisition configuration shared by reconstruction and batch runs."""
import math

AC225_WINDOWS = {
    "218keV": {"Photopeak": 0, "Lower": 1, "Upper": 2, "Energy_keV": 218.0,
               "Expected_Bounds_keV": [[196.2, 239.8], [174.4, 196.2], [239.8, 261.6]]},
    "440keV": {"Photopeak": 3, "Lower": 4, "Upper": 5, "Energy_keV": 440.0,
               "Expected_Bounds_keV": [[396, 484], [352, 396], [484, 528]]},
}


def frame_duration_seconds(ds):
    """Require a uniform, positive frame duration (DICOM stores milliseconds)."""
    rotations = getattr(ds, "RotationInformationSequence", [])
    values = [float(r.ActualFrameDuration) / 1000 for r in rotations
              if hasattr(r, "ActualFrameDuration")]
    if not values and hasattr(ds, "ActualFrameDuration"):
        values = [float(ds.ActualFrameDuration) / 1000]
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Missing or invalid ActualFrameDuration in projection DICOM.")
    if any(not math.isclose(v, values[0], abs_tol=1e-6) for v in values):
        raise ValueError("Variable frame durations are not supported.")
    return values[0]


def resolve_acquisition(ds, params):
    """Validate window indices, bounds, energy and requested exposure for one bed."""
    window = dict(params.get("Energy_Windows", {"Photopeak": 0, "Lower": 1, "Upper": 2}))
    energy = float(window.get("Energy_keV", params.get("Photopeak_Energy", 208)))
    if "Photopeak_Energy" in params and not math.isclose(energy, float(params["Photopeak_Energy"])):
        raise ValueError("Photopeak_Energy disagrees with Energy_Windows.Energy_keV.")
    scatter = params.get("Scatter", {})
    method = scatter.get("Method", "TEW")
    if method not in ("TEW", "None"):
        raise ValueError("Scatter.Method must be 'TEW' or 'None'.")
    roles = ["Photopeak", "Lower", "Upper"] if method == "TEW" else ["Photopeak"]
    bounds = {}
    indices = []
    for role in roles:
        index = window.get(role)
        if not isinstance(index, int) or not 0 <= index < len(ds.EnergyWindowInformationSequence):
            raise ValueError(f"Invalid {role} energy-window index: {index}")
        indices.append(index)
        ranges = ds.EnergyWindowInformationSequence[index].EnergyWindowRangeSequence
        if len(ranges) != 1:
            raise ValueError("Each selected window must contain exactly one energy range.")
        low, high = float(ranges[0].EnergyWindowLowerLimit), float(ranges[0].EnergyWindowUpperLimit)
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high):
            raise ValueError("Invalid energy-window boundaries.")
        bounds[role] = [low, high]
    if len(set(indices)) != len(indices):
        raise ValueError("Photopeak and scatter windows must be distinct.")
    if not bounds["Photopeak"][0] <= energy <= bounds["Photopeak"][1]:
        raise ValueError("Selected energy lies outside the photopeak window.")
    if method == "TEW" and not (bounds["Lower"][1] <= bounds["Photopeak"][0] + 0.05
                               and bounds["Upper"][0] >= bounds["Photopeak"][1] - 0.05):
        raise ValueError("Scatter windows must flank the photopeak without overlap.")
    expected = window.get("Expected_Bounds_keV")
    if expected is not None:
        for role, pair in zip(["Photopeak", "Lower", "Upper"], expected):
            if role in bounds and any(not math.isclose(a, b, abs_tol=0.05) for a, b in zip(bounds[role], pair)):
                raise ValueError(f"DICOM {role} bounds do not match the configured acquisition.")
    original = frame_duration_seconds(ds)
    reduction = params.get("Data_Reduction", {})
    if "Time" in reduction and "Target_Seconds" in reduction:
        raise ValueError("Specify Target_Seconds or legacy Time fraction, not both.")
    target = float(reduction.get("Target_Seconds", original * reduction.get("Time", 1.0)))
    # Scanner timing may differ from the nominal duration by a millisecond.
    if math.isclose(target, original, rel_tol=0, abs_tol=0.002):
        target = original
    if not math.isfinite(target) or not 0 < target <= original:
        raise ValueError(f"Target duration must be in (0, {original}] seconds.")
    return {"window": window, "bounds_keV": bounds, "energy_keV": energy,
            "scatter_method": method, "original_seconds": original,
            "target_seconds": target, "time_fraction": target / original}
