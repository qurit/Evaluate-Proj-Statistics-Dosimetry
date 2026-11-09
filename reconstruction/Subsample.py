from pytomography.io.SPECT.shared import subsample_projections, subsample_metadata
from torch import Tensor
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from typing import Tuple, Dict, Any

import torch

def binomial_thinning(projections: torch.Tensor, t_reduction_factor: float,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Thin nonnegative integer counts; preserve full-count data without sampling."""
    if not 0 <= t_reduction_factor <= 1:
        raise ValueError("t_reduction_factor must be in [0, 1].")
    if not torch.isfinite(projections).all() or (projections < 0).any():
        raise ValueError("Projection counts must be finite and nonnegative.")
    counts = projections.float()
    if not torch.equal(counts, counts.round()):
        raise ValueError("Binomial thinning requires integer projection counts.")
    if t_reduction_factor == 1:
        return projections.clone()
    return torch.binomial(counts, torch.full_like(counts, t_reduction_factor),
                          generator=generator)


def subsample_projections_number(
    photopeak: Tensor,
    scatter: Tensor,
    object_meta: SPECTObjectMeta,
    proj_meta: SPECTProjMeta,
    parameters: Dict[str, Any]) -> Tuple[Tensor, Tensor, SPECTObjectMeta, SPECTProjMeta]:
    """_summary_

    Parameters
    ----------
    photopeak : Tensor
        _description_
    scatter : Tensor
        _description_
    object_meta : SPECTObjectMeta
        _description_
    proj_meta : SPECTProjMeta
        _description_
    parameters : Dict[Any]
        _description_
        
    Returns
    -------
    Tuple[Tensor, Tensor, SPECTObjectMeta, SPECTProjMeta]
        _description_
    """
    
    if "Angle_Reduction" not in parameters or "Angle_Start_Index" not in parameters:
        raise AssertionError(f"Missing adequate parameters to subsample projecttions. Found {parameters}")
    
    object_meta, proj_meta = subsample_metadata(
        object_meta=object_meta,
        proj_meta=proj_meta,
        N_pixel=1,
        N_angle=parameters["Angle_Reduction"], 
        N_angle_start=parameters["Angle_Start_Index"]
    )
    
    # Photopeak
    photopeak = subsample_projections(
        projections=photopeak,
        N_pixel=1,
        N_angle=parameters["Angle_Reduction"],
        N_angle_start=parameters["Angle_Start_Index"],
                                    
    )
    
    # Scatter
    scatter = subsample_projections(projections=scatter,
                                    N_pixel=1,
                                    N_angle=parameters["Angle_Reduction"],
                                    N_angle_start=parameters["Angle_Start_Index"],
                                    )
    
    return photopeak, scatter, object_meta, proj_meta


def subsample_projections_time(projections: torch.Tensor, t_reduction_factor: float,
                              generator: torch.Generator | None = None) -> torch.Tensor:
    """Simulate a shorter acquisition using binomial thinning of raw counts."""
    return binomial_thinning(projections, t_reduction_factor, generator)
