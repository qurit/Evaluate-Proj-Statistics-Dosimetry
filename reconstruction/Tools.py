from pytomography.io.SPECT import dicom
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.priors import RelativeDifferencePrior, TopNAnatomyNeighbourWeight
from pytomography.algorithms import OSEM, BSREM, KEM
from pytomography.transforms.shared import KEMTransform
from pytomography.projectors.shared import KEMSystemMatrix
from pytomography.projectors.SPECT import SPECTSystemMatrix, MonteCarloHybridSPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood, MonteCarloHybridSPECTPoissonLogLikelihood
import pydicom
import torch
from typing import List, Dict
from reconstruction.Subsample import subsample_projections_number, subsample_projections_time
from pytomography.utils.scatter import get_smoothed_scatter
from pytomography.utils import simind_mc
from reconstruction.Acquisition import resolve_acquisition
from pathlib import Path
import json
import importlib.metadata

def reconstruct_study(projection_data_dcm: List[str], ct_data_dcm: List[str], params: Dict, output_path: str) -> None:
    """Reconstruct one or more beds and save DICOM plus provenance.json.

    Energy_Windows selects zero-based Photopeak/Lower/Upper indices and Energy_keV.
    Scatter.Method is TEW or None; Scatter.Smooth controls optional smoothing.
    Data_Reduction accepts Target_Seconds or the legacy Time fraction, and optional
    Projections {Angle_Reduction, Angle_Start_Index}. Seed controls binomial draws.
    Collimator and optional PSF_Parameters configure resolution modelling.
    Algorithm is OSEM, BSREM, KEM, or MC_OSEM; Algorithm_HyperParameters is passed
    to the algorithm. MC requires explicit isotope_names/isotope_ratios.
    Output_Normalization is raw (default) or original_duration; neither is an
    activity calibration. output_path must not exist. CT/projection files must be
    geometrically compatible; no registration is performed here.
    """
    if not projection_data_dcm or not ct_data_dcm:
        raise ValueError("Projection files and CT slices are required.")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    projections = dicom.load_multibed_projections(projection_data_dcm)
    provenance = []
    generator = None
    reconstructed_beds: List[torch.Tensor] = []
    
    for bed_pos_proj, bed_pos_dcm in zip(projections, projection_data_dcm):
                
        # Read original data
        dicom_ds = pydicom.dcmread(bed_pos_dcm, force=True)
        acquisition = resolve_acquisition(dicom_ds, params)
        window = acquisition["window"]
        index_photopeak = window["Photopeak"]
        energy = acquisition["energy_keV"]
        object_meta, proj_meta = dicom.get_metadata(bed_pos_dcm, index_peak=index_photopeak)
        if generator is None and params.get("Seed") is not None:
            generator = torch.Generator(device=bed_pos_proj.device).manual_seed(params["Seed"])
        fraction = acquisition["time_fraction"]
        photopeak_projections = subsample_projections_time(
            bed_pos_proj[index_photopeak], fraction, generator)
        scatter_projections = torch.zeros_like(photopeak_projections)
        if acquisition["scatter_method"] == "TEW":
            index_lower, index_upper = window["Lower"], window["Upper"]
            lower = subsample_projections_time(bed_pos_proj[index_lower], fraction, generator)
            upper = subsample_projections_time(bed_pos_proj[index_upper], fraction, generator)
            scatter_projections = dicom.compute_EW_scatter(
                projection_lower=lower, projection_upper=upper,
                width_peak=dicom.get_window_width(dicom_ds, index_photopeak),
                width_lower=dicom.get_window_width(dicom_ds, index_lower),
                width_upper=dicom.get_window_width(dicom_ds, index_upper))
            if params.get("Scatter", {}).get("Smooth", params.get("Data_Reduction", {}).get("Smooth_Scatter", False)):
                sigma = params.get("Scatter", {}).get("Sigma_cm", 0.5 / 2.355)
                scatter_projections = get_smoothed_scatter(
                    scatter=scatter_projections, proj_meta=proj_meta, sigma_r=sigma, sigma_z=sigma)
        provenance.append(acquisition)

        # Apply data reduction: subsample projections number
        if "Projections" in params.get("Data_Reduction", {}):
            photopeak_projections, scatter_projections, object_meta, proj_meta = subsample_projections_number(
                photopeak=photopeak_projections, 
                scatter=scatter_projections, 
                object_meta=object_meta, 
                proj_meta=proj_meta,
                parameters=params.get("Data_Reduction", {})["Projections"]
            )
                    
        subsets = params["Algorithm_HyperParameters"].get("n_subsets", 1)
        if not isinstance(subsets, int) or not 1 <= subsets <= photopeak_projections.shape[0]:
            raise ValueError("n_subsets must be between 1 and the retained projection count.")
        provenance[-1]["retained_projections"] = int(photopeak_projections.shape[0])
        # Build System Matrix
        
        # Attenuation
        attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT=ct_data_dcm, file_NM=bed_pos_dcm, index_peak=index_photopeak, E_SPECT=energy)
        att_transform = SPECTAttenuationTransform(attenuation_map)
        att_transform.configure(object_meta=object_meta, proj_meta=proj_meta)
        att_map = att_transform.attenuation_map
        
        # Resolution Model
        psf_meta = dicom.get_psfmeta_from_scanner_params(collimator_name=params["Collimator"], energy_keV=energy, **params.get("PSF_Parameters", {}))
        psf_transform = SPECTPSFTransform(psf_meta)
        
        system_matrix = SPECTSystemMatrix(
            obj2obj_transforms = [att_transform, psf_transform],
            proj2proj_transforms= [],
            object_meta = object_meta,
            proj_meta = proj_meta)
        
        likelihood = PoissonLogLikelihood(system_matrix=system_matrix, projections=photopeak_projections, additive_term=scatter_projections)
        
        # Reconstruct
        if params["Algorithm"] == "OSEM":
            reconstruction_algorithm = OSEM(likelihood)
            
        elif params["Algorithm"] == "BSREM":
            weight_top8anatomy = TopNAnatomyNeighbourWeight(anatomy_image=att_map, N_neighbours=8)
            prior_rdpap = RelativeDifferencePrior(beta=0.3, gamma=2, weight=weight_top8anatomy)
            reconstruction_algorithm = BSREM(
                likelihood,
                prior = prior_rdpap,
                relaxation_sequence = lambda n: 1/(n/50+1))
            
        elif params["Algorithm"] == "KEM":
            kem_params = params["KEM_Parameters"]
            kem_transform = KEMTransform(
                support_objects=[att_map],
                support_kernels_params=kem_params["support_kernels_params"],
                distance_kernel_params=kem_params["distance_kernel_params"],
                top_N=kem_params["top_N"],
                kernel_on_gpu=True
            )
            system_matrix_kem = KEMSystemMatrix(system_matrix, kem_transform)
            likelihood_kem = PoissonLogLikelihood(system_matrix_kem, projections=photopeak_projections, additive_term=scatter_projections)
            reconstruction_algorithm = KEM(likelihood_kem)
        
        elif params["Algorithm"] == "MC_OSEM":
            amap140 = dicom.get_attenuation_map_from_CT_slices(ct_data_dcm, bed_pos_dcm, E_SPECT=140.5) #needed for MC forward proj
            mc_params = params["MC_Parameters"]
            if "isotope_names" not in mc_params or "isotope_ratios" not in mc_params:
                raise ValueError("MC_Parameters requires explicit isotope_names and isotope_ratios supported by SIMIND.")
            energy_window_params = []
            for value in simind_mc.get_energy_window_params_dicom(bed_pos_dcm):
                parts = value.split(',')
                parts[:2] = [str(float(v)) for v in parts[:2]]
                energy_window_params.append(','.join(parts))
            # MC forward projection and analytical back projection
            system_matrix = MonteCarloHybridSPECTSystemMatrix(
                object_meta,
                proj_meta,
                obj2obj_transforms=[att_transform, psf_transform],
                proj2proj_transforms=[],
                attenuation_map_140keV=amap140,
                energy_window_params=energy_window_params,
                primary_window_idx=index_photopeak, 
                isotope_names=mc_params["isotope_names"], #simulate isotope
                isotope_ratios=mc_params["isotope_ratios"], #ratio of isotopes
                collimator_type=params["Collimator"],
                crystal_thickness=mc_params["crystal_thickness"], # cm
                cover_thickness=mc_params["cover_thickness"], # cm
                backscatter_thickness=mc_params["backscatter_thickness"], # cm
                advanced_energy_resolution_model=params["Model"],
                advanced_collimator_modeling=True,
                energy_resolution_140keV=10, # if not using model
                n_events=mc_params["n_events"],
                n_parallel=mc_params["n_parallel"]
            )

            likelihood = MonteCarloHybridSPECTPoissonLogLikelihood(system_matrix, photopeak_projections)
            reconstruction_algorithm = OSEM(likelihood)

        else:
            raise NotImplementedError(f"{params['Algorithm']} is not supported")
        
        reconstructed_beds.append(reconstruction_algorithm(**params["Algorithm_HyperParameters"]))

    # Single-bed studies do not require stitching.
    wb_recon = (reconstructed_beds[0] if len(reconstructed_beds) == 1 else
                dicom.stitch_multibed(recons=torch.stack(reconstructed_beds), files_NM=projection_data_dcm))
    normalization = params.get("Output_Normalization", "raw")
    if normalization not in ("raw", "original_duration"):
        raise ValueError("Output_Normalization must be 'raw' or 'original_duration'.")
    if normalization == "original_duration":
        fractions = [a["time_fraction"] for a in provenance]
        if len(set(fractions)) != 1:
            raise ValueError("Duration normalization requires the same retention fraction for every bed.")
        wb_recon = wb_recon / fractions[0]
    if not torch.isfinite(wb_recon).all() or wb_recon.max() <= 0:
        raise ValueError("Reconstruction must be finite with a positive maximum to export DICOM.")
    ds = dicom.save_dcm(
        save_path=output_path, object=wb_recon.clone(), file_NM=projection_data_dcm[0],
        recon_name=params["Recon_Name"], single_dicom_file=True, return_ds=True,
        scale_by_number_projections=False)
    ds.ImageType = ["DERIVED", "PRIMARY", "RECON TOMO", "EMISSION"]
    ds.SeriesDescription = params["Recon_Name"][:64]
    ds.DerivationDescription = (
        f"Binomial reduced acquisition; target seconds per projection: "
        f"{[a['target_seconds'] for a in provenance]}; normalization: {normalization}")
    output.mkdir(parents=True, exist_ok=False)
    ds.save_as(str(output / "reconstruction.dcm"), little_endian=True, implicit_vr=True)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "pydicom", "pytomography")}
    with (output / "provenance.json").open("w") as f:
        json.dump({"parameters": params, "acquisitions": provenance,
                   "projection_files": projection_data_dcm, "ct_files": ct_data_dcm,
                   "versions": versions, "units": "uncalibrated reconstructed values"}, f, indent=2)
