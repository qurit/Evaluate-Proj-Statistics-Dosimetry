from pytomography.io.SPECT import dicom
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.priors import RelativeDifferencePrior
from pytomography.priors import TopNAnatomyNeighbourWeight
from pytomography.algorithms import OSEM, BSREM, KEM
from pytomography.transforms.shared import KEMTransform
from pytomography.projectors.shared import KEMSystemMatrix
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
import pydicom
import torch
from typing import List, Dict
from reconstruction.Subsample import subsample_projections_number, subsample_projections_time
from pytomography.utils.scatter import get_smoothed_scatter

def reconstruct_study(projection_data_dcm: List[str], ct_data_dcm: List[str], params: Dict, output_path: str) -> None:
    """Reconstruct multi-bed SPECT projection data utilizing reconstruction parameters defined by the 
    dictionary parameters.

    Parameters
    ----------
    projection_data_dcm : List[str]
        List of paths to projection .dcm files, one for each bed. List[str]
    ct_data_dcm : List[str]
        List of paths to CT .dcm files, one for each slice. List[str]
    params : Dict
        Dictionary containing reconstruction parameters: Algorithm, Algorithm_HyperParameters, Projections_Fraction, Time_Proj_Fraction
    output_path : Path
        Path where reconstructed SPECT image will be stored.
    """
    
    # Define Photopeak, Lower and Upper energy window indices.
    index_photopeak = 0
    index_lower = 1
    index_upper = 2
    
    # Load Projections
    projections = dicom.load_multibed_projections(projection_data_dcm)
    
    # Confirm Energy Window Settings    
    for i, energy, win_name in zip([index_photopeak, index_lower, index_upper], [208, 180, 235], ["Peak", "Lower", "Upper"]):
        low, high = dicom.get_energy_window_bounds(projection_data_dcm[0], idx=i)
        if energy > high or energy < low:
            raise AssertionError(f"Energy Window = ({low}, {high})  keV does not correspond to {win_name}")
   
    reconstructed_beds: List[torch.Tensor] = []
    
    for bed_pos_proj, bed_pos_dcm in zip(projections, projection_data_dcm):
                
        # Read original data
        dicom_ds = pydicom.dcmread(bed_pos_dcm, force=True)
        object_meta, proj_meta = dicom.get_metadata(bed_pos_dcm, index_peak=index_photopeak)
        photopeak_projections = bed_pos_proj[index_photopeak]
        upper_projections = bed_pos_proj[index_upper]
        lower_projections = bed_pos_proj[index_lower]
        
        # Apply Projection Time Reduction (Poisson thinning):
        if "Time" in params["Data_Reduction"]:
            
            photopeak_projections = subsample_projections_time(projections=photopeak_projections, t_reduction_factor=params["Data_Reduction"]["Time"])
            upper_projections = subsample_projections_time(projections=upper_projections, t_reduction_factor=params["Data_Reduction"]["Time"])
            lower_projections = subsample_projections_time(projections=lower_projections, t_reduction_factor=params["Data_Reduction"]["Time"])
        
        # Estimate Scatter using TEW
        scatter_projections = dicom.compute_EW_scatter(
            projection_lower=lower_projections,
            projection_upper=upper_projections,
            width_peak=dicom.get_window_width(dicom_ds, index_photopeak),
            width_lower=dicom.get_window_width(dicom_ds, index_lower),
            width_upper=dicom.get_window_width(dicom_ds, index_upper)
        )
        
        if params["Data_Reduction"]["Smooth_Scatter"]:
            scatter_projections = get_smoothed_scatter(scatter=scatter_projections, proj_meta=proj_meta, sigma_r=1.0, sigma_z=1.0)
        
        # Apply data reduction: subsample projections number
        if "Projections" in params["Data_Reduction"]:
            photopeak_projections, scatter_projections, object_meta, proj_meta = subsample_projections_number(
                photopeak=photopeak_projections, 
                scatter=scatter_projections, 
                object_meta=object_meta, 
                proj_meta=proj_meta,
                parameters=params["Data_Reduction"]["Projections"]
            )
                    
        # Build System Matrix
        
        # Attenuation
        attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT=ct_data_dcm, file_NM=bed_pos_dcm, index_peak=index_photopeak)
        att_transform = SPECTAttenuationTransform(attenuation_map)
        att_transform.configure(object_meta=object_meta, proj_meta=proj_meta)
        att_map = att_transform.attenuation_map
        
        # Resolution Model
        psf_meta = dicom.get_psfmeta_from_scanner_params(collimator_name=params["Collimator"], energy_keV=params["Photopeak_Energy"])
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
        
        else:
            raise NotImplementedError(f"{params['Algorithm']} is not supported")
        
        reconstructed_beds.append(reconstruction_algorithm(**params["Algorithm_HyperParameters"]))
        
    # Stitch Reconstructions
    wb_recon = dicom.stitch_multibed(recons=torch.stack(reconstructed_beds), files_NM=projection_data_dcm)
    
    # Save DICOM
    print(f"Saving: {params['Recon_Name']} ... ")
    
    dicom.save_dcm(
        save_path=output_path,
        object=wb_recon,
        file_NM = projection_data_dcm[0],
        recon_name=params["Recon_Name"],
        single_dicom_file=True)
        
    
