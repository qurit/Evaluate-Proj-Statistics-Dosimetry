import os
from pytomography.io.SPECT import dicom
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
import pydicom
from pydicom import FileDataset
import torch

from typing import List, Dict, Tuple


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
    proj_dcm = pydicom.dcmread(projection_data_dcm[0])
    
    for i, energy, win_name in zip([index_photopeak, index_lower, index_upper], [208, 180, 235], ["Peak", "Lower", "Upper"]):
        low, high = dicom.get_energy_window_bounds(projection_data_dcm[0], idx=i)
        if energy > high or energy < low:
            raise AssertionError(f"Energy Window = ({low}, {high})  keV does not correspond to {win_name}")
   
    reconstructed_beds: List[torch.Tensor] = []
    
    for bed_pos_proj, bed_pos_dcm in zip(projections, projection_data_dcm):
        
        object_meta, proj_meta = dicom.get_metadata(bed_pos_dcm, index_peak=index_photopeak)
        photopeak_projections = bed_pos_proj[index_photopeak]
        scatter_projections = dicom.get_energy_window_scatter_estimate_projections(
            file=bed_pos_dcm, projections=bed_pos_proj, index_peak=index_photopeak, index_lower=index_lower, index_upper=index_upper
        )
        
        # Build System Matrix
        
        # Attenuation
        attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT=ct_data_dcm, file_NM=bed_pos_dcm, index_peak=index_photopeak)
        att_transform = SPECTAttenuationTransform(attenuation_map)
        
        # Resolution Model
        psf_meta = dicom.get_psfmeta_from_scanner_params(collimator_name="SY-ME", energy_keV=208)
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
        else:
            raise NotImplementedError(f"{params['Algorithm']} is not supported")
        
        reconstructed_beds.append(reconstruction_algorithm(**params["Algorithm_HyperParameters"]))
        
    # Stitch Reconstructions
    wb_recon = dicom.stitch_multibed(recons=torch.stack(reconstructed_beds), files_NM=projection_data_dcm)
    
    # Save DICOM
    dicom.save_dcm(
        save_path=output_path,
        object=wb_recon,
        file_NM = projection_data_dcm[0],
        recon_name="test",
        single_dicom_file=True)
        
    