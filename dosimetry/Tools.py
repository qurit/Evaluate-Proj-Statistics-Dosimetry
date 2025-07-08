from pytheranostics.dosimetry.OrganSDosimetry import OrganSDosimetry
from pytheranostics.dosimetry.VoxelSDosimetry import VoxelSDosimetry
from pytheranostics.ImagingTools.Tools import load_and_resample_RT_to_target, load_and_register_RT_to_target
from pytheranostics.ImagingDS.LongStudy import create_logitudinal_from_dicom
import matplotlib.pyplot as plt
from pathlib import Path
from pytheranostics.dosimetry.olinda import load_phantom_mass
import pandas
from typing import Dict, Any
from copy import deepcopy

import pandas

class DefaultR21Config:
    def __init__(self, mode: str = "Organ-Olinda") -> None:
        
        if mode not in ["Organ-Olinda", "Organ-Voxel"]:
            raise NotImplementedError(f"Mode {mode} not supported")
        
        # Default configuration
            
        self.config =  {
            
            "Operator": "Qurit",
            "PatientID": None,  # <--- Must be filled.
            "Gender": "Male",
            "Cycle": None, # <--- Must be filled.
            "DatabaseDir": None, # <--- Must be filled.
            "InjectionDate": None,   # <--- Must be filled. 
            "InjectionTime": None,   # <--- Must be filled.
            "InjectedActivity": None,  # <--- Must be filled.
            "Radionuclide": "Lu177",
            "ReferenceTimePoint": 0, 
            
            # By default, it will find the best fit order to the data, which in this case, being 2-time points, is mono-exponential.
            # Default A2 parameter is set to Lu-177 decay constant, in hours.
            "rois": {
                'Bladder': {"fit_order": 1, "param_init": {"A2": 0.00434}},  
                'Kidney_Left': {"fit_order": 1, "param_init": {"A2": 0.00434}}, 
                'Kidney_Right': {"fit_order": 1, "param_init": {"A2": 0.00434}}, 
                'Liver': {"fit_order": 1, "param_init": {"A2": 0.00434}}, 
                'ParotidGland_Left': {"fit_order": 1, "param_init": {"A2": 0.00434}}, 
                'ParotidGland_Right': {"fit_order": 1, "param_init": {"A2": 0.00434}}, 
                'SubmandibularGland_Left': {"fit_order": 1, "param_init": {"A2": 0.00434}}, 
                'SubmandibularGland_Right':{"fit_order": 1, "param_init": {"A2": 0.00434}}, 
                'Skeleton': {"fit_order": 1, "param_init": {"A2": 0.00434}}, 
                'Spleen': {"fit_order": 1, "param_init": {"A2": 0.00434}}, 
                'WholeBody': {"fit_order": 1, "param_init": {"A2": 0.00434}}, 
                'RemainderOfBody':{"fit_order": 1, "param_init": {"A2": 0.00434}}, 
                    },
            
            "Level": "Organ", 
            "Method": "S-Value", 
            "OutputFormat": "Olinda",
            "SalivaryGlandsSeparately": "Yes",
            "LesionDosimetry": "No"
            
        } 
        
        if mode == "Organ-Voxel":
            
            self.config["Method"] = "Voxel-S-Value"
            self.config["OutputFormat"] = "Dose-Map"
        
        return None

def read_recon_params_from_name(recon_file_name: str) -> Dict[str, Any]:
    """Reads the parameters associated with a reconstruction file with the filename format:
       `OSEM_24_8_Time_1.0_Proj_1_SctSmooth_True`

    Parameters
    ----------
    recon_file_name : str
        The reconstruction file name.

    Returns
    -------
    Dict[str, Any]
        A dictionary with the reconstruction parameters.
    """
    params = recon_file_name.split("_")

    return {
        "Time_Factor": float(params[4]), 
        "Projection_Factor": int(params[3]),
        "Smooth_Scatter": True if params[5] == "True" else False,
        "Algorithm": params[0],
        "n_iters": int(params[1]),
        "n_subsets": int(params[2])
           }
    
def run_patient_dosimetry_recons_batch(ct_paths: Dict[int, str], 
                                       rtstr_paths: Dict[int, str], 
                                       recons_paths: Dict[str, Dict[int, str]],
                                       output_dir: str,
                                       dosimetry_config: DefaultR21Config) -> pandas.DataFrame:
    """Run patient Dosimetry over a cycle of a patient and a batch of SPECT reconstructions

    Parameters
    ----------
    ct_paths : Dict[int, str]
        _description_
    rtstr_paths : Dict[int, str]
        _description_
    recons_path : Dict[str, Dict[int, str]]
        _description_
    output_dir : str
        _description_
    dosimetry_config: DefaultPR21Config
        _description_

    Returns
    -------
    pandas.DataFrame
        _description_
    """
    
    # General Parameters for acquired dosimetry PR-21 data
    vox_vol_ml = (4.79519987 / 10) ** 3
    lu177_cf = 9.26 # CPS/MBq Symbia T series at VGH, MELP Collimator with PR-21 Acquisition parameters (energy window settings).
    frame_duration = 15 # 15 seconds.
    num_projections = 96  # 96 per the protocol, 48 per detector head

    # General mapping of mask-names for acquired dosimetry PR-21 data
    # Convert masks names to standard naming convention in pyTheranostics.
    ct_mask_mapping = {
        "Liver": "Liver",
        "Kidney_L_m": "Kidney_Left",
        "Kidney_R_m": "Kidney_Right",
        "Spleen": "Spleen",
        "Bladder": "Bladder",
        "Parotid_L_m": "ParotidGland_Left",
        "Parotid_R_m": "ParotidGland_Right",
        "Submandibular_L_m": "SubmandibularGland_Left",
        "Submandibular_R_m": "SubmandibularGland_Right",
        "Skeleton": "Skeleton",                           
        "WBCT": "WholeBody"
                        }
    
    spect_mask_mapping = {
        "Liver": "Liver",
        "Kidney_L_a": "Kidney_Left",
        "Kidney_R_a": "Kidney_Right",
        "Spleen": "Spleen",
        "Bladder": "Bladder",
        "Parotid_L_a": "ParotidGland_Left",
        "Parotid_R_a": "ParotidGland_Right",
        "Submandibular_L_a": "SubmandibularGland_Left",
        "Submandibular_R_a": "SubmandibularGland_Right",
        "Skeleton": "Skeleton",                    
        "WBCT": "WholeBody"
                            }


    # Initialize results DataFrame
    results = pandas.DataFrame()
    
    # Create Longitudinal CT Study
    ct_paths_list = [path for _, path in ct_paths.items()]
    rtstr_paths_list = [path for _, path in rtstr_paths.items()]
    
    longCT = create_logitudinal_from_dicom(dicom_dirs=ct_paths_list, modality="CT")
    
    # Loop through available SPECT Reconstruction methods
    for recon_name, spect_paths in recons_paths.items():
        
        recon_parameters = read_recon_params_from_name(recon_file_name=recon_name)
        
        print("Recon Parameters:")
        print(recon_parameters)
        
        # We need to feed the calibration factor so that the reconstructed image, which is in units of Counts * Num_Projections is converted to units of activity.
        num_proj_recon = 1 / recon_parameters["Projection_Factor"] * num_projections
        proj_time = recon_parameters["Time_Factor"] * frame_duration
        
        raw_counts_to_bq_per_ml = 1 / (proj_time * num_proj_recon) * lu177_cf * 1e6 * 1 / vox_vol_ml

        # Create Longitudinal SPECT Study
        spect_paths_list = [spect_path for _, spect_path in spect_paths.items()]
        
        longSPECT = create_logitudinal_from_dicom(dicom_dirs=spect_paths_list,
                                                  modality="Lu177_SPECT", 
                                                  calibration_factor=raw_counts_to_bq_per_ml)
        
        # QA: Ensure we have as many time-points in SPECT as in CT
        if len(longSPECT.images) != len(longCT.images):
            raise AssertionError("Inconsistent number of CT and SPECT time-points.")
        
        # Loop through time-points and process RTStruct masks
        time_id = 0
        
        for ct_img_dir, rt_struct_dir in zip(ct_paths_list, rtstr_paths_list):

            # Get RT_Struct path to file (should be single file inside this folder)
            rt_struct_tmp = list(Path(rt_struct_dir).glob("*.dcm"))
            assert len(rt_struct_tmp) == 1
            rt_struct_path = str(rt_struct_tmp[0])
            
            ct_masks, nm_masks = load_and_resample_RT_to_target(ref_dicom_ct_dir=ct_img_dir, 
                                                                rt_struct_file=rt_struct_path, 
                                                                target_img=longSPECT.images[time_id])
            
            longCT.add_masks_to_time_point(time_id=time_id, masks=ct_masks, mask_mapping=ct_mask_mapping)
            longSPECT.add_masks_to_time_point(time_id=time_id, masks=nm_masks, mask_mapping=spect_mask_mapping)
            
            # QA: Save CT, SPECT and Masks for visualization
            # TODO:
            
            
            time_id += 1 
        
        
        # QA: ensure mandatory info is present in dosimetry config:
        #     PatientID, Cycle, Output Dir, Injection Dates and Time...
        
        dosimetry_config.config["DatabaseDir"] = output_dir
        
        if dosimetry_config.config["InjectedActivity"] is None:
            dosimetry_config.config["InjectedActivity"] = longSPECT.meta[0]["Injected_Activity_MBq"]
        
        # If no information about administered activity time, assume equal to imaging time.
        
        if dosimetry_config.config["InjectionDate"] is None:
            
            print("Missing Information about administered Activity. Assuming 7400 MBq administered at imaging time.")
            
            dosimetry_config.config["InjectionDate"] = longSPECT.meta[0]["AcquisitionDate"]
            dosimetry_config.config["InjectionTime"] = longSPECT.meta[0]["AcquisitionTime"]
        
        
        if (
            dosimetry_config.config["PatientID"] is None or 
            dosimetry_config.config["Cycle"] is None
            ):
            raise ValueError("Dosimetry config incomplete:", dosimetry_config.config)
        
        # Instantiate Dosimetry Calculator
        if dosimetry_config.config["Method"] == "S-Value":
                        
            dose_calculator = OrganSDosimetry(
                config=dosimetry_config.config,
                nm_data=longSPECT,
                ct_data=longCT
                )
                        
        elif dosimetry_config.config["Method"] == "Voxel-S-Value":
            
            dose_calculator = VoxelSDosimetry(
                config=dosimetry_config.config,
                nm_data=longSPECT,
                ct_data=longCT
                )
            
        else:
            raise ValueError(f"{dosimetry_config.config['Method']} not supported.")
        
        # Currently, we only compute Time-integrated Activity.
        dose_calculator.compute_tia()
            
        # Aggregate Results
        
        
    return results

