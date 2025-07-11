from pytheranostics.dosimetry.OrganSDosimetry import OrganSDosimetry
from pytheranostics.dosimetry.VoxelSDosimetry import VoxelSDosimetry
from pytheranostics.ImagingTools.Tools import load_and_resample_RT_to_target
from pytheranostics.ImagingDS.LongStudy import create_logitudinal_from_dicom
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List

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
       OSEM_i_20_s_4_Time_1_Proj_3_Start_0_SctSmooth_True_TP1

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
    """['OSEM', 'i', '20', 's', '4', 'Time', '1', 'Proj', '3', 'Start', '0', 'SctSmooth', 'True', 'TP1']"""

    return {
        "Algorithm": params[0],
        "n_iters": int(params[2]),
        "n_subsets": int(params[4]),
        "Time_Reduction_Factor": float(params[6]), 
        "Projection_Reduction_Factor": int(params[8]),
        "Smooth_Scatter": True if params[12] == "True" else False,
        "TP": params[13]
           }

def get_injection_info_from_excel(excel_path: str, patient_id: str) -> dict[str, Any]:
    """Reads injected activity, injection date, and time from an Excel file
    
    Parameters
    ----------
    excel_path : str
        Path to the excel file
    patient_id : str
        Patient ID to get matching information
    
    Returns
    -------
    dict[str,Any]
        a dictionary with 'InjectedActivity', 'InjectionDate', 'InjectionTime'
    """
    df = pd.read_excel(excel_path)

    row = df[df["PatientID"] == patient_id]

    if row.empty:
        raise ValueError(f"PatientID '{patient_id}' missing.")

    # Parse Injection Date
    injection_date_raw = row["Injection Date"].values[0]
    injection_date = pd.to_datetime(injection_date_raw).date()

    # Parse Injection Time with AM/PM handling
    injection_time_raw = row["Injection Time"].values[0]
    # If it's already datetime.time, just use it
    if isinstance(injection_time_raw, pd.Timestamp):
        injection_time = injection_time_raw.time()
    elif isinstance(injection_time_raw, str):
        injection_time = pd.to_datetime(injection_time_raw).time()
    else:
        # Otherwise, try to convert with pd.to_datetime and then get time
        injection_time = pd.to_datetime(str(injection_time_raw)).time()

    return {
        "InjectedActivity": float(row["Injected Activity [MBq]"].values[0]),
        "InjectionDate": injection_date.strftime("%Y%m%d"),
        "InjectionTime": injection_time.strftime("%H%M%S"), 
    }

def build_dosimetry_config_from_excel(excel_path, patient_id):
    config = DefaultR21Config()
    info = get_injection_info_from_excel(excel_path, patient_id)
    config.config["InjectedActivity"] = info["InjectedActivity"]
    config.config["InjectionDate"] = info["InjectionDate"]
    config.config["InjectionTime"] = info["InjectionTime"]
    config.config["PatientID"] = patient_id
    config.config["Cycle"] = 1
    return config

 
def reformat_dataframe(df: pd.DataFrame, patient_id: str, algo_params: Dict[str, Any]) -> pd.DataFrame:
    """Transforms a DataFrame indexed by region into a tidy format with columns for:
        - Region
        - Patient ID
        - Algorithm parameters 
    Parameters
    ----------
    df : pandas.DataFrame
        _description_
    patient_id : str
        _description_
    algo_params : Dict[str, Any]
        _description_

    Returns
    -------
    pandas.DataFrame
        The reformated dataframe
    """
    df_tidy = df.reset_index().rename(columns={"index": "Region"})
    df_tidy["PatientID"] = patient_id
    
    new_cols = ["PatientID", "Algorithm", "n_iters", "n_subsets", "Time_Reduction_Factor", "Projection_Reduction_Factor", "Smooth_Scatter"]
    for parameter in new_cols:
        if parameter != "PatientID":
            df_tidy[parameter] = algo_params[parameter]

    # Reorder:
    cols = df_tidy.columns.to_list()
    new_cols_order = new_cols + [col for col in cols if col not in new_cols]
    
    return df_tidy[new_cols_order]

    
def run_patient_dosimetry_recons_batch(ct_paths: Dict[int, str], 
                                       rtstr_paths: Dict[int, str], 
                                       recons_paths: Dict[str, Dict[int, str]],
                                       output_dir: str,
                                       dosimetry_config: DefaultR21Config) -> pd.DataFrame:
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
    results: List[pd.DataFrame] = []
    
    # Create Longitudinal CT Study
    ct_paths_list = [path for _, path in ct_paths.items()]
    rtstr_paths_list = [path for _, path in rtstr_paths.items()]
    
    longCT = create_logitudinal_from_dicom(dicom_dirs=ct_paths_list, modality="CT")
    
    # Loop through available SPECT Reconstruction methods
    for recon_name, spect_paths in recons_paths.items():
        
            # Get any time point folder name with _TPx suffix
        first_tp_path = next(iter(spect_paths.values()))
        folder_name = Path(first_tp_path).name  # get folder basename with _TPx suffix
    
        recon_parameters = read_recon_params_from_name(recon_file_name=folder_name)
        
        # Define output path to store results
        output_dir_recon = Path(output_dir) / f"{recon_name}"
        output_dir_recon.mkdir(parents=True, exist_ok=True)
        
        print("Recon Parameters:")
        print(recon_parameters)
        
        # We need to feed the calibration factor so that the reconstructed image, which is in units of Counts * Num_Projections is converted to units of activity.
        num_proj_recon = 1 / recon_parameters["Projection_Reduction_Factor"] * num_projections
        proj_time = recon_parameters["Time_Reduction_Factor"] * frame_duration
        
        raw_counts_to_bq_per_ml = 1 / (proj_time * num_proj_recon) * lu177_cf * 1e6 * 1 / vox_vol_ml

        # Create Longitudinal SPECT Study
        spect_paths_list = [spect_path for _, spect_path in spect_paths.items()]
        
        longSPECT = create_logitudinal_from_dicom(dicom_dirs=spect_paths_list,
                                                  modality="Lu177_SPECT", 
                                                  calibration_factor=raw_counts_to_bq_per_ml,
                                                  dosimetry_config=dosimetry_config)
        
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
            
            # QA: Masks for visualization
            longSPECT.save_masks_to_nii_at(time_id=time_id, out_path=output_dir_recon, 
                                            regions=["Liver", "Kidney_Left", "Kidney_Right",
                                                    "ParotidGland_Left", "ParotidGland_Right", 
                                                    "SubmandibularGland_Left", "SubmandibularGland_Right",
                                                    "Spleen",
                                                    "Skeleton"])
            
            time_id += 1 
        
        
        # QA: ensure mandatory info is present in dosimetry config:
        #     PatientID, Cycle, Output Dir, Injection Dates and Time...
        
        dosimetry_config.config["DatabaseDir"] = str(output_dir_recon)
        
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
        tmp_results = reformat_dataframe(df=dose_calculator.results.copy(), 
                                         patient_id=dosimetry_config.config["PatientID"],
                                         algo_params=recon_parameters)
        results.append(tmp_results)
        
    return pd.concat(results, ignore_index=True)

