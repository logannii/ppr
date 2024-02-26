import os
import os.path as op
import numpy as np
import nibabel as nib
import regtricks as rt
import h5py
import logging
import argparse
from glob import glob

LOGGING_LEVEL = logging.DEBUG

def _calc_norm_factor(oxdir, model_dict, fsdir=None):
    """Calculates the normalisation factor for the subject's ASL data.

    Parameters
    ----------
    oxdir : str
        The oxasl directory containing the subject's processed ASL data.
    model_dict : dict
        The model dictionary.
    fsdir : str, optional
        The FreeSurfer directory containing the subject's structural data.

    Returns
    -------
    float
        The normalisation factor.
    """

    if model_dict["iscalib"]:
        iscalib = "/calib_voxelwise"
        logging.debug("Model setting: calibrated. Finding perfusion image after calibration")
    else:
        iscalib = ""
        logging.debug("Model setting: uncalibrated. Finding perfusion image without calibration")
    if ~model_dict["isnorm"]:
        logging.debug("Model setting: not normalised. Returning normalisation factor 1.0")
        return 1.0
    if model_dict["norm_ispvcorr"]:
        ispvcorr = "_pvcorr"
        logging.debug("Model setting: using partial volume corrected perfusion image to calculate normalisation factor")
    else:
        ispvcorr = ""
        logging.debug("Model setting: using un-partial volume corrected perfusion image to calculate normalisation factor")

    # Load the ASL data
    perf_file = op.join(oxdir, f"output{ispvcorr}/struc{iscalib}/perfusion.nii.gz")
    if not op.exists(perf_file):
        logging.error(f"Perfusion image not found at {perf_file}")
        raise FileNotFoundError(f"Perfusion image not found at {perf_file}")
    perf = nib.load(perf_file).get_fdata()
    logging.debug(f"Loaded perfusion image from {perf_file}")

    # Load the pvgm data
    pvgm_file = op.join(oxdir, "structural/gm_pv.nii.gz")
    if not op.exists(pvgm_file):
        logging.error(f"GM partial volume map not found at {pvgm_file}")
        raise FileNotFoundError(f"GM partial volume map not found at {pvgm_file}")
    pvgm = nib.load(pvgm_file).get_fdata()
    logging.debug(f"Loaded GM partial volume map from {pvgm_file}")

    # Load freesurfer ribbon mask (if provided)
    if fsdir is not None:
        logging.debug(f"Using FreeSurfer directory to find ribbon mask at {fsdir}")
        ribbon_file = op.join(fsdir, "mri/ribbon.mgz")
        if not op.exists(ribbon_file):
            logging.error(f"Ribbon mask not found at {ribbon_file}")
            raise FileNotFoundError(f"Ribbon mask not found at {ribbon_file}")
        ribbon = rt.Registration.identity().apply_to_image(glob(ribbon_file)[0], perf, order=1)
        logging.debug(f"Loaded ribbon mask from {ribbon_file}")
        mask = (pvgm > model_dict["norm_thr"]) & (ribbon.get_fdata() > 0)
    else:
        logging.debug("No FreeSurfer directory provided. Using oxasl brain mask")
        mask = (pvgm > model_dict["norm_thr"]) & (perf > 0)

    # Calculate the normalisation factor
    norm_factor = perf[mask].mean()
    logging.debug(f"Calculated normalisation factor: {norm_factor}")
    return norm_factor


def predict(oxdir, modeldir, age, gender, space="native"):
    """Predicts the baseline perfusion image for a given subject. 

    Parameters
    ----------
    oxdir : str
        The oxasl directory containing the subject's processed ASL data.
    modeldir : str
        The path to the trained model.
    age : float
        The subject's age in years
    gender: str
        The gender of the subject. Either 'M' or 'F'
    space : str, optional
        The space of the output image. Either 'native' 'struct' or 'std'.
    """

    outdir = op.join(oxdir, "ppr", op.basename(modeldir).split(".")[0], space)
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(
        filename=op.join(outdir, "ppr.log"),
        level=LOGGING_LEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
        # handlers=[
        #     logging.FileHandler("ppr.log"),
        #     logging.StreamHandler()
        # ]
    )
    logging.info("Running personalised perfusion reference (PPR) script")
    logging.info(f"Output directory: {outdir}")

    logging.info(f"Loading PPR model from {modeldir}")
    with h5py.File(modeldir, "r") as f:
        model_dict = {}
        for k in f.keys():
            if isinstance(f[k], h5py.Group):
                model_dict[k] = {}
                for k2 in f[k].keys():
                    model_dict[k][k2] = f[k][k2][()]
            else:
                model_dict[k] = f[k][()]

    norm_factor = _calc_norm_factor(oxdir, model_dict)
    logging.debug(f"Input age: {age} years")
    age = age - model_dict["age_transform"]
    logging.debug(f"Transformed age: {age} years")
    logging.debug(f"Input gender {gender} (M/F)")
    gender = model_dict["gender_transform"][gender]
    logging.debug(f"Transformed gender: {gender} (0/1)")

    iscalib = "/calib_voxelwise" if model_dict["iscalib"] else ""

    logging.debug(f"Loading images from {oxdir}")
    native_spc = rt.ImageSpace(op.join(oxdir, "reg/aslref.nii.gz"))
    struct_spc = rt.ImageSpace(op.join(oxdir, "reg/strucref.nii.gz"))
    std_spc = rt.ImageSpace(op.join(oxdir, "reg/stdref.nii.gz"))

    logging.debug(f"Loading model parameters")
    beta_std = model_dict["params_array"]

    if space == "native":
        logging.info("Predicting baseline perfusion image in native space")
        target_spc = native_spc
        std2struct = rt.NonLinearRegistration.from_fnirt(
            op.join(oxdir, "reg/std2struc.nii.gz"),
            src=std_spc,
            ref=struct_spc,
            intensity_correct=False,
        )
        struct2asl = rt.Registration.from_flirt(
            op.join(oxdir, "reg/struc2asl.mat"),
            src=struct_spc,
            ref=native_spc,
        )
        std2asl = rt.chain(std2struct, struct2asl)

        beta = std2asl.apply_to_array(
            beta_std, 
            src=std_spc, 
            ref=native_spc, 
            cores=1
        )
        pvgm = nib.load(op.join(oxdir, "structural/gm_pv_asl.nii.gz")).get_fdata()
        pvwm = nib.load(op.join(oxdir, "structural/wm_pv_asl.nii.gz")).get_fdata()
        truth = nib.load(op.join(oxdir, f"output/native{iscalib}/perfusion.nii.gz")).get_fdata() / norm_factor

    elif space == "struct":
        logging.info("Predicting baseline perfusion image in structural space")
        target_spc = struct_spc
        std2struct = rt.NonLinearRegistration.from_fnirt(
            op.join(oxdir, "reg/std2struc.nii.gz"),
            src=std_spc,
            ref=struct_spc,
            intensity_correct=False,
        )
        asl2struct = rt.Registration.from_flirt(
            op.join(oxdir, "reg/asl2struc.mat"),
            src=native_spc,
            ref=struct_spc,
        )

        beta = std2struct.apply_to_array(
            beta_std, 
            src=std_spc, 
            ref=struct_spc, 
            cores=1
        )
        pvgm = asl2struct.apply_to_array(
            nib.load(op.join(oxdir, "structural/gm_pv_asl.nii.gz")).get_fdata(), 
            src=native_spc, 
            ref=struct_spc, 
            cores=1
        )
        pvwm = asl2struct.apply_to_array(
            nib.load(op.join(oxdir, "structural/wm_pv_asl.nii.gz")).get_fdata(),
            src=native_spc,
            ref=struct_spc,
            cores=1
        )
        truth = nib.load(op.join(oxdir, f"output/struc{iscalib}/perfusion.nii.gz")).get_fdata() / norm_factor

    elif space == "std":
        logging.info("Predicting baseline perfusion image in standard space")
        target_spc = std_spc
        asl2struct = rt.Registration.from_flirt(
            op.join(oxdir, "reg/asl2struc.mat"),
            src=native_spc,
            ref=struct_spc,
        )
        struct2std = rt.NonLinearRegistration.from_fnirt(
            op.join(oxdir, "reg/struc2std.nii.gz"),
            src=struct_spc,
            ref=std_spc,
            intensity_correct=False,
        )
        asl2std = rt.chain(asl2struct, struct2std)

        beta = beta_std
        pvgm = asl2std.apply_to_array(
            nib.load(op.join(oxdir, "structural/gm_pv_asl.nii.gz")).get_fdata(),
            src=native_spc,
            ref=std_spc,
            cores=1
        )
        pvwm = asl2std.apply_to_array(
            nib.load(op.join(oxdir, "structural/wm_pv_asl.nii.gz")).get_fdata(),
            src=native_spc,
            ref=std_spc,
            cores=1
        )
        truth = asl2std.apply_to_array(
            nib.load(op.join(oxdir, f"output/native{iscalib}/perfusion.nii.gz")).get_fdata(),
            src=native_spc,
            ref=std_spc,
            cores=1
        ) / norm_factor

    else:
        logging.error(f"Space {space} not recognised")
        raise ValueError(f"Space {space} not recognised")

    logging.debug("Generating baseline perfusion prediction")
    prediction = (
        beta[..., 0]
        + beta[..., 1] * age
        + beta[..., 2] * gender 
        + (
            beta[..., 3] 
            + beta[..., 4] * age
            + beta[..., 5] * gender
        ) 
        * pvgm
        + (
            beta[..., 6] 
            + beta[..., 7] * age
            + beta[..., 8] * gender
        ) 
        * pvwm
    )
    logging.debug("Generated baseline perfusion prediction successfully")

    target_spc.save_image(prediction, op.join(outdir, "prediction.nii.gz"))
    logging.info(f"Predicted baseline perfusion image saved to {op.join(outdir, 'prediction.nii.gz')}")
    target_spc.save_image(truth, op.join(outdir, "truth.nii.gz"))
    logging.info(f"Ground truth perfusion image saved to {op.join(outdir, 'truth.nii.gz')}")
    difference = truth - prediction
    target_spc.save_image(difference, op.join(outdir, "difference.nii.gz"))
    logging.info(f"Difference image saved to {op.join(outdir, 'difference.nii.gz')}")
    logging.info("Prediction done")