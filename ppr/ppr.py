import os
import os.path as op
import numpy as np
import nibabel as nib
import regtricks as rt
import h5py


def _calc_norm_factor(oxdir, model_dict):
    """Calculates the normalisation factor for the subject's ASL data.

    Parameters
    ----------
    oxdir : str
        The oxasl directory containing the subject's processed ASL data.

    Returns
    -------
    float
        The normalisation factor.
    """

    if ~model_dict["isnorm"]:
        return 1.0
    iscalib = "/calib_voxelwise" if model_dict["iscalib"] else ""
    ispvcorr = "_pvcorr" if model_dict["norm_ispvcorr"] else ""

    # Load the ASL data
    # TODO: Check if the file exists
    perf = nib.load(op.join(oxdir, f"output{ispvcorr}/struc{iscalib}/perfusion.nii.gz")).get_fdata()

    # Load the pvgm data
    # TODO: Check if the file exists
    pvgm = nib.load(op.join(oxdir, f"structural/gm_pv.nii.gz")).get_fdata()

    # Calculate the normalisation factor
    norm_factor = perf[(pvgm > model_dict["norm_thr"]) & (perf > 0)].mean()

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
    age = age - model_dict["age_transform"]
    gender = model_dict["gender_transform"][gender]

    iscalib = "/calib_voxelwise" if model_dict["iscalib"] else ""

    outdir = op.join(oxdir, "ppr", model_dict["name"].decode("utf-8"), space)
    os.makedirs(outdir, exist_ok=True)

    native_spc = rt.ImageSpace(op.join(oxdir, "reg/aslref.nii.gz"))
    struct_spc = rt.ImageSpace(op.join(oxdir, "reg/strucref.nii.gz"))
    std_spc = rt.ImageSpace(op.join(oxdir, "reg/stdref.nii.gz"))

    beta_std = model_dict["params_array"]

    if space == "native":
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
    target_spc.save_image(prediction, op.join(outdir, "prediction.nii.gz"))
    target_spc.save_image(truth, op.join(outdir, "truth.nii.gz"))
    residual = truth - prediction
    target_spc.save_image(residual, op.join(outdir, "residual.nii.gz"))