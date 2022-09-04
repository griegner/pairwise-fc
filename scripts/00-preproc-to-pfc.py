import argparse
from pathlib import Path

import numpy as np
from nilearn import datasets, maskers
from src import fmriprep, pairwise_fc


def main(args):

    data = fmriprep.Data(args.preproc, args.denoise_strategy)
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)["maps"]

    masker = maskers.NiftiLabelsMasker(
        atlas,
        mask_img=data.mask,
        smoothing_fwhm=args.smooth_fwhm,
        standardize="zscore",
        standardize_confounds=True,
        t_r=data.tr,
        strategy="mean",
    )

    roi_timeseries = masker.fit_transform(
        imgs=data.preproc, confounds=data.confounds, sample_mask=data.sample_mask
    )

    dm = data.make_design_matrix(hrf_model="spm + derivative", drop_constant=True)
    dm["stim"] = dm["stim"] - np.mean(
        [dm["stim"].min(), dm["stim"].max()]
    )  # zero center

    ppi = pairwise_fc.PPI(vectorize=True, fit_intercept=False, n_jobs=args.n_jobs)
    ppi_coefs = ppi.fit_transform(roi_timeseries, dm)

    np.save(args.output, ppi_coefs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="statistical pairwise interaction models of fMRI timeseries"
    )
    parser.add_argument("preproc", type=str, help="input fMRIprep preprocessed data")
    parser.add_argument(
        "output", type=str, help="output pairwise interaction numpy matrix"
    )
    parser.add_argument(
        "denoise_strategy",
        type=str,
        help=" - ".join(fmriprep.denoise_strategies.keys()),
    )
    parser.add_argument(
        "--smooth_fwhm",
        type=float,
        default=None,
        help="smoothing kernel in mm",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="number of jobs")

    args = parser.parse_args()

    assert Path(args.preproc).exists(), f"{args.preproc} does not exist"
    assert (
        args.denoise_strategy in fmriprep.denoise_strategies.keys()
    ), f"{args.denoise_strategy} is not a valid strategy"

    main(args)
