import argparse
from pathlib import Path

import numpy as np
from nilearn import connectome, datasets, maskers
from src import fmriprep


def main(args):

    data = fmriprep.Data(args.preproc, args.denoise_strategy)
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=1000, resolution_mm=2)["maps"]

    masker = maskers.NiftiLabelsMasker(
        atlas,
        mask_img=data.mask,
        smoothing_fwhm=args.smooth_fwhm,
        standardize="zscore",
        standardize_confounds=True,
        t_r=data.tr,
        strategy="mean",
    )

    corr_model = connectome.ConnectivityMeasure(
        kind="correlation", vectorize=True, discard_diagonal=True
    )  # LedoitWolf covariance estimator

    roi_timeseries = masker.fit_transform(
        imgs=data.preproc, confounds=data.confounds, sample_mask=data.sample_mask
    )

    corr_coefficients = corr_model.fit_transform([roi_timeseries])[0]
    corr_z_transform = np.arctanh(corr_coefficients)  # fisher z-transform
    np.save(args.output, corr_z_transform)


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
