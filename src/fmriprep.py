from pathlib import Path

import pandas as pd
from nilearn.interfaces import bids, fmriprep
from nilearn import image

denoise_strategies = {
    "HMPWMCSFScrub": dict(
        strategy=("motion", "high_pass", "wm_csf", "scrub"),
        motion="basic",
        wm_csf="derivatives",
        scrub=0,
        fd_threshold=0.5,
        std_dvars_threshold=3.0,
    ),
    "HMPCompCorScrub": dict(
        strategy=("motion", "high_pass", "compcor", "scrub"),
        motion="basic",
        compcor="anat_combined",
        scrub=0,
        fd_threshold=0.5,
        std_dvars_threshold=3.0,
    ),
}


class Data:
    def __init__(self, preproc, denoise_strategy):
        """mask, events, and confounds attributes of the fMRIPrep preprocessed data

        Parameters
        ----------
        preproc : str
            path to fMRIPrep preprocessed data (*desc-preproc_bold.nii.gz)
        denoise_strategy : str
            name of the confound regression strategy to use
        """
        self.preproc = preproc
        self.tr = image.load_img(preproc).header.get_zooms()[-1]
        self.keys = bids.parse_bids_filename(preproc)
        self.mask = f"{preproc[:-20]}-brain_mask.nii.gz"
        self.events = self._get_events_df()
        self.confounds, self.sample_mask = fmriprep.load_confounds(
            preproc, **denoise_strategies[denoise_strategy]
        )
        self.nvolumes_scrubbed = self._get_nvolumes_scrubbed()

    def _get_events_df(self):
        events = bids.get_bids_files(
            Path(self.preproc).parents[4] / "rawdata",
            file_tag="events",
            file_type="tsv",
            modality_folder="func",
            sub_label=self.keys["sub"],
            filters=[("task", self.keys["task"])],
        )[0]
        return pd.read_csv(events, sep="\t")

    def _get_nvolumes_scrubbed(self):
        """return the number of volumes scrubbed"""
        if self.sample_mask is None:
            return 0
        else:
            return self.confounds.index.difference(self.sample_mask).shape[0]
