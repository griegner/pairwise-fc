import numpy as np
from sklearn.linear_model import LinearRegression


class PPI:
    def __init__(self, vectorize=False, fit_intercept=False, n_jobs=1):
        """pairwise psychophysiological interaction (PPI) model

        Parameters
        ----------
        vectorize : bool, optional
            vectorize the pairwise connectivity matrix, by default False
        fit_intercept : bool, optional
            fit an intercept in the linear model, by default False
        n_jobs : int, optional
            number of jobs to use, by default 1
        """
        self.vectorize = vectorize
        self.linear_model = LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs)

    def fit_transform(self, roi_timeseries, design_matrix):
        """fit the PPI model to the input data

        Parameters
        ----------
        roi_timeseries : array-like, shape (n_samples, n_rois)
            timeseries from each region of interest
        design_matrix : array-like, shape (n_samples, n_regressors)
            design matrix

        Returns
        -------
        array-like, shape (n_rois, n_rois)
            pairwise PPI matrix
        """
        ppi_coefs = []
        for roi in roi_timeseries.T:
            ppi_matrix = design_matrix.assign(roi=roi, ppi=design_matrix["stim"] * roi)
            lm_fit = self.linear_model.fit(X=ppi_matrix, y=roi_timeseries)
            ppi_coefs.append(lm_fit.coef_[:, -1])

        ppi_coefs = np.stack(ppi_coefs)

        if self.vectorize:
            ppi_coefs = ppi_coefs.reshape(-1)

        return ppi_coefs
