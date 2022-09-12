"""
code copied from: https://github.com/asoroosh/xDF

Afyouni, Smith, and Nichols, “Effective Degrees of Freedom of the Pearson's Correlation Coefficient under Autocorrelation.”
"""

import numpy as np
import scipy.stats as sp
import statsmodels.stats.multitest as smmt


def nextpow2(x):
    """
    nextpow2 Next higher power of 2.
    nextpow2(N) returns the first P such that P >= abs(N).  It is
    often useful for finding the nearest power of two sequence
    length for FFT operations.
    """
    return 1 if x == 0 else int(2 ** np.ceil(np.log2(x)))


def CorrMat(ts, T, method="rho", copy=True):
    """
    Produce sample correlation matrices
    or Naively corrected z maps.
    """

    if copy:
        ts = ts.copy()

    if np.shape(ts)[1] != T:
        print("xDF::: Input should be in IxT form, the matrix was transposed.")
        ts = np.transpose(ts)

    N = np.shape(ts)[0]
    R = np.corrcoef(ts)

    Z = np.arctanh(R) * np.sqrt(T - 3)

    R[range(N), range(N)] = 0
    Z[range(N), range(N)] = 0

    return R, Z


def AC_fft(Y, T, copy=True):

    if copy:
        Y = Y.copy()

    if np.shape(Y)[1] != T:
        print("AC_fft::: Input should be in IxT form, the matrix was transposed.")
        Y = np.transpose(Y)

    print("AC_fft::: Demean along T")
    mY2 = np.mean(Y, axis=1)
    Y = Y - np.transpose(np.tile(mY2, (T, 1)))

    nfft = int(nextpow2(2 * T - 1))  # zero-pad the hell out!
    yfft = np.fft.fft(Y, n=nfft, axis=1)  # be careful with the dimensions
    ACOV = np.real(np.fft.ifft(yfft * np.conj(yfft), axis=1))
    ACOV = ACOV[:, 0:T]

    Norm = np.sum(np.abs(Y) ** 2, axis=1)
    Norm = np.transpose(np.tile(Norm, (T, 1)))
    xAC = ACOV / Norm  # normalise the COVs

    bnd = (np.sqrt(2) * 1.3859) / np.sqrt(T)  # assumes normality for AC
    CI = [-bnd, bnd]

    return xAC, CI


def xC_fft(Y, T, mxL=[], copy=True):

    # ***********************************
    # This should be checked! There shouldn't be any complex numbers!!
    # __main__:74: ComplexWarning: Casting complex values to real discards the imaginary part
    # This is because Python, in contrast to Matlab, produce highly prcise imaginary parts
    # by defualt, when you wanna do ifft, just use np.real()
    # ***********************************

    if copy:
        Y = Y.copy()

    if np.shape(Y)[1] != T:
        print("xC_fft::: Input should be in IxT form, the matrix was transposed.")
        Y = np.transpose(Y)

    if not np.size(mxL):
        mxL = T

    I = np.shape(Y)[0]

    print("xC_fft::: Demean along T")
    mY2 = np.mean(Y, axis=1)
    Y = Y - np.transpose(np.tile(mY2, (T, 1)))

    nfft = nextpow2(2 * T - 1)  # zero-pad the hell out!
    yfft = np.fft.fft(Y, n=nfft, axis=1)  # be careful with the dimensions

    mxLcc = (mxL - 1) * 2 + 1
    xC = np.zeros([I, I, mxLcc])

    XX = np.triu_indices(I, 1)[0]
    YY = np.triu_indices(I, 1)[1]

    for i in np.arange(np.size(XX)):  # loop around edges.

        xC0 = np.fft.ifft(yfft[XX[i], :] * np.conj(yfft[YY[i], :]), axis=0)
        xC0 = np.real(xC0)
        xC0 = np.concatenate((xC0[-mxL + 1 : None], xC0[0:mxL]))

        xC0 = np.fliplr([xC0])[0]
        Norm = np.sqrt(
            np.sum(np.abs(Y[XX[i], :]) ** 2) * np.sum(np.abs(Y[YY[i], :]) ** 2)
        )

        xC0 = xC0 / Norm
        xC[XX[i], YY[i], :] = xC0
        del xC0

    xC = xC + np.transpose(xC, (1, 0, 2))
    lidx = np.arange(-(mxL - 1), mxL)

    return xC, lidx


def shrinkme(ac, T):
    """
    Shrinks the *early* bucnhes of autocorr coefficients beyond the CI.
    Yo! this should be transformed to the matrix form, those fors at the top
    are bleak!

    SA, Ox, 2018
    """
    ac = ac.copy()

    if np.shape(ac)[1] != T:
        ac = ac.T

    bnd = (np.sqrt(2) * 1.3859) / np.sqrt(T)  # assumes normality for AC

    N = np.shape(ac)[0]
    msk = np.zeros(np.shape(ac))
    BreakPoint = np.zeros(N)
    for i in np.arange(N):
        TheFirstFalse = np.where(
            np.abs(ac[i, :]) < bnd
        )  # finds the break point -- intercept
        if (
            np.size(TheFirstFalse) == 0
        ):  # if you coulnd't find a break point, then continue = the row will remain zero
            continue
        else:
            BreakPoint_tmp = TheFirstFalse[0][0]
        msk[i, :BreakPoint_tmp] = 1
        BreakPoint[i] = BreakPoint_tmp
    return ac * msk, BreakPoint


def curbtaperme(ac, T, M, verbose=True):
    """
    Curb the autocorrelations, according to Anderson 1984
    multi-dimensional, and therefore is fine!
    SA, Ox, 2018
    """
    ac = ac.copy()
    M = int(round(M))
    msk = np.zeros(np.shape(ac))
    if len(np.shape(ac)) == 2:
        if verbose:
            print("curbtaperme::: The input is 2D.")
        msk[:, 0:M] = 1

    elif len(np.shape(ac)) == 3:
        if verbose:
            print("curbtaperme::: The input is 3D.")
        msk[:, :, 0:M] = 1

    elif len(np.shape(ac)) == 1:
        if verbose:
            print("curbtaperme::: The input is 1D.")
        msk[0:M] = 1

    ct_ts = msk * ac

    return ct_ts


def SumMat(Y0, T, copy=True):
    """
    Parameters
    ----------
    Y0 : a 2D matrix of size TxN

    Returns
    -------
    SM : 3D matrix, obtained from element-wise summation of each row with other
         rows.

    SA, Ox, 2019
    """

    if copy:
        Y0 = Y0.copy()

    if np.shape(Y0)[0] != T:
        print("SumMat::: Input should be in TxN form, the matrix was transposed.")
        Y0 = np.transpose(Y0)

    N = np.shape(Y0)[1]
    Idx = np.triu_indices(N)  # F = (N*(N-1))/2
    SM = np.empty([N, N, T])
    for i in np.arange(0, np.size(Idx[0]) - 1):
        xx = Idx[0][i]
        yy = Idx[1][i]
        SM[xx, yy, :] = Y0[:, xx] + Y0[:, yy]
        SM[yy, xx, :] = Y0[:, yy] + Y0[:, xx]

    return SM


def ProdMat(Y0, T, copy=True):
    """
    Parameters
    ----------
    Y0 : a 2D matrix of size TxN

    Returns
    -------
    SM : 3D matrix, obtained from element-wise multiplication of each row with
         other rows.

    SA, Ox, 2019
    """

    if copy:
        Y0 = Y0.copy()

    if np.shape(Y0)[0] != T:
        print("ProdMat::: Input should be in TxN form, the matrix was transposed.")
        Y0 = np.transpose(Y0)

    N = np.shape(Y0)[1]
    Idx = np.triu_indices(N)  # F = (N*(N-1))/2
    SM = np.empty([N, N, T])
    for i in np.arange(0, np.size(Idx[0]) - 1):
        xx = Idx[0][i]
        yy = Idx[1][i]
        SM[xx, yy, :] = Y0[:, xx] * Y0[:, yy]
        SM[yy, xx, :] = Y0[:, yy] * Y0[:, xx]

    return SM


def tukeytaperme(ac, T, M, verbose=True):
    """
    performs single Tukey tapering for given length of window, M, and initial
    value, intv. intv should only be used on crosscorrelation matrices.

    SA, Ox, 2018
    """
    ac = ac.copy()
    # ----Checks:
    if not T in np.shape(ac):
        raise ValueError("tukeytaperme::: There is something wrong, mate!")
        # print('Oi')
    # ----

    M = int(np.round(M))

    tukeymultiplier = (1 + np.cos(np.arange(1, M) * np.pi / M)) / 2
    tt_ts = np.zeros(np.shape(ac))

    if len(np.shape(ac)) == 2:
        if np.shape(ac)[1] != T:
            ac = ac.T
        if verbose:
            print("tukeytaperme::: The input is 2D.")
        N = np.shape(ac)[0]
        tt_ts[:, 0 : M - 1] = np.tile(tukeymultiplier, [N, 1]) * ac[:, 0 : M - 1]

    elif len(np.shape(ac)) == 3:
        if verbose:
            print("tukeytaperme::: The input is 3D.")
        N = np.shape(ac)[0]
        tt_ts[:, :, 0 : M - 1] = (
            np.tile(tukeymultiplier, [N, N, 1]) * ac[:, :, 0 : M - 1]
        )

    elif len(np.shape(ac)) == 1:
        if verbose:
            print("tukeytaperme::: The input is 1D.")
        tt_ts[0 : M - 1] = tukeymultiplier * ac[0 : M - 1]

    return tt_ts


def binarize(W, copy=True):
    """
    Binarizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*
    Parameters
    ----------
    W : NxN np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.
    Returns
    -------
    W : NxN np.ndarray
        binary connectivity matrix
    """
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W


def xDF_Calc(
    ts, T, method="truncate", methodparam="adaptive", verbose=True, TV=True, copy=True
):

    # -------------------------------------------------------------------------
    ##### READ AND CHECK 0---------------------------------------------------------

    # if not verbose: blockPrint()

    if copy:  # Make sure you are not messing around with the original time series
        ts = ts.copy()

    if np.shape(ts)[1] != T:
        if verbose:
            print("xDF::: Input should be in IxT form, the matrix was transposed.")
        ts = np.transpose(ts)

    N = np.shape(ts)[0]

    ts_std = np.std(ts, axis=1, ddof=1)
    ts = ts / np.transpose(np.tile(ts_std, (T, 1)))
    # standardise
    print("xDF_Calc::: Time series standardised by their standard deviations.")
    # READ AND CHECK 0---------------------------------------------------------
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ##### Estimate xC and AC ------------------------------------------------------

    # Corr----------------------------------------------------------------------
    rho, znaive = CorrMat(ts, T)
    rho = np.round(rho, 7)
    znaive = np.round(znaive, 7)
    # Autocorr------------------------------------------------------------------
    [ac, CI] = AC_fft(ts, T)
    ac = ac[:, 1 : T - 1]
    # The last element of ACF is rubbish, the first one is 1, so why bother?!
    nLg = T - 2

    # Cross-corr----------------------------------------------------------------
    [xcf, lid] = xC_fft(ts, T)

    xc_p = xcf[:, :, 1 : T - 1]
    xc_p = np.flip(xc_p, axis=2)
    # positive-lag xcorrs
    xc_n = xcf[:, :, T:-1]
    # negative-lag xcorrs

    # -------------------------------------------------------------------------
    ##### Start of Regularisation--------------------------------------------------
    if method.lower() == "tukey":
        if methodparam == "":
            M = np.sqrt(T)
        else:
            M = methodparam
        if verbose:
            print(
                "xDF_Calc::: AC Regularisation: Tukey tapering of M = "
                + str(int(np.round(M)))
            )
        ac = tukeytaperme(ac, nLg, M)
        xc_p = tukeytaperme(xc_p, nLg, M)
        xc_n = tukeytaperme(xc_n, nLg, M)

        # print(np.round(ac[0,0:50],4))

    elif method.lower() == "truncate":
        if type(methodparam) == str:  # Adaptive Truncation
            if methodparam.lower() != "adaptive":
                raise ValueError(
                    "What?! Choose adaptive as the option, or pass an integer for truncation"
                )
            if verbose:
                print("xDF_Calc::: AC Regularisation: Adaptive Truncation")
            [ac, bp] = shrinkme(ac, nLg)
            # truncate the cross-correlations, by the breaking point found from the ACF. (choose the largest of two)
            for i in np.arange(N):
                for j in np.arange(N):
                    maxBP = np.max([bp[i], bp[j]])
                    xc_p[i, j, :] = curbtaperme(
                        xc_p[i, j, :], nLg, maxBP, verbose=False
                    )
                    xc_n[i, j, :] = curbtaperme(
                        xc_n[i, j, :], nLg, maxBP, verbose=False
                    )
        elif type(methodparam) == int:  # Npne-Adaptive Truncation
            if verbose:
                print(
                    "xDF_Calc::: AC Regularisation: Non-adaptive Truncation on M = "
                    + str(methodparam)
                )
            ac = curbtaperme(ac, nLg, methodparam)
            xc_p = curbtaperme(xc_p, nLg, methodparam)
            xc_n = curbtaperme(xc_n, nLg, methodparam)

        else:
            raise ValueError(
                "xDF_Calc::: methodparam for truncation method should be either str or int."
            )
    # Start of Regularisation--------------------------------------------------
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    ##### Start of the Monster Equation--------------------------------------------
    # -------------------------------------------------------------------------

    wgt = np.arange(nLg, 0, -1)
    wgtm2 = np.tile((np.tile(wgt, [N, 1])), [N, 1])
    wgtm3 = np.reshape(wgtm2, [N, N, np.size(wgt)])
    # this is shit, eats all the memory!
    Tp = T - 1

    """
     VarHatRho = (Tp*(1-rho.^2).^2 ...
     +   rho.^2 .* sum(wgtm3 .* (SumMat(ac.^2,nLg)  +  xc_p.^2 + xc_n.^2),3)...         %1 2 4
     -   2.*rho .* sum(wgtm3 .* (SumMat(ac,nLg)    .* (xc_p    + xc_n))  ,3)...         % 5 6 7 8
     +   2      .* sum(wgtm3 .* (ProdMat(ac,nLg)    + (xc_p   .* xc_n))  ,3))./(T^2);   % 3 9 
    """

    # Da Equation!--------------------
    VarHatRho = (
        Tp * (1 - rho**2) ** 2
        + rho**2
        * np.sum(wgtm3 * (SumMat(ac**2, nLg) + xc_p**2 + xc_n**2), axis=2)
        - 2 * rho * np.sum(wgtm3 * (SumMat(ac, nLg) * (xc_p + xc_n)), axis=2)
        + 2 * np.sum(wgtm3 * (ProdMat(ac, nLg) + (xc_p * xc_n)), axis=2)
    ) / (T**2)

    # -----------------------------------------------------------------------
    # End of the Monster Equation--------------------------------------------
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    ##### Truncate to Theoritical Variance --------------------------------------
    TV_val = (1 - rho**2) ** 2 / T
    TV_val[range(N), range(N)] = 0

    idx_ex = np.where(VarHatRho < TV_val)
    NumTVEx = (np.shape(idx_ex)[1]) / 2
    # print(NumTVEx)

    if NumTVEx > 0 and TV:
        if verbose:
            print("Variance truncation is ON.")
        # Assuming that the variance can *only* get larger in presence of autocorrelation.
        VarHatRho[idx_ex] = TV_val[idx_ex]
        # print(N)
        # print(np.shape(idx_ex)[1])
        FGE = N * (N - 1) / 2
        if verbose:
            print(
                "xDF_Calc::: "
                + str(NumTVEx)
                + " ("
                + str(round((NumTVEx / FGE) * 100, 3))
                + "%) edges had variance smaller than the textbook variance!"
            )
    else:
        if verbose:
            print("xDF_Calc::: NO truncation to the theoritical variance.")
    # Sanity Check:
    #        for ii in np.arange(NumTVEx):
    #            print( str( idx_ex[0][ii]+1 ) + '  ' + str( idx_ex[1][ii]+1 ) )

    # -------------------------------------------------------------------------
    #####Start of Statistical Inference -------------------------------------------

    # Well, these are all Matlab and pretty useless -- copy pasted them just in case though...
    # Pearson's turf -- We don't really wanna go there, eh?
    # rz      = rho./sqrt((ASAt));     %abs(ASAt), because it is possible to get negative ASAt!
    # r_pval  = 2 * normcdf(-abs(rz)); %both tails
    # r_pval(1:nn+1:end) = 0;          %NaN screws up everything, so get rid of the diag, but becareful here.

    # Our turf--------------------------------
    rf = np.arctanh(rho)
    sf = VarHatRho / (
        (1 - rho**2) ** 2
    )  # delta method; make sure the N is correct! So they cancel out.
    rzf = rf / np.sqrt(sf)
    f_pval = 2 * sp.norm.cdf(-abs(rzf))  # both tails

    # diagonal is rubbish;
    VarHatRho[range(N), range(N)] = 0
    f_pval[
        range(N), range(N)
    ] = 0  # NaN screws up everything, so get rid of the diag, but becareful here.
    rzf[range(N), range(N)] = 0

    # End of Statistical Inference ---------------------------------------------
    # -------------------------------------------------------------------------

    # if not verbose: enablePrint()

    xDFOut = {
        "p": f_pval,
        "z": rzf,
        "znaive": znaive,
        "v": VarHatRho,
        "TV": TV_val,
        "TVExIdx": idx_ex,
    }

    return xDFOut


def stat_threshold(Z, mce="fdr_bh", a_level=0.05, side="two", copy=True):
    """
    Threshold z maps

    Parameters
    ----------

    mce: multiple comparison error correction method, should be
    among of the options below. [defualt: 'fdr_bh'].
    The options are from statsmodels packages:

        `b`, `bonferroni` : one-step correction
        `s`, `sidak` : one-step correction
        `hs`, `holm-sidak` : step down method using Sidak adjustments
        `h`, `holm` : step-down method using Bonferroni adjustments
        `sh`, `simes-hochberg` : step-up method  (independent)
        `hommel` : closed method based on Simes tests (non-negative)
        `fdr_i`, `fdr_bh` : Benjamini/Hochberg  (non-negative)
        `fdr_n`, `fdr_by` : Benjamini/Yekutieli (negative)
        'fdr_tsbh' : two stage fdr correction (Benjamini/Hochberg)
        'fdr_tsbky' : two stage fdr correction (Benjamini/Krieger/Yekutieli)
        'fdr_gbs' : adaptive step-down fdr correction (Gavrilov, Benjamini, Sarkar)
    """

    if copy:
        Z = Z.copy()

    if side == "one":
        sideflag = 1
    elif side == "two" or "double":
        sideflag = 2

    Idx = np.triu_indices(Z.shape[0], 1)
    Zv = Z[Idx]

    Pv = sp.norm.cdf(-np.abs(Zv)) * sideflag

    [Hv, adjpvalsv] = smmt.multipletests(Pv, method=mce)[:2]
    adj_pvals = np.zeros(Z.shape)
    Zt = np.zeros(Z.shape)

    Zv[np.invert(Hv)] = 0
    Zt[Idx] = Zv
    Zt = Zt + Zt.T

    adj_pvals[Idx] = adjpvalsv
    adj_pvals = adj_pvals + adj_pvals.T

    adj_pvals[range(Z.shape[0]), range(Z.shape[0])] = 0

    return Zt, binarize(Zt), adj_pvals
