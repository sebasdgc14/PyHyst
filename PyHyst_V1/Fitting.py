from Utilities import *
import numpy as np
import pandas as pd
import logging
from logging import getLogger
from scipy.optimize import curve_fit, nnls
from scipy.interpolate import CubicSpline
from sklearn.metrics import r2_score

logger = getLogger(__name__)

class Fitting:
  """
  This class performs curve fitting on magnetic hysteresis data using multiple modeling approaches.
  It supports single-function fitting, multi-basis adaptive fitting, and user-defined fitting strategies for Mih and Mrh components of hysteresis curves.
  The class integrates logistic, tanh, and sech² functions and facilitates interactive parameter estimation, visualization, and performance comparison between fitting models.
  """
  def __init__(self, Sample, mode, func_up=None, guess_up=None, func_lo=None, guess_lo=None, guess_diego=None):
    """
    Initializes the Fitting class with a sample object containing magnetic hysteresis data and begins the fitting process by invoking the interactive method selection prompt.
    """
    self.Sam = Sample
    self.mode = mode

    # If custom functions are provided, use them; otherwise, use defaults
    self.func_up = func_up if func_up else None
    self.guess_up = guess_up if guess_up else None
    self.func_lo = func_lo if func_lo else None
    self.guess_lo = guess_lo if guess_lo else None
    self.guess_diego = guess_diego if guess_diego else None
    # Start the fitting process
    if   self.mode == 'Single': self.Single_Fit()
    elif self.mode == 'Multi': self.Multi_Fit()
    elif self.mode == 'Diego': self.Diego()
    elif self.mode == 'Custom': self.Custom_Fit()
    else: pass

  ##############################################################################################################################################################################
  # Lets define the static methods for single fitting method
  @staticmethod
  def Logistic(x, x0, b):
    """
    Returns a logistic function value centered at x0 with steepness parameter b. Serves as the core sigmoid function for fitting.
    """
    return 1/(1 + np.exp(-b * (x - x0)))

  @staticmethod
  def Logistic_Displaced(x, x0, b, D):
    """
    Returns a displaced logistic function by adding offset D. Used for asymmetric fitting constructions.
    """
    return D + Fitting.Logistic(x, x0, b)

  @staticmethod
  def double_logistic_even(x, x0, A, b):
    """
    Even-symmetric double logistic curve for modeling Mrh. Uses mirrored and shifted logistic curves to maintain symmetry.
    """
    r = np.zeros(len(x))
    for i in range(len(x)):
      if x[i] <= 0:
        r[i] = A* (Fitting.Logistic_Displaced(x[i], x0, b, 0))
      else:
        r[i] = A*(-Fitting.Logistic_Displaced(x[i], -x0, b, -1))
    return r

  @staticmethod
  def double_logistic_odd(x, x0, A, b):
    """
    Odd-symmetric double logistic curve for modeling Mih. Uses subtraction of mirrored logistic curves for antisymmetric behavior.
    """
    return A*(Fitting.Logistic(x, x0, b) - Fitting.Logistic(-x, x0, b))

  @staticmethod
  def tanh_odd(x, x0, A, b):
    """
    Odd-symmetric hyperbolic tangent function for modeling Mih. Offers a smooth transition with tunable steepness.
    """
    return A * np.tanh(b * (x - x0))

  @staticmethod
  def guess_initial_params_log(x, y, window_frac=0.05):
    """
    Improved estimator for [x0, A, b] parameters for logistic-type models.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Amplitude: half the peak-to-peak range
    A = (np.max(y) - np.min(y)) / 2

    # Derivative estimate
    dy = np.gradient(y, x)
    max_slope_idx = np.argmax(np.abs(dy))
    x0 = x[max_slope_idx]

    # Fit a linear model locally to get more robust slope
    n = len(x)
    win = max(3, int(n * window_frac))
    start = max(0, max_slope_idx - win // 2)
    end = min(n, max_slope_idx + win // 2 + 1)

    x_local = x[start:end]
    y_local = y[start:end]

    # Linear fit
    if len(x_local) >= 2:
        coeffs = np.polyfit(x_local, y_local, 1)
        slope = coeffs[0]
    else:
        slope = dy[max_slope_idx]

    # Use logistic model property: slope_max ≈ b / 4 => b ≈ 4 * slope / A
    b = 4 * slope / (A + 1e-8)

    # Robust clipping: allow broader range but prevent divergence
    b = np.clip(np.abs(b), 1e-4, 1.0)

    return [x0, A, b]

  @staticmethod
  def guess_initial_params_tanh(x, y):
    """
    Estimates [x0, A, b] for a hyperbolic tangent model using the location of steepest slope and signal amplitude.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # Estimate x0 from maximum slope location
    dy = np.gradient(y, x)
    x0 = x[np.argmax(np.abs(dy))]
    # Amplitude estimation
    A = (np.max(y) - np.min(y)) / 2
    # Estimate b from the slope at x0
    slope = dy[np.argmax(np.abs(dy))]
    b = np.abs(slope / (A * (1 - np.tanh(0)**2) + 1e-8))  # Use derivative of tanh
    # Optional clip to prevent extreme steepness
    b = np.clip(b, 1e-5, 1e-1)
    return [x0, A, b]

  ##############################################################################################################################################################################
  ## Now lets define the relevant functions for multi function fitting and basis generators
  @staticmethod
  def sech2(x):
    """
    Computes the square of the hyperbolic secant, often used as a localized peak-like function. Useful for modeling even-symmetric features.
    """
    return 1 / np.cosh(x)**2

  @staticmethod
  def BG_Tanh_Mih(H, centers, widths):
    """
    Generates a set of odd-symmetric tanh basis functions for fitting Mih, using combinations of center and width parameters.
    """
    H_norm = H / H.max()
    basis = []
    for c, w in zip(centers, widths):
      basis.append(np.tanh((H_norm - c) / w)- np.tanh((-H_norm - c) / w))
    return np.column_stack(basis)

  @staticmethod
  def BG_DLO_Mih(H, centers, widths):
    """
    Generates a basis of double logistic odd functions for fitting Mih, built from varying center and width parameters.
    """
    H_norm = H
    basis = []
    for c, w in zip(centers, widths):
      basis.append(Fitting.double_logistic_odd(H_norm, c, 1, w))
    return np.column_stack(basis)

  @staticmethod
  def BG_Sech2_Mrh(H, centers, widths):
    """
    Constructs a basis of even-symmetric sech² functions for Mrh, combining mirrored contributions for symmetric behavior.
    """
    H_norm = H / H.max()
    basis = []
    for c, w in zip(centers, widths):
      term = Fitting.sech2((H_norm - c) / w) + Fitting.sech2((H_norm + c) / w)
      basis.append(term)
    return np.column_stack(basis)

  @staticmethod
  def BG_DLE_Mrh(H, centers, widths):
    """
    Generates a basis of double logistic even functions for Mrh, using mirrored logistic curves with varying parameters.
    """
    H_norm = H
    basis = []
    for c, w in zip(centers, widths):
      basis.append(Fitting.double_logistic_even(H_norm, c, 1, w))
    return np.column_stack(basis)

  @staticmethod
  def compute_adjusted_r2(y_true, y_pred, p):
    """
    Calculates Adjusted R^2 for fitting to account for number of parameters.
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return r2, adj_r2
  ##############################################################################################################################################################################
  # Diego function fitting
  @staticmethod
  def UpperDiego(H, Ms, Hc, Mr, Chi):
    return (2 * Ms / np.pi) * np.arctan(((H + Hc) / Hc) * np.tan(np.pi * Mr / (2 * Ms))) + Chi * H
  @staticmethod
  def LowerDiego(H, Ms, Hc, Mr, Chi):
      return (2 * Ms / np.pi) * np.arctan(((H - Hc) / Hc) * np.tan(np.pi * Mr / (2 * Ms))) + Chi * H
  ##############################################################################################################################################################################
  # Parameter estimation through cubic splines
  @staticmethod
  def Estimate_Mr_Hc(branch, field_col, fit_col='Fit', n_points=3):
    """
    Estimate remanent magnetization (Mr) and coercive field (Hc) from a hysteresis branch
    using cubic spline interpolation.

    Parameters
    ----------
    branch_df : pd.DataFrame
        The DataFrame containing the hysteresis branch data.
    field_col : str
        Column name for the magnetic field.
    fit_col : str
        Column name for the fitted magnetization data.
    n_points : int
        Number of points to use on either side of zero (for Mr)
        or the magnetization zero-crossing (for Hc).

    Returns
    -------
    Mr : float
        Remanent magnetization, interpolated at H = 0.
    Hc : float
        Coercive field, interpolated at M = 0.
    """
    H = branch[field_col].values
    M = branch[fit_col].values

    # === Estimate Mr: Interpolate around H = 0 ===
    idx_sorted_H = np.argsort(np.abs(H))
    idx_window_H = np.sort(idx_sorted_H[:2 * n_points])
    H_window = H[idx_window_H]
    M_window = M[idx_window_H]

    sort_H = np.argsort(H_window)
    spline_Mr = CubicSpline(H_window[sort_H], M_window[sort_H])
    Mr = spline_Mr(0.0)

    # === Estimate Hc: Interpolate around M = 0 ===
    sign_changes = np.where(np.diff(np.sign(M)))[0]
    if len(sign_changes) == 0:
        Hc = np.nan  # No zero crossing found
    else:
        i = sign_changes[0]  # First zero-crossing (can improve this if needed)
        start = max(i - n_points + 1, 0)
        end = min(i + n_points + 1, len(H))
        Hc_window = H[start:end]
        Mc_window = M[start:end]

        sort_Hc = np.argsort(Hc_window)
        spline_Hc = CubicSpline(Hc_window[sort_Hc], Mc_window[sort_Hc])
        roots = spline_Hc.roots()
        roots = roots[(roots >= Hc_window.min()) & (roots <= Hc_window.max())]
        Hc = roots[0] if len(roots) > 0 else np.nan
    return Mr, Hc


  ##############################################################################################################################################################################
  def Single_Fit(self):
    """
    Fits Mih and Mrh using a single mathematical function each (logistic or tanh). Automatically estimates parameters.
    """
    logger.info(f"Single fit was chosen")

    Ms_estimate = getattr(self.Sam, 'Ms', None)
    if Ms_estimate is None:
        Ms_estimate = self.Sam.Data[self.Sam.MagHead].max()

    # Data
    x   = self.Sam.Aux[self.Sam.FieldHead]
    mih = self.Sam.Aux['Mih']
    mrh = self.Sam.Aux['Mrh']
    # Mrh
    p0evenguess = Fitting.guess_initial_params_log(x, mrh)
    pfiteven, pcoveven = curve_fit(Fitting.double_logistic_even, x, mrh, p0=p0evenguess, maxfev=5000)
    r2_mrh = r2_score(mrh, Fitting.double_logistic_even(x, *pfiteven))
    # Mih
    p0oddguesslog  = Fitting.guess_initial_params_log(x, mih)
    p0oddguesstanh = Fitting.guess_initial_params_tanh(x, mih)
    pfitoddlog, pcovoddlog = curve_fit(Fitting.double_logistic_odd, x, mih, p0=[p0oddguesslog[0], Ms_estimate, p0oddguesslog[2]], maxfev=5000)
    pfitoddtanh, pcovoddtanh = curve_fit(Fitting.tanh_odd, x, mih, p0=p0oddguesstanh)
    r2_logistic = r2_score(mih, Fitting.double_logistic_odd(x, *pfitoddlog))
    r2_tanh = r2_score(mih, Fitting.tanh_odd(x, *pfitoddtanh))
    # Setting metrics
    if r2_logistic > r2_tanh:
      logger.info(f"Double Logistic was better fit. R^2: {r2_logistic:.4f} > {r2_tanh:.4f}")
      best_odd_func = Fitting.double_logistic_odd
      pfitodd = pfitoddlog
      setattr(self.Sam, 'Metrics', {'Mih': r2_logistic, 'Mrh': r2_mrh})
    else:
      logger.info(f"Tanh was better fit. R^2: {r2_tanh:.4f} > {r2_logistic:.4f}")
      best_odd_func = Fitting.tanh_odd
      pfitodd = pfitoddtanh
      setattr(self.Sam, 'Metrics', {'Mih': r2_tanh, 'Mrh': r2_mrh})
    Mih_fit = best_odd_func(x, *pfitodd)
    Mrh_fit = Fitting.double_logistic_even(x, *pfiteven)
    Up_fit = Mih_fit + Mrh_fit
    Lo_fit = Mih_fit - Mrh_fit
    self.Sam.Aux['Mrh_fit'] = Mrh_fit
    self.Sam.Aux['Mih_fit'] = Mih_fit
    self.Sam.Up['Fit'] = Up_fit
    self.Sam.Lo['Fit'] = Lo_fit[::-1].reset_index(drop=True)
    # Update Data
    self.Sam.Data = Update_Data(self.Sam.Up, self.Sam.Lo)
    ### Parameters
    self.Parameter_Extraction()

  ##############################################################################################################################################################################
  def Multi_Fit(self):
    """
    Uses a linear combination of multiple basis functions (tanh, logistic, sech²) to fit Mih and Mrh with improved flexibility and accuracy. Solves the system using non-negative least squares and plots the resulting fit.
    For samples in which branches cross, this method is ill-defined. This is because the base for Mrh is all positive functions and crossing branches neccessarily involves negative M values in crossing regions.
    """

    logger.info(f"Multi fit was chosen")

    ### Variable setting
    H        = self.Sam.Aux[self.Sam.FieldHead]
    Mih_true = self.Sam.Aux['Mih']
    Mrh_true = self.Sam.Aux['Mrh']
    ### Fitting process
    ## Mih
    # Tanh
    C_Tanh_Mih    = np.linspace(0.01, 0.9, 10)
    W_Tanh_Mih    = np.full_like(C_Tanh_Mih, 0.2)
    B_Tanh_Mih    = Fitting.BG_Tanh_Mih(H, C_Tanh_Mih, W_Tanh_Mih)
    # DLO
    DLO_Guess0     = Fitting.guess_initial_params_log(H, Mih_true)
    DLO_Guess, _   = curve_fit(Fitting.double_logistic_odd, H, Mih_true, p0=DLO_Guess0, maxfev=5000)
    W_DLO_Mih     = np.linspace(0.8*DLO_Guess[2], 4*DLO_Guess[2], 10)
    C_DLO_Mih     = np.linspace(0.5*DLO_Guess[0], 1*DLO_Guess[0], 10)
    B_DLO_Mih     = Fitting.BG_DLO_Mih(H, C_DLO_Mih, W_DLO_Mih )
    # Complete
    Basis_Mih     = np.hstack((B_Tanh_Mih, B_DLO_Mih))
    Coeffs_Mih, _ = nnls(Basis_Mih, Mih_true)
    Mih_fit       = Basis_Mih @Coeffs_Mih

    ## Mrh
    # Sech2
    C_Sech2_Mrh   = np.linspace(0.0, 0.9, 10)  # includes zero
    W_Sech2_Mrh   = np.full_like(C_Sech2_Mrh, 0.2)
    B_Sech2_Mrh   = Fitting.BG_Sech2_Mrh(H, C_Sech2_Mrh, W_Sech2_Mrh)
    # DLE
    DLE_Guess0    = Fitting.guess_initial_params_log(H, Mrh_true)
    DLE_Guess, _  = curve_fit(Fitting.double_logistic_even, H, Mrh_true, p0=DLE_Guess0, maxfev=5000)
    W_DLE_Mrh     = np.linspace(0.8*DLE_Guess[2], 4*DLE_Guess[2], 10)
    C_DLE_Mrh     = np.linspace(0.5*DLE_Guess[0], 1*DLE_Guess[0], 10)
    B_DLE_Mrh1     = Fitting.BG_DLE_Mrh(H, C_DLE_Mrh, W_DLE_Mrh)
    B_DLE_Mrh2    = -Fitting.BG_DLE_Mrh(H, C_DLE_Mrh, W_DLE_Mrh)
    B_DLE_Mrh     = np.hstack((B_DLE_Mrh1, B_DLE_Mrh2))

    # Complete
    Basis_Mrh     = np.hstack((B_Sech2_Mrh, B_DLE_Mrh))
    Coeffs_Mrh, _ = nnls(Basis_Mrh, Mrh_true)
    Mrh_fit       = Basis_Mrh @ Coeffs_Mrh
    # Final calcs
    Up_fit = Mih_fit + Mrh_fit
    Lo_fit = Mih_fit - Mrh_fit
    # Performing F-test for sample
    LowerInv     = Branch_Inversion(self.Sam.Lo)
    LowerInv_fit = -Lo_fit
    # Adding fits to sample attributes
    self.Sam.Aux['Mrh_fit'] = Mrh_fit
    self.Sam.Aux['Mih_fit'] = Mih_fit
    self.Sam.Up['Fit'] = Up_fit
    self.Sam.Lo['Fit'] = Lo_fit[::-1]
    # Update Data
    self.Sam.Data = Update_Data(self.Sam.Up, self.Sam.Lo)
    ### Adjusted R^2 for fit goodness
    r2_mih, adj_r2_mih = Fitting.compute_adjusted_r2(Mih_true, Mih_fit, p=len(Coeffs_Mih))
    r2_mrh, adj_r2_mrh = Fitting.compute_adjusted_r2(Mrh_true, Mrh_fit, p=len(Coeffs_Mrh))
    setattr(self.Sam, 'Metrics', {'Mih': adj_r2_mih, 'Mrh': adj_r2_mrh})
    ### Parameters
    self.Parameter_Extraction()

  ##############################################################################################################################################################################
  def Diego(self):
    """
    PlaceHolder name and description
    """
    Ms = self.Sam.Data[self.Sam.MagHead].max()
    pfitup, _ = curve_fit(Fitting.UpperDiego, self.Sam.Up[self.Sam.FieldHead], self.Sam.Up[self.Sam.MagHead], p0=[Ms, self.guess_diego[0], self.guess_diego[1], self.guess_diego[2]])
    pfitlo, _ = curve_fit(Fitting.LowerDiego, self.Sam.Lo[self.Sam.FieldHead], self.Sam.Lo[self.Sam.MagHead], p0=[Ms, self.guess_diego[0], self.guess_diego[1], self.guess_diego[2]])
    Ms_Up, Ms_Lo = pfitup[0], pfitlo[0]
    Mr_Up, Mr_Lo = pfitup[2], pfitlo[2]
    Hc_Up, Hc_Lo = pfitup[1], pfitlo[1]
    Chi = pfitup[3]
    # Adding fit to sample attribute
    self.Sam.Up['Fit'] = Fitting.UpperDiego(self.Sam.Up[self.Sam.FieldHead], *pfitup)
    self.Sam.Lo['Fit'] = Fitting.LowerDiego(self.Sam.Lo[self.Sam.FieldHead], *pfitlo)
    # Update Data
    self.Sam.Data = Update_Data(self.Sam.Up, self.Sam.Lo)
    ### Parameters
    setattr(self.Sam, 'Params', {'Ms_Up': Ms_Up, 'Ms_Lo': Ms_Lo, 'Mr_Up': Mr_Up, 'Mr_Lo': Mr_Lo, 'Hc_Up': Hc_Up, 'Hc_Lo': Hc_Lo, 'Chi': Chi})

  ##############################################################################################################################################################################

  def Custom_Fit(self):
    """
    Fits upper and lower branches with user-provided functions and guesses.
    All functions and guesses must be provided as class attributes (set from GUI).
    Stores optimal parameters for GUI access.
    """
    logger.info(f"Custom fit was chosen")

    # --- Upper branch ---
    if self.func_up is not None:
        try:
            func_up = eval(self.func_up, {"np": np})
        except Exception as e:
            logger.error(f"Error parsing upper function: {e}")
            self.opt_params_up = None
            self.opt_params_lo = None
            return
    else:
        logger.error("No upper branch function provided.")
        self.opt_params_up = None
        self.opt_params_lo = None
        return

    if self.guess_up is not None:
        guess_up = self.guess_up
    else:
        logger.error("No upper branch parameter guesses provided.")
        self.opt_params_up = None
        self.opt_params_lo = None
        return

    # --- Lower branch ---
    if self.func_lo is not None:
        try:
            func_lo = eval(self.func_lo, {"np": np})
        except Exception as e:
            logger.error(f"Error parsing lower function: {e}")
            self.opt_params_up = None
            self.opt_params_lo = None
            return
    else:
        logger.error("No lower branch function provided.")
        self.opt_params_up = None
        self.opt_params_lo = None
        return

    if self.guess_lo is not None:
        guess_lo = self.guess_lo
    else:
        logger.error("No lower branch parameter guesses provided.")
        self.opt_params_up = None
        self.opt_params_lo = None
        return

    # Extract x and y values
    x_up = self.Sam.Up[self.Sam.FieldHead]
    y_up = self.Sam.Up[self.Sam.MagHead]
    x_lo = self.Sam.Lo[self.Sam.FieldHead]
    y_lo = self.Sam.Lo[self.Sam.MagHead]

    def wrap_up(x, *params):
        return func_up(x, *params)

    def wrap_lo(x, *params):
        return func_lo(x, *params)

    try:
        popt_up, _ = curve_fit(wrap_up, x_up, y_up, p0=guess_up, maxfev=5000)
        popt_lo, _ = curve_fit(wrap_lo, x_lo, y_lo, p0=guess_lo, maxfev=5000)
    except Exception as e:
        logger.error(f"Fitting failed: {e}")
        self.opt_params_up = None
        self.opt_params_lo = None
        return

    self.opt_params_up = popt_up
    self.opt_params_lo = popt_lo

    self.Sam.Up['Fit'] = wrap_up(x_up, *popt_up)
    self.Sam.Lo['Fit'] = wrap_lo(x_lo, *popt_lo)
    # Update Data
    self.Sam.Data = Update_Data(self.Sam.Up, self.Sam.Lo)

    r2_up = r2_score(y_up, self.Sam.Up['Fit'])
    r2_lo = r2_score(y_lo, self.Sam.Lo['Fit'])
    setattr(self.Sam, 'Metrics', {'Upper': r2_up, 'Lower': r2_lo})
    logger.info(self.Sam.Metrics)
    # Parameters
    self.Parameter_Extraction()

  ##############################################################################################################################################################################
  def Parameter_Extraction(self):
    """
    Estimates parameters after fitting
    """
    # Ms
    if not hasattr(self.Sam, 'Ms'):
      Ms_Up = self.Sam.Up['Fit'][self.Sam.Up[self.Sam.FieldHead]>=0.8*self.Sam.Up[self.Sam.FieldHead].max()].mean()
      Ms_Lo = self.Sam.Lo['Fit'][self.Sam.Lo[self.Sam.FieldHead]<=0.8*self.Sam.Lo[self.Sam.FieldHead].min()].mean()
    else:
      Ms_Up, Ms_Lo = self.Sam.Ms, -self.Sam.Ms
    # Upper
    Mr_Up, Hc_Up = Fitting.Estimate_Mr_Hc(self.Sam.Up, self.Sam.FieldHead)
    # Lower
    Mr_Lo, Hc_Lo = Fitting.Estimate_Mr_Hc(self.Sam.Lo, self.Sam.FieldHead)
    # Setting attribute in sample
    setattr(self.Sam, 'Params', {'Ms_Up': Ms_Up, 'Ms_Lo': Ms_Lo, 'Mr_Up': Mr_Up, 'Mr_Lo': Mr_Lo, 'Hc_Up': Hc_Up, 'Hc_Lo': Hc_Lo})
    logger.info(f"Extracted Parameters: {self.Sam.Params}")