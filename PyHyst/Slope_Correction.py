from Utilities import *
import numpy as np
from scipy.optimize import lsq_linear
import logging
from logging import getLogger

logger = getLogger(__name__)

class Slope_Correction:
  """
  Applies high-field slope correction to magnetic hysteresis data.

  This class fits the positive high-field portion of the upper branch of the magnetization
  curve using the approach to saturation equation (Eq. 18 from Jackson & Solheid, 2010):
  
      M(H) = χH + Ms + αH^β

  The slope (χ) is then subtracted from both the upper and lower branches to correct
  for non-saturating trends in high fields. Saturation magnetization (Ms) is estimated
  from the fit. Optionally, a logger or plot_callback can be supplied for GUI compatibility.

  References:
  - Jackson & Solheid (2010): On the quantitative analysis and evaluation of magnetic hysteresis data.
  - Von Dobeneck (1996): A systematic analysis of natural magnetic mineral assemblages based on modelling hysteresis loops with coercivity-related hyperbolic basis functions.


  Attributes
  ----------
  Sam : Read_Data object
      The sample containing field and magnetization data to be corrected.
  App : bool
      Flag to apply the correction immediately upon initialization.
  logger : callable or None
      Optional logging callback to report results/messages (e.g., to a GUI).
  plot_callback : callable or None
      Optional plot callback for rendering results (instead of plt.show()).
  """
  def __init__(self, Sample, method):
    """
    Initialize slope correction with the given sample and apply if specified.

    Parameters
    ----------
    Sample : Read_Data
        Input sample containing magnetic hysteresis data.
    Apply : bool
        Whether to apply slope correction immediately.
    logger : callable, optional
        Function to send log/output messages (e.g., GUI text box).
    plot_callback : callable, optional
        Function to handle plotting (e.g., embedded matplotlib widget).
    """
    self.Sam  = Sample
    self.method = method

    if self.method   == 'None'  : logger.info("Slope correction skipped (mode: None)")
    elif self.method == 'Approach': self.Approach_Saturation()
    elif self.method == 'Linear'  : self.Linear_Saturation()

  ##############################################################################################################################################################################
  @staticmethod
  def fit_Eq18(H, M, beta=-2):
    """
    Fit the approach to saturation equation: M = χH + Ms + αH^β

    Parameters
    ----------
    H : array-like
        Applied field values (positive high-field).
    M : array-like
        Corresponding magnetization values.
    beta : float, optional
        Exponent for the non-linear term (default is -2).

    Returns
    -------
    params : ndarray
        Fitted parameters [χ, Ms, α].
    fitted : ndarray
        Fitted magnetization curve.
    """
    A = np.vstack([H, np.ones_like(H), H**beta]).T
    bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, 0]) # makes sure alph<0 to ensure approach to saturation
    result = lsq_linear(A, M, bounds=bounds)
    params = result.x
    logger.info(f'Ms: {params[1]:.4e}, Chi: {params[0]:.4e}, Alpha: {params[2]:.4e}')
    return params, A @ params

  ##############################################################################################################################################################################
  def Approach_Saturation(self, beta=-2):
    """
    Applies slope correction to both upper and lower branches of the hysteresis loop.

    High-field positive values from the upper branch are fitted to estimate the slope (χ),
    saturation magnetization (Ms), and α term. This slope is subtracted from both branches,
    and dependencies are updated accordingly.

    Parameters
    ----------
    beta : float, optional
        Exponent in the approach to saturation model (default is -2).
    """
    # Extract upper branch data
    H_up = self.Sam.Up[self.Sam.FieldHead]
    M_up = self.Sam.Up[self.Sam.MagHead]
    # Extract lower branch data
    H_lo = self.Sam.Lo[self.Sam.FieldHead]
    M_lo = self.Sam.Lo[self.Sam.MagHead]

    # Step 1: Select high-field positive part (≥70% of max field)
    mask_hf = H_up >= 0.7 * H_up.max()
    H_hf = H_up[mask_hf]
    M_hf = M_up[mask_hf]

    params, _ = Slope_Correction.fit_Eq18(H_hf, M_hf)

    # Step 1: Correct Upper branch
    M_up_corrected = M_up - params[0]*H_up
    # Update upper branch dependencies
    self.Sam.Up[self.Sam.FieldHead] = H_up
    self.Sam.Up[self.Sam.MagHead]   = M_up_corrected
    self.Sam.Up[self.Sam.MomHead]   = Update_Moment(M_up_corrected, self.Sam.Vol, self.Sam.Mass)

    # Step 2: Correct Lower Branch
    M_lo_corrected = M_lo - params[0]*H_lo
    # Update lower branch dependencies
    self.Sam.Lo[self.Sam.FieldHead] = H_lo
    self.Sam.Lo[self.Sam.MagHead]   = M_lo_corrected
    self.Sam.Lo[self.Sam.MomHead]   = Update_Moment(M_lo_corrected, self.Sam.Vol, self.Sam.Mass)

    # Step 3: Update main structures
    self.Sam.Data = Update_Data(self.Sam.Up, self.Sam.Lo)
    self.Sam.Aux  = Update_Aux(self.Sam.Up, self.Sam.Lo, self.Sam.FieldHead, self.Sam.MagHead)
    self.Sam.Err  = Update_ErrorCurve(self.Sam.Up[self.Sam.FieldHead], self.Sam.Up[self.Sam.MagHead], self.Sam.Lo[self.Sam.MagHead], self.Sam.FieldHead, self.Sam.MagHead)
   
    # Set saturation as determined by fit
    setattr(self.Sam, 'Ms', params[1])
    logger.info("Slope correction applied. Saturation magnetization set.")

  ##############################################################################################################################################################################
  def Linear_Saturation(self):
    """
    Applies linear slope correction to both upper and lower branches of the hysteresis loop.

    High-field positive values from the upper branch are fitted to estimate the slope (χ). 
    This slope is subtracted from both branches, and dependencies are updated accordingly.
    """
    # Extract upper branch data
    H_up = self.Sam.Up[self.Sam.FieldHead]
    M_up = self.Sam.Up[self.Sam.MagHead]
    # Extract lower branch data
    H_lo = self.Sam.Lo[self.Sam.FieldHead]
    M_lo = self.Sam.Lo[self.Sam.MagHead]

    # Step 1: Select high-field positive part (≥80% of max field)
    mask_hf_up = H_up >= 0.8 * H_up.max()
    H_hf_up = H_up[mask_hf_up]
    M_hf_up = M_up[mask_hf_up]

    # Fit linear function to high-field region
    params_up = np.polyfit(H_hf_up, M_hf_up, 1)  # [slope, intercept]

    # Subtract linear contribution from magnetization
    M_up_corrected = M_up - params_up[0]*H_up
    M_lo_corrected = M_lo - params_up[0]*H_lo

    # Update upper branch dependencies
    self.Sam.Up[self.Sam.FieldHead] = H_up
    self.Sam.Up[self.Sam.MagHead]   = M_up_corrected
    self.Sam.Up[self.Sam.MomHead]   = Update_Moment(M_up_corrected, self.Sam.Vol, self.Sam.Mass)

    # Update lower branch dependencies
    self.Sam.Lo[self.Sam.FieldHead] = H_lo
    self.Sam.Lo[self.Sam.MagHead]   = M_lo_corrected
    self.Sam.Lo[self.Sam.MomHead]   = Update_Moment(M_lo_corrected, self.Sam.Vol, self.Sam.Mass)

    # Update main structures
    self.Sam.Data = Update_Data(self.Sam.Up, self.Sam.Lo)
    self.Sam.Aux  = Update_Aux(self.Sam.Up, self.Sam.Lo, self.Sam.FieldHead, self.Sam.MagHead)
    self.Sam.Err  = Update_ErrorCurve(self.Sam.Up[self.Sam.FieldHead], self.Sam.Up[self.Sam.MagHead], self.Sam.Lo[self.Sam.MagHead], self.Sam.FieldHead, self.Sam.MagHead)

    # Set saturation as determined by fit (from upper branch)
    setattr(self.Sam, 'Ms', params_up[1])
    logger.info("Linear slope correction applied only to high-field region. Saturation magnetization set.")