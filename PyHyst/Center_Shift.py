from Utilities import *
import numpy as np
from scipy.stats import linregress
from sklearn.metrics import r2_score, mean_squared_error
import logging
from logging import getLogger

logger = getLogger(__name__)

class Center_Shift:
  """
  Performs regular gridding and optionally corrects horizontal (Hoff) and vertical (Moff) shifts in magnetic hysteresis data.

  The class finds the center of symmetry through R² maximization and optionally applies the corresponding shift to the upper and lower branches.
  Regardless of correction, the data is interpolated onto a mathematically defined symmetrical grid based on the lambda parameter.
  This ensures consistent field positions for further analysis (e.g., auxiliary curves, error curves, fitting).

  References:
  - Jackson & Solheid (2010): On the quantitative analysis and evaluation of magnetic hysteresis data.
  - Von Dobeneck (1996): A systematic analysis of natural magnetic mineral assemblages based on modelling hysteresis loops with coercivity-related hyperbolic basis functions.

  Attributes
  ----------
  Sample : Read_Data object
    Contains sample data to analyze and optionally correct.
  Lambda : float or int
    Parameter controlling the exponential spacing of the regular symmetric grid.
  Apply  : bool
    If True, modifies the original sample object in-place; otherwise, works on a deep copy.
  """
  def __init__(self, Sample, Apply, Lambda=12):
    """
    Initializes Center_Shift and triggers default behavior.

    Parameters
    ----------
    Sample : Read_Data
        Object with loaded magnetic data.
    Lambda : float
        Initial Lambda value for gridding.
    Apply : bool, default=False
        If True, modifies original object. Otherwise, works on a copy.
    """
    self.Sam = Sample
    self.Lam = Lambda
    self.App = Apply

    # Initialize
    self.Error_Curve()
    
  ##############################################################################################################################################################################
  def Range(self):
    """
    Returns a symmetric range of small field values around zero.
    Used for testing horizontal shifts in center finding.
    """
    length     = self.Sam.Up[self.Sam.FieldHead].max()*0.01
    range = np.linspace(-length, length, 200)
    range = np.insert(range, int(len(range)/2), 0)
    return range

  ##############################################################################################################################################################################
  def Center(self):
    """
    Determines optimal horizontal (Hoff) and vertical (Moff) shift to achieve center symmetry.
    Returns
    -------
    BestHoff : float
        Best horizontal offset.
    BestMoff : float
        Corresponding vertical offset.
    """
    R2_values = []
    range = self.Range()
    for hoff in range:
      InvertedField = - self.Sam.Lo[self.Sam.FieldHead] - 2*hoff
      InvertedMag   = - self.Sam.Lo[self.Sam.MagHead]
      InterpolMag   = Interpolate(InvertedField, InvertedMag, self.Sam.Up[self.Sam.FieldHead])
      R2_values.append(r2_score(self.Sam.Up[self.Sam.MagHead], InterpolMag))
    # Saving corresponding values to max R2
    Best_idx    = np.argmax(R2_values)
    BestHoff    = range[Best_idx]
    BestHoff_R2 = R2_values[Best_idx]
    # Given best R2 determine Moff
    InvertedField = - self.Sam.Lo[self.Sam.FieldHead] - 2*BestHoff
    InvertedMag   = - self.Sam.Lo[self.Sam.MagHead]
    InterpolMag   = Interpolate(InvertedField, InvertedMag, self.Sam.Up[self.Sam.FieldHead])
    # Given InterpolMag, that has the magnetization of inverted lower through BestHoff, find the intercept of the line
    slope, interc, _, _, _ = linregress(self.Sam.Up[self.Sam.MagHead], InterpolMag) # The intercept of the best fit line is 2M0
    BestMoff    = interc/2
    logger.info(f"Optimal center shift found: Hoff={BestHoff}, Moff={BestMoff}, R2={BestHoff_R2:.4f}")
    return BestHoff, BestMoff

  ##############################################################################################################################################################################
  def Gridding(self):
    """
    Finds and applies an optimal lambda value to generate a regular grid of field values.
    Returns
    -------
    RegGrid : np.ndarray
        The regular field grid for interpolation.
    """
    FieldMax = self.Sam.Up[self.Sam.FieldHead].max()
    n = len(self.Sam.Up) // 2
    indices = np.arange(-n, n + 1)
    # Check for array length
    UpField = self.Sam.Up[self.Sam.FieldHead].values
    if len(UpField) != len(indices):
        min_len = min(len(UpField), len(indices))
        UpField = UpField[:min_len]
        indices = indices[:min_len]
    # Create regular grid and find optimal parameter
    lambdas = np.linspace(1, 2*self.Lam, 2000)
    rmses = np.zeros(len(lambdas))
    for i, lam in enumerate(lambdas):
        RegGrid = - (np.sign(indices) * FieldMax / lam) * ((lam + 1) ** (np.abs(indices) / n) - 1)
        rmses[i] = np.sqrt(mean_squared_error(UpField, RegGrid))
    self.Lam = lambdas[np.argmin(rmses)]
    # Using best lambda recalcualte grid
    RegGrid = - (np.sign(indices) * FieldMax / self.Lam) * ((self.Lam + 1) ** (np.abs(indices) / n) - 1)
    return RegGrid

  ##############################################################################################################################################################################
  def Correction(self):
    """
    Given center symmetry point, corrects upper and lower branches. Grids them both to a regular grid.
    """
    # Check for same branch length
    self.Sam.Up, self.Sam.Lo = Equalize_Branches_By_Interpolation(self.Sam.Up, self.Sam.Lo, self.Sam.FieldHead, self.Sam.MagHead, self.Sam.MomHead, self.Sam.Vol, self.Sam.Mass)
    # Get Center values
    hoff, moff = self.Center()
    setattr(self.Sam, 'Center', (hoff, moff))
    # Given center values shift branches
    if self.App == True:
      self.Sam.Up[self.Sam.FieldHead] += hoff
      self.Sam.Up[self.Sam.MagHead] += moff
      self.Sam.Lo[self.Sam.FieldHead] += hoff
      self.Sam.Lo[self.Sam.MagHead] += moff
    # Create regular grid and calculate interpolated magnetization with corrected branch values after center correction
    Grid = self.Gridding()
    MagUpInter = Interpolate(self.Sam.Up[self.Sam.FieldHead], self.Sam.Up[self.Sam.MagHead], Grid)
    MagLoInter = Interpolate(self.Sam.Lo[self.Sam.FieldHead], self.Sam.Lo[self.Sam.MagHead], -Grid)
    # Update branches with interpolated magnetization values at regular grid
    self.Sam.Up[self.Sam.FieldHead] = Grid
    self.Sam.Up[self.Sam.MagHead]   = MagUpInter
    self.Sam.Up[self.Sam.MomHead]   = Update_Moment(MagUpInter, self.Sam.Vol, self.Sam.Mass)
    self.Sam.Lo[self.Sam.FieldHead] = -Grid
    self.Sam.Lo[self.Sam.MagHead]   = MagLoInter
    self.Sam.Lo[self.Sam.MomHead]   = Update_Moment(MagLoInter, self.Sam.Vol, self.Sam.Mass)
    # Update all dependent values
    self.Sam.Data = Update_Data(self.Sam.Up, self.Sam.Lo)
    self.Sam.Aux  = Update_Aux(self.Sam.Up, self.Sam.Lo, self.Sam.FieldHead, self.Sam.MagHead)
    logger.info("Branches center corrected and gridded. Data and auxiliary curves updated.")

  ################################################################################################################################################################################
  def Error_Curve(self):
    """
    Calculates the error curve for the sample after (optional) center correction and gridding.

    Parameters
    ----------
    shift_center : bool, default=True
        If True, apply horizontal and vertical center correction (hoff, moff).
        If False, skip shifting and only perform gridding/interpolation.
    """
    # Perform correction with or without center shift
    self.Correction()

    # Compute error curve after correction and gridding
    ErrorCurve = Update_ErrorCurve(
        self.Sam.Up[self.Sam.FieldHead],
        self.Sam.Up[self.Sam.MagHead],
        self.Sam.Lo[self.Sam.MagHead],
        self.Sam.FieldHead,
        self.Sam.MagHead)

    # Attach error curve to sample object
    setattr(self.Sam, 'Err', ErrorCurve)
    logger.info("Error curve calculated and stored.")