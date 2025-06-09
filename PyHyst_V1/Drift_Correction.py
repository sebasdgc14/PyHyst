from Utilities import *
import numpy as np
import logging
from logging import getLogger

logger = getLogger(__name__)

class Drift_Correction:
  """
  Applies drift correction methods to magnetic sample data.

  This class provides functionality to correct measurement drift in magnetization
  experiments. It supports both automatic and manual drift correction techniques
  based on methodologies described in J&S (2010) and HystLab.

  References:
  - Jackson & Solheid (2010): On the quantitative analysis and evaluation of magnetic hysteresis data.
  - Von Dobeneck (1996): A systematic analysis of natural magnetic mineral assemblages based on modelling hysteresis loops with coercivity-related hyperbolic basis functions.

  Attributes:
  ----------
  Sample  : Read_Data object
      Stores sample data, either original or a deep copy (depending on `Apply` flag).
  Apply  : bool
      Determines if corrections are applied to the original sample or a copy.
  Mode : str
      Defines drift correction mode, either "Auto" or "Manual".
  """
  def __init__(self, Sample, Mode):
    """
    Initializes the drift correction object.

    Parameters
    ----------
    Sample : Read_Data
        Sample data containing magnetic measurements.
    Apply : bool
        Flag to apply corrections directly to original sample data.
    Mode : str
        Correction mode: 'Auto', 'Symmetric', 'Positive', 'Upper', or 'None'.

    Automatically computes closure error and drift distribution to select
    or perform the appropriate correction method.
    """

    self.Sam  = Sample
    self.Mode = Mode

    # Now a routine for automatic drift correction
    self.Closure_Error()      # Closure error
    self.Drift_Distribution() # Calculate drift ratio
    self.McePercent = abs(self.Mce /self.Sam.Data[self.Sam.MagHead].max())*100
    logger.info(f"Closure error: {self.Mce:.4e}, Percent of max magnetization: {self.McePercent:.2f}%")

    # Now initiation
    if self.Mode   == 'Auto'     : self.Auto_Correction()
    elif self.Mode == 'Symmetric': self.SymmetricAveraging_TipToTipClosure()
    elif self.Mode == 'Positive' : self.Pos_Field_Correction()
    elif self.Mode == 'Upper'    : self.Upper_Branch_Correction()
    elif self.Mode == 'None'     : logger.info("Drift correction skipped (mode: None)")

  ##############################################################################################################################################################################
  def Auto_Correction(self):
    """
    Performs automatic drift correction based on closure error and drift distribution.

    - If closure error percentage ≤ 5%, applies symmetric averaging with tip-to-tip closure.
    - Otherwise, applies positive field correction.

    This method implements logic from Jackson & Solheid (2010) for automated drift correction.
    """
    if self.McePercent <= 5:
      logger.info("Applying Symmetric Averaging Tip-To-Tip Closure (Auto mode)")
      self.SymmetricAveraging_TipToTipClosure()
    else:
      logger.info("Applying Positive Field Correction (Auto mode)")
      self.Pos_Field_Correction()
    # Upper branch correction forces perfect symmetry and usually does not produce reasonable results, will only be availabe for manual correction

  ##############################################################################################################################################################################
  def Drift_Distribution(self):
    """
    Calculates the drift distribution ratio in the sample data.

    Computes the ratio of median drift in high-field to low-field regions,
    following Jackson & Solheid (2010) and HystLab criteria.

    Sets:
    -----
    self.DR : float
        Ratio of high-field to low-field drift magnitude.
    """
    HFT = 0.75*self.Sam.Data[self.Sam.FieldHead].max()                             # High-field threshold
    HFD = abs(self.Sam.Err[abs(self.Sam.Err[self.Sam.FieldHead])>HFT])             # High Field drift
    LFD = abs(self.Sam.Err[abs(self.Sam.Err[self.Sam.FieldHead])<=HFT])            # Low Field drift
    DriftRatio = np.median(HFD[self.Sam.MagHead])/np.median(LFD[self.Sam.MagHead]) # definition taken from HystLab
    self.DR = DriftRatio
    logger.info(f"Drift distribution ratio (high/low field): {self.DR:.3f}")

  ##############################################################################################################################################################################
  def Closure_Error(self):
    """
    Calculates the closure error (Mce) of the hysteresis loop.

    The closure error is the difference in magnetization between the
    highest and lowest applied magnetic fields.

    Sets:
    -----
    self.Mce : float
        Closure error magnitude.
    """
    H1  = self.Sam.Err[self.Sam.FieldHead].idxmax() # Highest field
    HN  = self.Sam.Err[self.Sam.FieldHead].idxmin() # Lowest field
    Mce = self.Sam.Err[self.Sam.MagHead][H1] - self.Sam.Err[self.Sam.MagHead][HN]
    self.Mce = Mce
    
  ##############################################################################################################################################################################
  def SymmetricAveraging_TipToTipClosure(self):
    """
    Applies symmetric averaging drift correction with tip-to-tip closure.

    Implements the method of Von Dobeneck (1996) and Jackson & Solheid (2010) for cases
    where distortions occur at intermediate fields and closure error is near zero.

    Adjusts upper and lower branches to reduce drift-induced asymmetries and updates
    all dependent data structures.
    """
    Var = (self.Sam.Up[self.Sam.MagHead]-self.Sam.Lo[self.Sam.MagHead].iloc[::-1])*0.5
    A = self.Sam.Up.loc[self.Sam.Up[self.Sam.FieldHead].idxmax(), self.Sam.MagHead]
    B = self.Sam.Lo.loc[self.Sam.Lo[self.Sam.FieldHead].idxmax(), self.Sam.MagHead]
    C = self.Sam.Up.loc[self.Sam.Up[self.Sam.FieldHead].idxmin(), self.Sam.MagHead]
    D = self.Sam.Lo.loc[self.Sam.Lo[self.Sam.FieldHead].idxmin(), self.Sam.MagHead]
    Fix = (A - B + C - D) * 0.25
    self.Sam.Up[self.Sam.MagHead] = Var - Fix
    self.Sam.Lo[self.Sam.MagHead] = - self.Sam.Up[self.Sam.MagHead].iloc[::-1]
    # Update dependencies
    self.Sam.Up[self.Sam.MomHead] = Update_Moment(self.Sam.Up[self.Sam.MagHead], self.Sam.Vol, self.Sam.Mass)
    self.Sam.Lo[self.Sam.MomHead] = Update_Moment(self.Sam.Lo[self.Sam.MagHead], self.Sam.Vol, self.Sam.Mass)
    self.Sam.Data = Update_Data(self.Sam.Up, self.Sam.Lo)
    self.Sam.Aux  = Update_Aux(self.Sam.Up, self.Sam.Lo, self.Sam.FieldHead, self.Sam.MagHead)
    self.Sam.Err  = Update_ErrorCurve(self.Sam.Up[self.Sam.FieldHead], self.Sam.Up[self.Sam.MagHead], self.Sam.Lo[self.Sam.MagHead], self.Sam.FieldHead, self.Sam.MagHead)
    logger.info("Symmetric averaging correction applied.")

  ##############################################################################################################################################################################
  def Pos_Field_Correction(self):
    """
    Applies positive field drift correction.

    Based on Jackson & Solheid (2010), this correction modifies only the positive field
    regions to better estimate key magnetic parameters when drift errors dominate at large fields.

    Updates all dependent data after correction.
    """
    # Filtering positive fields from upper and lower branch
    PosUp = self.Sam.Up[self.Sam.Up[self.Sam.FieldHead]>=0]
    PosLo = self.Sam.Lo[self.Sam.Lo[self.Sam.FieldHead]>=0]
    # Filtering error curve
    PosErr =  self.Sam.Err[self.Sam.Err[self.Sam.FieldHead]>=0]
    NegErr =  self.Sam.Err[self.Sam.Err[self.Sam.FieldHead]<=0]
    # Correcting Original dataframe
    self.Sam.Up.loc[self.Sam.Up[self.Sam.FieldHead] >= 0, self.Sam.MagHead] = (PosUp[self.Sam.MagHead] - PosErr[self.Sam.MagHead])
    self.Sam.Lo.loc[self.Sam.Lo[self.Sam.FieldHead] >= 0, self.Sam.MagHead] = (PosLo[self.Sam.MagHead] - NegErr[self.Sam.MagHead])
    # Update dependencies
    self.Sam.Up[self.Sam.MomHead] = Update_Moment(self.Sam.Up[self.Sam.MagHead], self.Sam.Vol, self.Sam.Mass)
    self.Sam.Lo[self.Sam.MomHead] = Update_Moment(self.Sam.Lo[self.Sam.MagHead], self.Sam.Vol, self.Sam.Mass)
    self.Sam.Data = Update_Data(self.Sam.Up, self.Sam.Lo)
    self.Sam.Aux  = Update_Aux(self.Sam.Up, self.Sam.Lo, self.Sam.FieldHead, self.Sam.MagHead)
    self.Sam.Err  = Update_ErrorCurve(self.Sam.Up[self.Sam.FieldHead], self.Sam.Up[self.Sam.MagHead], self.Sam.Lo[self.Sam.MagHead], self.Sam.FieldHead, self.Sam.MagHead)
    logger.info("Positive field correction applied.")

  ##############################################################################################################################################################################
  def Upper_Branch_Correction(self):
    """
    Applies drift correction by adjusting only the upper branch magnetization.

    Used when drift is primarily contained in lower field regions and perfect symmetry
    is enforced by correcting the upper branch data.

    Updates all dependent data after correction.
    """
    self.Sam.Up[self.Sam.MagHead] = self.Sam.Up[self.Sam.MagHead] - self.Sam.Err[self.Sam.MagHead]
    # Update dependencies
    self.Sam.Up[self.Sam.MomHead] = Update_Moment(self.Sam.Up[self.Sam.MagHead], self.Sam.Vol, self.Sam.Mass)
    self.Sam.Data = Update_Data(self.Sam.Up, self.Sam.Lo)
    self.Sam.Aux  = Update_Aux(self.Sam.Up, self.Sam.Lo, self.Sam.FieldHead, self.Sam.MagHead)
    self.Sam.Err  = Update_ErrorCurve(self.Sam.Up[self.Sam.FieldHead], self.Sam.Up[self.Sam.MagHead], self.Sam.Lo[self.Sam.MagHead], self.Sam.FieldHead, self.Sam.MagHead)
    logger.info("Upper branch correction applied.")
