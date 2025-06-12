from Utilities import *
import logging
from logging import getLogger

logger = getLogger(__name__)

class Auxiliary_Curves:
  """
  Receives a Read_Data object and computes auxiliary magnetic hysteresis curves:
  - Induced Hysteretic Magnetization (Mih)
  - Remanent Hysteretic Magnetization (Mrh)

  These are calculated from the upper and lower hysteresis branches using:
    Mih = (Up + reversed(Lo)) / 2
    Mrh = (Up - reversed(Lo)) / 2

  The resulting curves are stored back in the sample object under the attribute `Aux`.

  References:
  - Jackson & Solheid (2010): On the quantitative analysis and evaluation of magnetic hysteresis data.
  - Von Dobeneck (1996): A systematic analysis of natural magnetic mineral assemblages based on modelling hysteresis loops with coercivity-related hyperbolic basis functions.

  Attributes
  ----------
  Sam : Read_Data
      Instance of the Read_Data class containing cleaned magnetization data.

  """
  def __init__(self, Sample):
    """
    Initializes the Auxiliary_Curves object by receiving a Read_Data instance
    and automatically performing auxiliary curve calculation.

    Parameters
    ----------
    Sample : Read_Data
        Instance of the class holding processed sample data.
    """
    self.Sam = Sample

    # Run as instantiated
    self.Calculation()

  ##############################################################################################################################################################################
  def Calculation(self):
    """
    Calculates Mih and Mrh using Up and Lo branches.
    Stores results as a new DataFrame in Sample.Aux with columns:
    - Magnetic Field (Oe)
    - Mrh (Remanent Hysteretic Magnetization)
    - Mih (Induced Hysteretic Magnetization)
    """
    try:
      Aux = Update_Aux(self.Sam.Up, self.Sam.Lo, self.Sam.FieldHead, self.Sam.MagHead)
      # Set new attribute to sample data
      setattr(self.Sam, 'Aux', Aux)
      logger.info("Auxiliary curves (Mih, Mrh) calculated and stored successfully.")
    except Exception as e:
      logger.error(f"Error in calculating auxiliary curves: {e}")