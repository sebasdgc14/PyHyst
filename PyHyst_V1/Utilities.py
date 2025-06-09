import pandas as pd
from scipy.interpolate import interp1d
import logging
from logging import getLogger

logger = getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Branch Splitting and Dimension Matching
# --------------------------------------------------------------------------------------------------
def Branch_Dimensions(Up, Lo, Data, FieldHead):
  """
  Ensures that 'Up' and 'Lo' hysteresis branches have matching lengths.
  If they differ by 1 point, removes a redundant point based on a threshold criterion.
  Raises an error if the difference is greater than 1.

  Parameters:
    Up (DataFrame): Upper branch.
    Lo (DataFrame): Lower branch.
    Data (DataFrame): Original dataset.
    FieldHead (str): Column name for magnetic field.

  Returns:
    tuple: (Up, Lo) DataFrames with equal lengths.
  """
  if len(Up) != len(Lo):
    diff = len(Up) - len(Lo)
    if abs(diff)==1:
      logger.info(f'Method chosen for splitting branches leads to a difference in size of: {diff}')
      logger.info('Now removing redundant data from largest branch to equate lengths...')
      if diff > 0:
        Threshold = Data[FieldHead].max()*0.25
        IndexThre = (Data.loc[Up.index, FieldHead] - Threshold).abs().idxmin()
        Up = Up.drop(index=IndexThre).reset_index(drop=True)
      if diff < 0:
        Threshold = Data[FieldHead].min()*0.25
        IndexThre = (Data.loc[Lo.index, FieldHead] + Threshold).abs().idxmin()
        Lo = Lo.drop(index=IndexThre).reset_index(drop=True)
    else:
      raise ValueError('Method chosen for splitting branches leads to a difference in size bigger than 1, check input data' + str(diff)) # Raise ValueError
    return Up, Lo
  else:
    return Up, Lo

def Branch_Splitting(Data, FieldHead, Method):
  """
  Splits the hysteresis curve into upper and lower branches.

  Parameters:
    Data (DataFrame): Full hysteresis dataset.
    FieldHead (str): Column name of magnetic field.
    Method (str): Splitting method, either 'Diff' or 'Zero'.

  Returns:
    tuple: (Up, Lo) branches as DataFrames.
  """
  try:
    if Method == 'Diff':
      logger.info('Splitting branches using the difference method...')
      Up = Data[Data[FieldHead].diff() < 0].reset_index(drop=True)
      Lo = Data[Data[FieldHead].diff() > 0].reset_index(drop=True)
    elif Method == 'Zero':
      logger.info('Splitting branches using the zero-crossing method...')
      MinIndex = Data[FieldHead].idxmin()
      Up = Data.iloc[:MinIndex+1].reset_index(drop=True)
      Lo = Data.iloc[MinIndex+1:].reset_index(drop=True)
    else:
      raise ValueError("Invalid Method. Use 'Diff' or 'Zero'.")

    if abs(len(Up) - len(Lo)) <= 1:
      Up, Lo = Branch_Dimensions(Up, Lo, Data, FieldHead)
      return Up, Lo
    else:
      logger.info('Method chosen for splitting branches leads to a difference in size of: ', abs(len(Up) - len(Lo)))
      logger.info('Now equating branch lengths through interpolation at the larger branch fields')
      return Up, Lo

  except ValueError as e:
      logger.error(f"Error during branch splitting: {e}")
      return None, None

def Equalize_Branches_By_Interpolation(Up, Lo, FieldHead, MagHead, MomHead, Volume=None, Mass=None):
  """
  Interpolates the smaller branch to match the field sampling of the larger one.
  Updates magnetization and moment using volume or mass.

  Parameters:
    Up, Lo (DataFrame): Upper and lower branches.
    FieldHead (str): Field column name.
    MagHead (str): Magnetization column name.
    MomHead (str): Moment column name.
    Volume (float, optional): Sample volume.
    Mass (float, optional): Sample mass.

  Returns:
    tuple: Interpolated (Up, Lo) DataFrames.
  """
  if len(Up) == len(Lo):
    return Up.reset_index(drop=True), Lo.reset_index(drop=True)
  # Determine which branch is smaller
  elif len(Up) > len(Lo):
    NewMag = Interpolate(Lo[FieldHead], Lo[MagHead], Up[FieldHead])
    NewMom = Update_Moment(NewMag, Volume, Mass)
    Lo_new = pd.DataFrame({FieldHead: Up[FieldHead], MomHead: NewMom, MagHead: NewMag})
    Lo_new = Lo_new[::-1].reset_index(drop=True)
    return Up.reset_index(drop=True), Lo_new
  elif len(Up) < len(Lo):
    NewMag = Interpolate(Up[FieldHead], Up[MagHead], Branch_Reversion(Lo[FieldHead]))
    NewMom = Update_Moment(NewMag, Volume, Mass)
    Up_new = pd.DataFrame({FieldHead: Lo[FieldHead], MomHead: NewMom, MagHead: NewMag})
    Up_new = Up_new[::-1].reset_index(drop=True)
    return Up_new, Lo.reset_index(drop=True)

# --------------------------------------------------------------------------------------------------
# Data Updating and Transformation
# --------------------------------------------------------------------------------------------------
def Update_Branches(Data, FieldHead, Method):
  """
  Wrapper to return Up and Lo branches from a full dataset.

  Parameters:
    Data (DataFrame): Full hysteresis dataset.
    FieldHead (str): Field column name.
    Method (str): Branch split method.

  Returns:
    tuple: (Up, Lo) branches.
  """
  Up, Lo = Branch_Splitting(Data, FieldHead, Method)
  return Up, Lo

def Update_Data(Up, Lo):
  """
  Combines Up and Lo branches into one DataFrame.

  Parameters:
    Up, Lo (DataFrame): Branches to combine.

  Returns:
    DataFrame: Combined dataset.
  """
  Data = pd.concat([Up, Lo]).reset_index(drop=True)
  return Data

def Update_Moment(Magnetization, Volume, Mass):
  """
  Computes moment from magnetization using volume or mass.

  Parameters:
    Magnetization (array-like): Magnetization values.
    Volume (float, optional): Sample volume.
    Mass (float, optional): Sample mass.

  Returns:
    array-like: Computed magnetic moment.
  """
  if Volume is not None : Moment = Magnetization * Volume
  elif Mass is not None : Moment = Magnetization * Mass
  else      : Moment = Magnetization
  return Moment

def Update_Aux(Up, Lo, FieldHead, MagHead):
  """
  Calculates auxiliary Mih and Mrh curves from Up and Lo branches.

  Parameters:
    Up, Lo (DataFrame): Upper and lower branches.
    FieldHead (str): Field column name.
    MagHead (str): Magnetization column name.

  Returns:
    DataFrame: DataFrame containing Field, Mrh, and Mih.
  """
  # As auxiliary curves are defined for same fields, then a reversion must be done on Lo branch
  Mih = (Up[MagHead] + Branch_Reversion(Lo)[MagHead])*0.5
  Mrh = (Up[MagHead] - Branch_Reversion(Lo)[MagHead])*0.5
  Aux = pd.DataFrame({FieldHead: Up[FieldHead], 'Mrh': Mrh, 'Mih': Mih})
  return Aux

def Update_ErrorCurve(FieldUp, MagUp, MagLo, FieldHead, MagHead):
  """
  Computes the error curve between upper and inverted lower branches.

  Parameters:
    FieldUp (array-like): Field values of Up branch.
    MagUp (array-like): Magnetization of Up.
    MagLo (array-like): Magnetization of Lo.
    FieldHead (str): Name of field column in result.
    MagHead (str): Name of magnetization column in result.

  Returns:
    DataFrame: Error curve DataFrame.
  """
  MagIL = - MagLo
  error = MagUp - MagIL
  ErrorCurve = pd.DataFrame({FieldHead: FieldUp, MagHead: error})
  return ErrorCurve

# --------------------------------------------------------------------------------------------------
# Branch Inversion and Reversion
# --------------------------------------------------------------------------------------------------
def Branch_Inversion(Branch):
  """
  Inverts the branch (multiplies by -1).

  Parameters:
    Branch (Series or DataFrame): Input data.

  Returns:
    Same type: Inverted branch.
  """
  return Branch * -1

def Branch_Reversion(Branch):
  """
  Reverses the branch order and resets the index.

  Parameters:
    Branch (Series or DataFrame): Input data.

  Returns:
    Same type: Reversed branch.
  """
  return Branch.copy().iloc[::-1].reset_index(drop=True)

# --------------------------------------------------------------------------------------------------
# Interpolation Utility
# --------------------------------------------------------------------------------------------------
def Interpolate(Field, Mag, NewField):
  """
  Interpolates magnetization to a new field grid.

  Parameters:
    Field (array-like): Original field values.
    Mag (array-like): Original magnetization values.
    NewField (array-like): New field grid.

  Returns:
    array-like: Interpolated magnetization.
  """
  Interpolator = interp1d(Field, Mag, kind='linear', fill_value='extrapolate')
  InterpolMag  = Interpolator(NewField)
  return InterpolMag
