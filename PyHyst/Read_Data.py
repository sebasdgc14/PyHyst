from Utilities import *
import re
import pandas as pd
import pint
import logging
from logging import getLogger

logger = getLogger(__name__)

class Read_Data:
  """
  Read_Data:
  Handles the extraction and formatting of hysteresis data from `.DAT` files.
  It processes sample metadata (volume, mass), cleans data, calculates magnetization, and splits branches.

  References:
  - Jackson & Solheid (2010): On the quantitative analysis and evaluation of magnetic hysteresis data.
  - Von Dobeneck (1996): A systematic analysis of natural magnetic mineral assemblages based on modelling hysteresis loops with coercivity-related hyperbolic basis functions.


  Attributes:
      Path (str): Path to the .DAT file.
      Method (str): Method used for branch splitting ('Diff' or 'Zero').
      Volume (float or None): Volume of the sample.
      Mass (float or None): Mass of the sample.
      VolStr (str): Key string for volume identification in the file.
      MassStr (str): Key string for mass identification in the file.
      VolPatt (re.Pattern): Regex pattern to extract volume.
      MassPatt (re.Pattern): Regex pattern to extract mass.
      VolUnit (str): Unit for volume (default: 'cm ** 3').
      MassUnit (str): Unit for mass (default: 'g').
      FieldHead (str): Field column header in .DAT file.
      MomHead (str): Moment column header in .DAT file.
      MagHead (str): Computed magnetization column name.
      Data (DataFrame): Cleaned and processed data.
      Up (DataFrame): Upper branch of the hysteresis loop.
      Lo (DataFrame): Lower branch of the hysteresis loop.
  """
  def __init__(self, Path, Method, 
               Volume=None, Mass=None, 
               VolStr=None, MassStr=None, 
               VolPatt=None, MassPatt=None, 
               VolUnit=False, MassUnit=False,
               TargetVolUnit=False, TargetMassUnit=False, 
               FieldUnit=False, MomUnit=False,
               TargetFieldUnit=False, TargetMomUnit=False,
               FieldHead=False, MomHead=False):
    # Set path to data
    self.Path   = Path
    self.Method = Method
    # Setup default volume/mass search string and pattern, as well as units and header for search in path
    self.VolStr    = VolStr    if VolStr    else 'SAMPLE_VOLUME'
    self.MassStr   = MassStr   if MassStr   else 'SAMPLE_MASS'
    self.VolPatt   = VolPatt   if VolPatt   else  re.compile(r'(\d+\.?\d*),\s*' + re.escape(self.VolStr))
    self.MassPatt  = MassPatt  if MassPatt  else  re.compile(r'(\d+\.?\d*),\s*' + re.escape(self.MassStr))
    self.VolUnit   = VolUnit   if VolUnit   else 'cm ** 3'
    self.MassUnit  = MassUnit  if MassUnit  else 'g'
    self.TargetVolUnit   = TargetVolUnit   if TargetVolUnit   else 'cm ** 3'
    self.TargetMassUnit  = TargetMassUnit  if TargetMassUnit  else 'g'
    self.FieldUnit = FieldUnit if FieldUnit else 'Oe'
    self.MomUnit   = MomUnit   if MomUnit   else 'emu'
    self.TargetFieldUnit = TargetFieldUnit if TargetFieldUnit else 'Oe'
    self.TargetMomUnit   = TargetMomUnit   if TargetMomUnit   else 'emu'
    # Setup default headers for field and moment
    self.FieldHead = FieldHead if FieldHead else 'Magnetic Field (Oe)'
    self.MomHead   = MomHead   if MomHead   else 'Moment (emu)'
    self.MagHead   = None
    # Initialize sample's info
    self.Data = pd.DataFrame()
    self.Vol  = Volume if Volume else None
    self.Mass = Mass   if Mass   else None
    # Run as instantiated
    self.Data_Clean()

###########################################################################################################################################################################################################################
  def Vol_Mass_Search(self):
    """
    Searches the .DAT file for sample volume and mass using predefined or user-specified
    strings and regex patterns.

    Side Effects:
      - Updates self.Vol and/or self.Mass.
      - Prints status messages if values are not found.
    """
    # Path exploration
    with open(self.Path, 'r', errors='ignore') as file:
      for linenum, line in enumerate(file, start=1):
        match1 = self.VolPatt.search(line)
        match2 = self.MassPatt.search(line)
        if match1 and self.Vol is None  : self.Vol  = float(match1.group(1))
        elif match2 and self.Mass is None : self.Mass = float(match2.group(1))
        if self.Vol is not None and self.Mass is not None: break
    if   self.Vol  is None and self.Mass        : logger.warning('Mass found but not volume')
    elif self.Mass is None and self.Vol         : logger.warning('Volume found but not mass')
    elif self.Mass is None and self.Vol is None : logger.warning('No volume or mass found')

###########################################################################################################################################################################################################################
  def Mass_Volume_Conversion(self):
    """
    Converts sample volume to cm³ and mass to g using pint's unit registry.

    Requirements:
        - The 'pint' package must be installed.
        - Units must be interpretable by pint.

    Side Effects:
        - Converts and updates self.Vol and/or self.Mass in-place.
    """
    ureg = pint.UnitRegistry()
    try:
      ureg.define('emu = 1e-3 * A * m**2 = electromagnetic_moment')
    except pint.errors.DefinitionSyntaxError:
      pass  # Already defined
    if self.VolUnit and self.TargetVolUnit and self.Vol:
      self.Vol = (self.Vol * ureg(self.VolUnit)).to(ureg(self.TargetVolUnit)).magnitude
    if self.MassUnit and self.TargetMassUnit and self.Mass:
      self.Mass = (self.Mass * ureg(self.MassUnit)).to(ureg(self.TargetMassUnit)).magnitude

###########################################################################################################################################################################################################################
  def Field_Moment_Conversion(self):
    """
    Converts field and moment units using pint's unit registry.

    Requirements:
        - The 'pint' package must be installed.
        - Units must be interpretable by pint.

    Side Effects:
        - Updates self.and self.MomHead with target units.
    """
    ureg = pint.UnitRegistry()
    try:
      ureg.define('emu = 1e-3 * A * m**2 = electromagnetic_moment')
    except pint.errors.DefinitionSyntaxError:
      pass 
    try:
      ureg.define('Oe = 1e-3 * A / m = oersted')
    except pint.errors.DefinitionSyntaxError:
      pass
    if self.FieldUnit and self.TargetFieldUnit:
      self.Data[self.FieldHead] = (self.Data[self.FieldHead].values * ureg(self.FieldUnit)).to(ureg(self.TargetFieldUnit)).magnitude
      self.Data[self.FieldHead] = self.Data[self.FieldHead].astype(float)
    if self.MomUnit and self.TargetMomUnit:
      self.Data[self.MomHead] = (self.Data[self.MomHead].values * ureg(self.MomUnit)).to(ureg(self.TargetMomUnit)).magnitude
      self.Data[self.MomHead] = self.Data[self.MomHead].astype(float)

########################################################################################################################################################################################################################### 
  def Data_Extraction(self):
    """
    Extracts moment and field data from the .DAT file, calculates magnetization
    based on available volume or mass, and populates self.Data.

    Workflow:
      - Searches for volume/mass.
      - Converts units.
      - Reads table starting from the header row.
      - Adds magnetization column to the DataFrame.

    Side Effects:
      - Updates self.Data with relevant columns.
      - Sets self.MagHead according to normalization (by Vol or Mass).
      - Prints warnings if headers are not found.
    """
    # Run requisite methods
    self.Vol_Mass_Search()
    self.Mass_Volume_Conversion()

    # Look for header matches
    row = None
    with open(self.Path, 'r', errors='ignore') as file:
      for linenum, line in enumerate(file, start=1):
        if self.FieldHead in line and self.MomHead in line:
          row = linenum - 1
          break
      if row is None: logger.error('Headers not found in same line')

    # Read Table
    try                   : data = pd.read_table(self.Path, sep=',', header = row, encoding='latin1')
    except Exception as e : logger.error(f"Error reading the table: {e}")

    # Extraction
    if self.FieldHead in data.columns and self.MomHead in data.columns:
      self.Data = data[[self.FieldHead, self.MomHead]].copy()
    else: logger.error('Headers not found in provided file')

    # Convert units
    self.Field_Moment_Conversion()

    # Add Magnetization to sample data
    if self.Vol:
      self.MagHead = f'Magnetization ({self.TargetMomUnit}/{self.TargetVolUnit})'
      self.Data[self.MagHead] = self.Data[self.MomHead] / self.Vol
    elif self.Mass:
      self.MagHead = f'Magnetization ({self.TargetMomUnit}/{self.TargetMassUnit})'
      self.Data[self.MagHead] = self.Data[self.MomHead] / self.Mass
    else:
      self.MagHead = self.MomHead
      self.Data[self.MagHead] = self.Data[self.MomHead]
      
###########################################################################################################################################################################################################################
  def Data_Clean(self):
    """
    Cleans and formats the extracted data by:
        - Removing NaN values.
        - Resetting index.
        - Splitting the data into upper and lower branches.
        - Interpolating for equal field alignment.
        - Merging back into a cleaned dataset.

    Side Effects:
        - Updates self.Data, self.Up, and self.Lo.
        - Ensures uniform sampling across branches.
    """
    self.Data_Extraction()
    self.Data        = self.Data.dropna().reset_index(drop=True)
    # Splitting and dimension checking
    self.Up, self.Lo = Update_Branches(self.Data, self.FieldHead, self.Method)
    self.Up, self.Lo = Equalize_Branches_By_Interpolation(self.Up, self.Lo, self.FieldHead, self.MagHead, self.MomHead, self.Vol, self.Mass)
    # Concatenating branches into a new data
    self.Data = Update_Data(self.Up, self.Lo)