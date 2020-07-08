# ISCE
Code associated with radar processing software ISCE

#### topsrun.py
Batch process radar data with JPL software ISCE. Can define either a maximum time step between acquisitions to be paired, a single master that gets paired with all other acquisitions, or pairing all acquisitions (N x N).

#### tops.py
Defines python class object Pair and associated methods

#### isce2png_tiff.py
Convert interferograms or coherence images output by ISCE into png with associated kml file or geotiff.

#### insarhelpers.py
Set of functions that can be used for insar processing. Includes converting phase to deformation, extracting number of days between interferograms, getting geographical extent of radar data, etc.

#### move_faulty.py
Identify ISCE pairs that did not process and move to target directory

#### move_dirs.py
Set of functions to move directories based on a list in txt files

### cleanup.py
uses move_dirs.py and files created by move_faulty.py to clean up directories. Probably broken.
