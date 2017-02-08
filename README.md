# SdHAnalysis
___________________________________________
#### Data processing pipeline for analysis of quantum oscillations.
-----------------------
SdHAnalysis is a collection of Python functions for analyzing Shubnikov-de Haas oscillations measured in pulsed and dc magnetic fields. A dictionary is created for each field sweep/pulse, and this dictionary contains the original data and any additional data created throughout the analysis. Currently, it is assumed that the resistivity is measured using a tunnel diode oscillator (i.e. SdH manifests as oscillations in frequency as a function of inverse field).
##### Data processing steps:
1. Import data, clean it if necessary, and wrap it in a dict.
2. Invert the magnetic field, spline fit to get evenly spaced points, and subtract a polynomial (in inverse field) magnetoresistance background signal.
3. Identify peaks in the FFT corresponding to SdH and magnetic breakdown orbits, and any mixing signals or harmonics.
4. Filter the oscillatory magnetoresistance signal to isolate each of the orbits of interest.
5. Calculate the amplitude of magnetoresistance oscillations as a function of inverse field.
6. Fit the data to theoretical models to extract materials parameters like effective mass, g-factor, Dingle temperature, and magnetic breakdown field.

SdH oscillations at many temperatures are needed to calculate effective mass (using fits to the Lifshitz–Kosevich formula). I hope to implement this calculation soon.

Calculation of the Dingle temperature and magnetic breakdown field is in general not possible from SdH unless one of the two is known from a separate measurement. I don't when I'll get around to implementing this calculation in some form.
##### Dependencies:
- [`numpy`](http://www.numpy.org)
- [`scipy`](https://www.scipy.org)
- [`pandas`](http://pandas.pydata.org)
- [`matplotlib`](http://matplotlib.org)
- [`detect_peaks`](https://github.com/demotu/BMC/blob/master/functions/detect_peaks.py) (author: Marcos Duarte)
- For use in [Jupyter](http://jupyter.org) notebooks

Author: Logan Bishop-Van Horn (2017)