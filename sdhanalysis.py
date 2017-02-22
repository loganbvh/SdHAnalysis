import numpy as np
import pandas as pd
from numpy import fft

from scipy import interpolate
from scipy import optimize
from scipy import signal

from detect_peaks import detect_peaks

import matplotlib.pyplot as plt

""" Data processing pipeline for analysis of SdH in pulsed fields.
    Logan Bishop-Van Horn (2017)
"""

###################################################################################################
#                                                                                                 #
###################################################################################################
       
class SdHDataSet:
    """ SdHDataSet handles SdH data and analysis for a single magnet sweep/pulse.
    Attributes:
        name: (string) name/description of dataset
        date: (string) date data was taken
        UpRaw: (pandas DataFrame) raw up-sweep data
        DownRaw: (pandas DataFrame) raw down-sweep data
        Upcs: (pandas DataFrame) spline fit/background subtracted
            up-sweep data
        Downcs: (pandas DataFrame) spline fit/backgroudn subtracted
            down-sweep data
        nskip: (int) number of points to skip at beginning of signal
        FFTpeaks: (pandas DataFrame) FFT peak amplitudes and locations (1/B)
        Orbits: (dict) contains data specific to a given orbit/frequency
            Orbits keys:
                'Osc': (pandas DataFrame) oscillations vs. 1/B
                'Peaks': (pandas DataFrame) oscillation amplitude vs. 1/B
    Methods:
        load_data() (Adapt this method for your particular dataset)
        subtract_background()
        get_fft_peaks()
        isolate_orbit()
        get_peak_amplitudes()
    """
    def __init__(self, name, date):
        self.name = name
        self.date = date
        self.Orbits = {}
        
    def load_data(self, year, num, plot=True):
        self.UpRaw, self.DownRaw = sdh_load(year, num, plot=plot)
    
    def subtract_background(self, deg, Bmin, Bmax, npnts=2**13, plot=True, yscale=1e5, save=False):
        self.Upcs, self.Downcs = back_subtract(self.UpRaw, self.DownRaw, deg, Bmin, Bmax,
                                               npnts=npnts, plot=plot, yscale=yscale, save=save)
        
    def get_fft_peaks(self, nskip=100, mph=None, mpd=1,
                  threshold=0, edge='rising', kpsh=False, valley=False,
                  show=True, ax=None, nignore=5, xmax=None, keep_ind=None, save=False):
        self.nskip = nskip
        self.FFTpeaks = fft_peaks(self.Downcs, nskip=nskip, mph=mph, mpd=mpd,
                                             threshold=threshold, edge=edge, kpsh=kpsh,
                                             valley=valley, show=show, ax=ax, nignore=nignore,
                                             xmax=xmax, keep_ind=keep_ind, save=save)
    
    def isolate_orbit(self, orbit, center_freq, passband, order=2, method='gust', plot=True, save=False):
        df_orbitdata = filter_orbit(self.Downcs, center_freq, passband, orbit,
                                  self.nskip, order=order, method=method, plot=plot, save=save)
        self.Orbits.update({orbit: {'Osc': df_orbitdata}})
        
    def get_peak_amplitudes(self, orbit, show=True, save=False):
        orbit_dict = self.Orbits[orbit]
        df_data = self.Downcs
        df_peaks = peak_amplitudes(orbit_dict, orbit, self.nskip, show=show, save=save)
        self.Orbits[orbit].update({'Peaks': df_peaks})
        
    def plot_orbit_amplitudes(self):
        orbits = self.Orbits
        for orbit in sorted(orbits.keys()):
            plt.plot(orbits[orbit]['Peaks'].InvField, 1e-3*orbits[orbit]['Peaks'].Amp, lw=2, label=orbit)
        plt.legend(loc=0)
        plt.xlabel(r'Inverse Field (T${}^{-1}$)')
        plt.ylabel('Amplitude (kHz)')
        plt.xlim(orbits[list(orbits.keys())[0]]['Peaks'].InvField.min())
        plt.title(self.name+' orbit amplitudes')
        plt.show()
        
###################################################################################################
#                                                                                                 #
###################################################################################################

def sdh_load(year, num, plot):
    """ Loads delimited text frequency vs. field data
    File format:
        First column: Frequency
        Second column: Field
        Header: yes, one line
    Returns: DataFrame for both up and down sweeps
    Any function that returns a dict with a separate DataFrame 
        for up and down sweeps will work here.
    """
    fileu = 'Jun'+str(year)+'_2002s0'+str(num)+'u.txt'
    filed = 'Jun'+str(year)+'_2002s0'+str(num)+'d.txt'
    col_names = ['Freq', 'Field']
    df_sdhu = pd.read_csv(fileu, sep='\t')
    df_sdhd = pd.read_csv(filed, sep='\t')
    df_sdhu.columns = col_names
    df_sdhd.columns = col_names
    df_sdhu = pd.DataFrame({'Freq': df_sdhu.Freq, 
                            'Field': df_sdhu.Field, 'InvField':1/df_sdhu.Field[::-1]})
    df_sdhd = pd.DataFrame({'Freq': df_sdhd.Freq, 
                            'Field': df_sdhd.Field, 'InvField':1/df_sdhd.Field[::-1]})
    sdh_dict = {'Up': df_sdhu, 'Down': df_sdhd}
    if plot:
        plt.plot(df_sdhu.Field, 1e-6*df_sdhu.Freq, label='Up')
        plt.plot(df_sdhd.Field, 1e-6*df_sdhd.Freq, label='Down')
        plt.xlim(0)
        plt.xlabel('Field (T)')
        plt.ylabel('Frequency (MHz)')
        plt.legend(loc=0)
        plt.show()
    return df_sdhu, df_sdhd

###################################################################################################
#                                                                                                 #
###################################################################################################

def back_subtract(df_up, df_down, deg, Bmin, Bmax, npnts, plot, yscale, save):
    """ Performs subtraction of polynomial background on Freq vs. Inverse Field data.
    Inputs:
        df_up: DataFrame containing freq, field, inv field for up sweep
        df_down: DataFrame containint freq, field, inve field for down sweep
        deg: degree of polynomial fit (typically 5, 7, or 9)
        Bmin: minimum field for section of data to fit
        Bmax: maximum field for section of data to fit
        npnts: number of points to spline to (2**13-2**15 is reasonable)
        plot and yscale: If you want to plot output.
        save: Once you're happy with the background subtraction,
            add new data to dict
    Returns: DataFrames for spline fit, background subtracted up and down sweeps
    """
    Binv = np.linspace(1/Bmax, 1/Bmin, npnts)
    tck = interpolate.splrep(df_up.InvField[df_up.InvField>=1/Bmax][df_up.InvField<=1/Bmin].values[::-1],
                             df_up.Freq[df_up.InvField>=1/Bmax][df_up.InvField<=1/Bmin].values[::-1], s=0)
    new_up = interpolate.splev(Binv, tck, der=0)
    df_splup = pd.DataFrame({'Freq': new_up, 'Field': 1/Binv, 'InvField': Binv})
    
    tck = interpolate.splrep(df_down.InvField[df_down.InvField>=1/Bmax][df_down.InvField<=1/Bmin].values[::-1],
                             df_down.Freq[df_down.InvField>=1/Bmax][df_down.InvField<=1/Bmin].values[::-1], s=0)
    new_down = interpolate.splev(Binv, tck, der=0)
    df_spldown = pd.DataFrame({'Freq': new_down, 'Field': 1/Binv, 'InvField': Binv})
    
    coeffup = np.polyfit(df_up.InvField[df_up.InvField>=1/Bmax][df_up.InvField<=1/Bmin],
                       df_up.Freq[df_up.InvField>=1/Bmax][df_up.InvField<=1/Bmin], deg)
    coeffdown = np.polyfit(df_down.InvField[df_down.InvField>=1/Bmax][df_down.InvField<=1/Bmin],
                       df_down.Freq[df_down.InvField>=1/Bmax][df_down.InvField<=1/Bmin], deg)
    fitup = np.polyval(coeffup, Binv)
    fitdown = np.polyval(coeffdown, Binv)
    sub_up = new_up-fitup
    sub_down = new_down-fitdown
    if plot:
        plt.plot(Binv, 1e-3*sub_down, 'b', label='Down')
        plt.plot(Binv, 1e-3*sub_up, 'r', label='Up')
        plt.xlim(1/Bmax, 1/Bmin)
        plt.ylim(-yscale*1e-3,yscale*1e-3)
        plt.legend(loc=0)
        plt.xlabel(r'Inverse Field (T${}^{-1}$)')
        plt.ylabel(r'$\Delta$f (kHz)')
        plt.show()
    if not save:
        print('Happy with this background subtraction?')
        print('If so, set save=True and run again.')
        return None, None
    df_upcs = pd.DataFrame({'Field': 1/Binv, 'InvField': Binv,
                            'Freq': new_up, 'FreqSub': sub_up})
    df_downcs =  pd.DataFrame({'Field': 1/Binv, 'InvField': Binv,
                               'Freq': new_down, 'FreqSub': sub_down})
    return df_upcs, df_downcs
    
###################################################################################################
#                                                                                                 #
###################################################################################################

def fft_peaks(df_datacs, nskip, mph, mpd, threshold, edge, kpsh,
              valley, show, ax, nignore, xmax, keep_ind, save):
    """ Uses detect_peaks module to find peak frequencies using FFT of delta freq. vs inverse field.
    Inputs:
        df_datacs: DataFrame containing background subtracted data
        nskip: Number of points to skip at the beginning of the signal
        mph--ax: See documentation for deteck_peaks:
            http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
        nignore: Throw out peak if index < nignore
        xmax: x-axis maximum for plotting
        keep_ind: List of indices in the array of peak locations to keep
            e.g. if you want to keep 1st, 3rd, 4th, and 7th peaks:
                keep_ind = [0, 2, 3, 6]
            (inspect plot to decide which peak locations to keep)
        save: Once you've chosen the indices of the peak locations to keep, set save=True
            and run fft_peaks() again
    Returns: DataFrame containing amplitudes, frequencies of peaks if save=True
    """
    
    dt = df_datacs.InvField[1]-df_datacs.InvField[0]
    fftdata = abs(fft.rfft(df_datacs.FreqSub[nskip:]))
    f = fft.rfftfreq(len(df_datacs.FreqSub)-nskip, d=dt)
    peak_ind = detect_peaks(fftdata, mph, mpd, threshold, edge, kpsh, valley, show, ax, nignore, xmax, keep_ind)
    df_fftpeaks = pd.DataFrame({'Freq': [f[i] for i in peak_ind], 'Amp': [fftdata[i] for i in peak_ind],
                             'Ind': [i for i in peak_ind]})
    if not save:
        print('Happy with these peaks?')
        print('If so, set save=True and run again.\n')
        print(df_fftpeaks)
        return None
    print('The following {} peaks have been added to the dataset:\n'.format(len(peak_ind)))
    print(df_fftpeaks)
    return df_fftpeaks
    
###################################################################################################
#                                                                                                 #
###################################################################################################

def bandpass(lowcut, highcut, fs, order=2):
    nyq = fs/2
    low = lowcut/nyq
    high = highcut/nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(freq_data, lowcut, highcut, fs, order, method):
    b, a = bandpass(lowcut, highcut, fs, order=order)
    freq_filt = signal.filtfilt(b, a, freq_data, method=method)
    return freq_filt

def filter_orbit(df_datacs, center_freq, passband, orbit, nskip, order, method, plot, save):
    """ Bandpass filter to isolate a specific orbit/fundamental frequency.
    Inputs:
        df_datacs: DataFrame containing background subtracted data
        center_freq: Frequency (in tesla) of the peak you want to isolate
        passband: Filter will allow center_freq +/- passband to pass (tesla)
        orbit: (string) name of orbit/fundamental frequency, used as dict key
        nskip: number of points to skip at beginning of signal
        order: order of the filter
        plot: (Boolean) plotting option
        save: If you're satisfied with the filtered signal, set save=True
    Returns: DataFrame with orbit properties if save=True:
        ({'Freq': filtered signal, 'InvField': inverse field})
    """
    freq_data = df_datacs.FreqSub.values[nskip:]
    inv_field = df_datacs.InvField.values[nskip:]
    fs = 1/(inv_field[1]-inv_field[0])
    lowcut, highcut = center_freq-passband, center_freq+passband
    freq_filt = bandpass_filter(freq_data, lowcut, highcut, fs, order=order, method=method)
    if plot:
        plt.plot(df_datacs.InvField[nskip:], 1e-3*df_datacs.FreqSub[nskip:], label='Raw')
        plt.plot(df_datacs.InvField[nskip:], 1e-3*freq_filt, 'r', label='{} T bandpass'.format(center_freq))
        plt.xlabel(r'Inverse Field (T${}^{-1}$)')
        plt.ylabel(r'$\Delta$f (kHz)')
        plt.legend(loc=0)
        plt.title(orbit+' orbit')
        plt.show()
    if not save:
        print('Happy with the filtered signal?')
        print('If so, set save=True and run again.')
        return None
    print('The filtered '+orbit+' orbit has been added to the dataset.')
    return pd.DataFrame({'Freq': freq_filt, 'InvField': df_datacs.InvField.values[nskip:]})

###################################################################################################
#                                                                                                 #
###################################################################################################

def peak_amplitudes(orbit_dict, orbit, nskip, show, save):
    """ Finds peak amplitude as a function of inverse field.
    Inputs:
        orbit_dict: dict containing each separate orbit
        orbit: (string) name of orbit you're looking at (a key in orbit_dict)
        nskip: number of points to skip at beginning of signal 
        show: (Boolean) plotting option for detect_peaks()
        save: if detect_peaks() was successful, set save=True and run again
    Returns: DataFrame containing peak amplitude vs. inverse field if save=True
    """
    peaks = abs(orbit_dict['Osc'].Freq)
    peak_ind = detect_peaks(peaks, show=show)
    peak_fields = np.array([orbit_dict['Osc'].InvField[i] for i in peak_ind])
    peak_amps = np.array([peaks[i] for i in peak_ind])
    if not save:
        print('Happy with the peaks detected?')
        print('If so, set save=True and run again.\n')
        return None
    print('The below peaks have been added to the dataset under orbit '+orbit+'.\n')
    df_peaks = pd.DataFrame({'Amp': peak_amps, 'InvField': peak_fields})
    plt.plot(peak_fields, 1e-3*peak_amps, 'o')
    plt.xlabel(r'Inverse Field (T${}^{-1}$)')
    plt.ylabel(r'Amplitude (kHz)')
    plt.title(orbit+' orbit amplitudes')
    plt.show()
    return df_peaks

###################################################################################################
#                                                                                                 #
###################################################################################################
