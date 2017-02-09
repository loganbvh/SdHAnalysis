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

def sdh_load(year, num, plot=True):
    """ Loads delimited text frequency vs. field data
    File format:
        First column: Frequency
        Second column: Field
        Header: yes, one line
    Returns: dict containing a DataFrame for both up and down sweeps
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
    return sdh_dict

###################################################################################################
#                                                                                                 #
###################################################################################################

def back_subtract(sdh_dict, deg, Bmin, Bmax, npnts=2**13, plot=True, yscale=1e5, save=False):
    """ Performs subtraction of polynomial background on Freq vs. Inverse Field data.
    Inputs:
        sdh_dict: dict output from data loading function
        deg: degree of polynomial fit (typically 5, 7, or 9)
        Bmin: minimum field for section of data to fit
        Bmax: maximum field for section of data to fit
        npnts: number of points to spline to (2**13-2**15 is reasonable)
        plot and yscale: If you want to plot output.
        save: Once you're happy with the background subtraction,
            add new data to dict
    Returns: updated dict if save is True
    """
    df_up = sdh_dict['Up']
    df_down = sdh_dict['Down']
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
    if save:
        sdh_dict.update({'Upcs': pd.DataFrame({'Field': 1/Binv, 'InvField': Binv,
                                     'Freq': new_up, 'FreqSub': sub_up}),
                    'Downcs': pd.DataFrame({'Field': 1/Binv, 'InvField': Binv,
                                        'Freq': new_down, 'FreqSub': sub_down})})
        print('Dict updated.\nKeys:',sdh_dict.keys())
    return sdh_dict
    
###################################################################################################
#                                                                                                 #
###################################################################################################

def get_fft_peaks(sdh_dict, direc='Down', nskip=500, mph=None, mpd=1,
                  threshold=0, edge='rising', kpsh=False, valley=False,
                  show=True, ax=None, nignore=5, xmax=None, keep_ind=None, save=False):
    """ Uses detect_peaks module to find peak frequencies using FFT of delta freq. vs inverse field.
    Inputs:
        sdh_dict: dict output from back_subtract()
        direc: Up or down sweep (default: 'Down')
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
            and run get_fft_peaks() again
    Returns: Updated sdh_dict if save=True
    """
    direc = direc+'cs'
    dt = sdh_dict[direc].InvField[1]-sdh_dict[direc].InvField[0]
    fftdata = abs(fft.rfft(sdh_dict[direc].FreqSub[nskip:]))
    f = fft.rfftfreq(len(sdh_dict[direc].FreqSub)-nskip, d=dt)
    peak_ind = detect_peaks(fftdata, mph, mpd, threshold, edge, kpsh,valley, show, ax, nignore, xmax, keep_ind)
    
    df_peaks = pd.DataFrame({'Freq': [f[i] for i in peak_ind], 'Amp': [fftdata[i] for i in peak_ind],
                             'Ind': [i for i in peak_ind]})
    if save:
        sdh_dict.update({'FFTPeaks': df_peaks, 'nskip': nskip})
        print('The following {} peaks have been added to the dict:'.format(len(df_peaks)))
        print(df_peaks,'\n')
        print('Keys:', sdh_dict.keys())
    else:
        print(df_peaks)
    return sdh_dict
    
###################################################################################################
#                                                                                                 #
###################################################################################################

def bandpass(lowcut, highcut, fs, order=2):
    nyq = fs/2
    low = lowcut/nyq
    high = highcut/nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(freq_data, lowcut, highcut, fs, order=2):
    b, a = bandpass(lowcut, highcut, fs, order=order)
    freq_filt = signal.filtfilt(b, a, freq_data)
    return freq_filt

def isolate_orbit(sdh_dict, center_freq, passband, orbit, order=2, direc='Down', plot=True, save=False):
    """ Bandpass filter to isolate a specific orbit/fundamental frequency.
    Inputs:
        sdh_dict: dict output from get_fft_peaks()
        center_freq: Frequency (in tesla) of the peak you want to isolate
        passband: Filter will allow center_freq +/- passband to pass (tesla)
        orbit: (string) name of orbit/fundamental frequency, used as dict key
        direc: Up or down sweep (default: 'Down')
        save: If you're satisfied with the filtered signal, set save=True
    Returns: Updated sdh_dict if save=True
        Note: sdh_dict[orbit] is a dict itself.
    """
    nskip = sdh_dict['nskip']
    direc = direc+'cs'
    freq_data = sdh_dict[direc].FreqSub.values[nskip:]
    inv_field = sdh_dict[direc].InvField.values[nskip:]
    fs = 1/(inv_field[1]-inv_field[0])
    lowcut, highcut = center_freq-passband, center_freq+passband
    freq_filt = bandpass_filter(freq_data, lowcut, highcut, fs, order=order)
    if plot:
        plt.plot(sdh_dict[direc].InvField[nskip:], 1e-3*sdh_dict[direc].FreqSub[nskip:], label='Raw')
        plt.plot(sdh_dict[direc].InvField[nskip:], 1e-3*freq_filt, 'r', label='{} T bandpass'.format(center_freq))
        plt.xlabel(r'Inverse Field (T${}^{-1}$)')
        plt.ylabel(r'$\Delta$f (kHz)')
        plt.legend(loc=0)
        plt.title(orbit+' orbit')
        plt.show()
    if save:
        sdh_dict.update({orbit: {'Freq': freq_filt}})
        print('Filtered '+orbit+' orbit has been added to the dict.')
        print('Keys: ', sdh_dict.keys())
    return sdh_dict

###################################################################################################
#                                                                                                 #
###################################################################################################

def get_peak_amplitudes(sdh_dict, orbit, direc='Down', show=True, save=False):
    """ Finds peak amplitude as a function of inverse field.
    Inputs:
        sdh_dict: dict output from isolate_orbit()
        orbit: (string) name of orbit used in isolate_orbit()
        direc: Up or down sweep (default: 'Down')
        show: Plotting option for detect_peaks()
        save: If detect_peaks() was successful, set save=True and run again
    Returns: sdh_dict with peak amplitudes and locations added to dict sdh_dict[orbit]
    """
    nskip = sdh_dict['nskip']
    direc = direc+'cs'
    peaks = abs(sdh_dict[orbit]['Freq'])
    peak_ind = detect_peaks(peaks, show=show)
    peak_fields = np.array([sdh_dict[direc].InvField[nskip+i] for i in peak_ind])
    peak_amps = np.array([peaks[i] for i in peak_ind])
    if save:
        print('Peak amplitudes of '+orbit+' orbit have been added to the dict.')
        print('Keys: ', sdh_dict.keys())
        df_peaks = pd.DataFrame({'Amp': peak_amps, 'InvField': peak_fields})
        sdh_dict[orbit].update({'Peaks': df_peaks})
        plt.plot(peak_fields, 1e-3*peak_amps, 'o')
        plt.xlabel(r'Inverse Field (T${}^{-1}$)')
        plt.ylabel(r'Amplitude (kHz)')
        plt.title(orbit+' orbit amplitudes')
        plt.show()
    return sdh_dict

###################################################################################################
#                                                                                                 #
###################################################################################################

