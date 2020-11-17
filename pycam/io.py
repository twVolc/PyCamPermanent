# -*- coding: utf-8 -*-

"""
Contains some simple functions for saving data
"""

from pyplis import LineOnImage
from pyplis.fluxcalc import EmissionRates
from .setupclasses import SpecSpecs
from .utils import check_filename
import numpy as np
import cv2
import os


def save_img(img, filename, ext='.png'):
    """Saves image
    img: np.array
        Image array to be saved
    filename: str
        File path for saving
    ext: str
        File extension for saving, including "."
    """
    lock = filename.replace(ext, '.lock')
    open(lock, 'a').close()

    # Save image
    cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Remove lock to free image for transfer
    os.remove(lock)


def save_spectrum(wavelengths, spectrum, filename):
    """Saves spectrum as numpy .mat file
    wavelengths: NumPy array-like object
        Wavelength values held in array
    spectrum: NumPy array-like object
        Spectrum digital numbers held in array
    filename: str
        File path for saving
    """
    # Create lock file to secure file until saving is complete
    lock = filename.replace(SpecSpecs().file_ext, '.lock')
    open(lock, 'a').close()

    # Pack wavelengths and spectrum into single array
    spec_array = np.array([wavelengths, spectrum])

    # Save spectrum
    np.save(filename, spec_array)

    # Remove lock
    os.remove(lock)


def load_spectrum(filename):
    """Essentially a wrapper to numpy load function, with added filename check
    :param  filename:   str     Full path of spectrum to be loaded"""
    try:
        check_filename(filename, SpecSpecs().file_ext.split('.')[-1])
    except:
        raise
    spec_array = np.load(filename)
    wavelengths = spec_array[0, :]
    spectrum = spec_array[1, :]
    return wavelengths, spectrum


def spec_txt_2_npy(directory):
    """Generates numpy arrays of spectra text files (essentially compressing them)"""

    # List all text files
    txt_files = [f for f in os.listdir(directory) if '.txt' in f]

    for file in txt_files:
        spec = np.loadtxt(directory + file)
        wavelengths = spec[:, 0]
        spectrum = spec[:, 1]

        save_spectrum(wavelengths, spectrum, directory + file.replace('txt', 'npy'))


def save_pcs_line(line, filename):
    """
    Saves PCS line coordinates so that it can be reloaded
    :param line:        LineOnImage
    :param filename:    str
    :return:
    """
    with open(filename, 'w') as f:
        f.write('x={},{}\n'.format(int(np.round(line.x0)), int(np.round(line.x1))))
        f.write('y={},{}\n'.format(int(np.round(line.y0)), int(np.round(line.y1))))
        f.write('orientation={}\n'.format(line.normal_orientation))


def load_pcs_line(filename, color='blue', line_id='line'):
    """
    Loads PCS line and returns it as a pyplis object
    :param filename:
    :return:
    """
    if not os.path.exists(filename):
        print('Cannot get line from filename as no file exists at this path')
        return

    with open(filename, 'r') as f:
        for line in f:
            if 'x=' in line:
                coords = line.split('x=')[-1].split('\n')[0]
                x0, x1 = [int(x) for x in coords.split(',')]
            elif 'y=' in line:
                coords = line.split('y=')[-1].split('\n')[0]
                y0, y1 = [int(y) for y in coords.split(',')]
            elif 'orientation=' in line:
                orientation = line.split('orientation=')[-1].split('\n')[0]

    pcs_line = LineOnImage(x0=x0, y0=y0, x1=x1, y1=y1,
                           normal_orientation=orientation,
                           color=color,
                           line_id=line_id)

    return pcs_line


def save_emission_rates_as_txt(path, emission_dict, save_all=False):
    """
    Saves emission rates as text files every hour - emission rates are split into hour-long
    :param path:            str     Directory to save to
    :param emission_dict:   dict    Dictionary of emission rates for different lines and different flow modes
                                    Assumed to be time-sorted
    :param save_all:        bool    If True, the entire time series is saved, even if the hour isn't complete
    :return:
    """
    file_fmt = "pyplis_EmissionRates_{}_{}_{}.txt"
    date_fmt = "%Y%m%d"
    time_fmt = "%H%M"
    emis_attrs = ['_start_acq', '_phi', '_phi_err', '_velo_eff', '_velo_eff_err']

    # Try to make directory if it is not valid
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except BaseException as e:
            print('Could not save emission rate data as path definition is not valid:\n'
                  '{}'.format(e))

    # Loop through lines (includes 'total' and save data to it
    for line_id in emission_dict:
        # Make dir for specific line if it doesn't already exist
        line_path = os.path.join(path, 'line_{}'.format(line_id))
        if not os.path.exists(line_path):
            os.mkdir(line_path)

        for flow_mode in emission_dict[line_id]:
            emis_dict = emission_dict[line_id][flow_mode]
            # Check there is data in this dictionary - if not, we don't save this data
            if len(emis_dict._start_acq) == 0:
                continue

            # Make line directory
            full_path = os.path.join(line_path, flow_mode)
            if not os.path.exists(full_path):
                os.mkdir(full_path)

            start_time = emis_dict._start_acq[0]
            start_time_hr = start_time.hour
            end_time_hr = emis_dict._start_acq[-1].hour
            if not save_all:
                end_time_hr -= 1           # We don't want the most recent hour as this may contain incomplete data
                if end_time_hr < start_time_hr:
                    # In this case there is no data to be saved, so we move to next dataset
                    continue

            # Have to make a new EmissionRates object to save data
            for hour in range(start_time_hr, end_time_hr + 1):
                # Arrange times of file
                file_date = start_time.strftime(date_fmt)
                file_start_time = start_time.replace(hour=hour, minute=0, second=0)
                file_end_time = start_time.replace(hour=hour, minute=59, second=0)
                start_time_str = file_start_time.strftime(time_fmt)
                end_time_str = file_end_time.strftime(time_fmt)

                # Generate filename
                filename = file_fmt.format(file_date, start_time_str, end_time_str)
                pathname = os.path.join(full_path, filename)

                # We don't overwrite data, so if the file already exists we continue without saving
                if os.path.exists(pathname):
                    continue
                else:
                    # Make new emission rates object to save
                    emis_rates = EmissionRates(line_id, velo_mode=flow_mode)
                    indices = tuple([(file_start_time < np.array(emis_dict._start_acq)) &
                                     (np.array(emis_dict._start_acq) <= file_end_time)])
                    # Loop through attributes in emission rate object and set them to new object
                    # This loop is just cleaner than writing out each attribute...
                    for attr in emis_attrs:
                        setattr(emis_rates, attr, np.array(getattr(emis_dict, attr))[indices])

                    # Save object
                    emis_rates.to_pandas_dataframe().to_csv(pathname)

