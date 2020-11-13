# -*- coding: utf-8 -*-

"""
Contains some simple functions for saving data
"""

from pyplis import LineOnImage
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
