# -*- coding: utf-8 -*-

"""
Contains some simple functions for saving data
"""

from pyplis import LineOnImage
from pyplis.fluxcalc import EmissionRates
from .setupclasses import SpecSpecs, CameraSpecs
from .utils import check_filename
import numpy as np
import scipy.io
import cv2
import os
import datetime


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


def save_light_dil_line(line, filename):
    """Saves light dilution line to text file - same function as draw_pcs_line, so just a wrapper for this"""
    save_pcs_line(line, filename)


def load_light_dil_line(filename, color='blue', line_id='line'):
    """Loads light dilution line from text file"""
    line = load_pcs_line(filename, color, line_id)
    return line


def save_fov_txt(filename, fov_obj):
    """
    Saves fov data to text file
    
    """
    pass

def load_fov_txt(filename):
    """
    Loads fov data from a txt file
    :param filename:
    :return:
    """
    pass


def save_so2_img_raw(path, img, filename=None, img_end='cal', ext='.mat'):
    """
    Saves tau or calibrated image. Saves the raw_data
    :param path:        str     Directory path to save image to
    :param img:         Img     pyplis.Img object to be saved
    :param filename:    str     Filename to be saved. If None, fielname is determined from meta data of Img
    :param img_end:     str     End of filename - describes the type of file
    :param ext:         str     File extension (takes .mat, .npy, .fts)
    """
    # Define accepted save types
    save_funcs = {'.mat': scipy.io.savemat,
                  '.npy': np.save,
                  '.fts': None}

    if filename is not None:
        ext = '.' + filename.split('.')[-1]

    # Check we have a valid filename
    if ext not in save_funcs:
        print('Unrecognised file extension for saving SO2 image. Image will not be saved')
        return

    if filename is None:
        # Put time into a string
        time_str = img.meta['start_acq'].strftime(CameraSpecs().file_datestr)

        filename = '{}_{}{}'.format(time_str, img_end, ext)

    if ext == '.fts':
        img.save_as_fits(path, filename)    # Uee pyplis built-in function for saving
    else:
        full_path = os.path.join(path, filename)

        if os.path.exists(full_path):
            print('Overwriting file to save image: {}'.format(full_path))

        # If we are saving as a matlab file we need to make a dictionary to save for the scipy.io.savemat argument
        if ext == '.mat':
            save_obj = {'img': img.img}
        else:
            save_obj = img.img

        # SAVE IMAGE
        save_funcs[ext](full_path, save_obj)


def save_so2_img(path, img, filename=None, compression=0, max_val=None):
    """
    Scales image and saves as am 8-bit PNG image - for easy viewing. No data integrity is saved with this function
    :param path:    str             Path to directory to save image
    :param img:     pyplis.Img
    :param compression:     int     Compression of PNG (0-9)
    :param max_val:  float/int      Maximum value of image to normalise to
    """
    if filename is None:
        # Put time into a string
        time_str = img.meta['start_acq'].strftime(CameraSpecs().file_datestr)

        filename = '{}_img.png'.format(time_str)
    full_path = os.path.join(path, filename)
    if os.path.exists(full_path):
        print('Overwriting file to save image: {}'.format(full_path))

    # Scale image and convert to 8-bit
    if max_val is None:
        max_val = np.nanmax(img.img)
    arr = img.img
    arr[arr > max_val] = max_val
    arr[arr < 0] = 0
    im2save = np.array((arr / max_val) * 255, dtype=np.uint8)

    png_compression = [cv2.IMWRITE_PNG_COMPRESSION, compression]  # Set compression value

    # Save image
    cv2.imwrite(full_path, im2save, png_compression)


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


def write_witty_schedule_file(filename, time_on, time_off):
    """
    Writes a file for controlling the Witty Pi on/off scheduling
    :param filename:    str         Full path to file for writing
    :param time_on:     datetime    Time to turn pi on each day
    :param time_ff:     datetime    Time to turn pi off each day
    """
    time_fmt = '%H:%M:%S'
    time_on_str = time_on.strftime(time_fmt)

    if time_off - time_on > datetime.timedelta(0):
        time_delt_on = time_off - time_on
        num_hours_on, rem = divmod(time_delt_on.total_seconds(), 60*60)
        num_mins_on = rem / 60

        time_delt_off = datetime.timedelta(hours=24) - time_delt_on
        num_hours_off, rem = divmod(time_delt_off.total_seconds(), 60 * 60)
        num_mins_off = rem / 60

    elif time_off - time_on < datetime.timedelta(0):
        time_delt_off = time_on - time_off
        num_hours_off, rem = divmod(time_delt_off.total_seconds(), 60 * 60)
        num_mins_off = rem / 60

        time_delt_on = datetime.timedelta(hours=24) - time_delt_off
        num_hours_on, rem = divmod(time_delt_on.total_seconds(), 60 * 60)
        num_mins_on = rem / 60

    else:
        # TODO time_off and time on are the same - we don't ever turn the pi off. Work out how to cancel script use
        # TODO on witty pi
        pass

    with open(filename, 'w', newline='\n') as f:
        f.write('# Raspberry Pi start-up/shut-down schedule script\n')

        # Add lines for quicker/easier access when reading file
        f.write('# on_time={}\n'.format(time_on.strftime('%H:%M')))
        f.write('# off_time={}\n'.format(time_off.strftime('%H:%M')))

        f.write('BEGIN 2020-01-01 {}\n'.format(time_on_str))
        f.write('END 2038-01-01 12:00:00\n')
        f.write('ON H{:.0f} M{:.0f}\n'.format(num_hours_on, num_mins_on))
        f.write('OFF H{:.0f} M{:.0f}\n'.format(num_hours_off, num_mins_off))


def read_witty_schedule_file(filename):
    """Read witty schedule file"""
    with open(filename, 'r', newline='\n') as f:
        for line in f:
            if 'on_time=' in line:
                on_time = line.split('=')[1].split('\n')[0]
                on_hour, on_min = [int(x) for x in on_time.split(':')]
            elif 'off_time=' in line:
                off_time = line.split('=')[1].split('\n')[0]
                off_hour, off_min = [int(x) for x in off_time.split(':')]
    try:
        return (on_hour, on_min), (off_hour, off_min)
    except AttributeError:
        print('File not in expected format to retrieve start-up/shut-down information for instrument')


def write_script_crontab(filename, cmd, time_on):
    """
    Writes crontab script to filename
    :param  filename:   str     File to write to
    :param  time_on:    list    List of times to start script
    :param  cmd:        list    List of commands relating to times
    """
    if len(cmd) != len(time_on):
        print('Lengths of lists of crontab commands and times must be equal')
        return

    with open(filename, 'w', newline='\n') as f:
        f.write('# Crontab schedule file written by pycam\n')

        # Setup path for shell
        f.write('PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin\n')

        # Loops through commands and add them to crontab
        for i in range(len(cmd)):
            # Organise time object
            time_obj = time_on[i]
            if isinstance(time_obj, datetime.datetime):
                time_str = '{} * * * '.format(time_obj.strftime('%M %H'))
            # If time obj isn't datetime object we assume it is in the correct timing format for crontab
            else:
                time_str = time_obj

            command = cmd[i]
            line = time_str + command

            f.write('{}\n'.format(line))


def read_script_crontab(filename, cmds):
    """Reads file containing start/stop pycam script times"""
    times = {}

    with open(filename, 'r') as f:
        for line in f:
            # Loop through commands to check if any are in the current file line
            for cmd in cmds:
                if cmd in line:
                    minute, hour = line.split()[0:2]
                    # If hour is * then we are running defined by minutes only
                    if hour == '*':
                        hour = 0
                        # We then need to catch this case, where 0 means hourly, so we set minute to 60
                        if minute == '0':
                            minute = 60
                    else:
                        # We now need to catch other cases where running defined by minutes only '*/{}' fmt
                        minute = minute.split('/')[-1]
                    # If the line is commented out, we set everything to 0 (e.g. used for temperature logging)
                    if line[0] == '#':
                        minute = 0
                        hour = 0

                    times[cmd] = (int(hour), int(minute))
    return times



