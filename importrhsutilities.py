import sys, struct, math, os, time
import numpy as np
import matplotlib.pyplot as plt

# Define plural function
def plural(n):
    """Utility function to optionally pluralize words based on the value of n.
    """

    if n == 1:
        return ''
    else:
        return 's'
        
# Define read_qstring function
def read_qstring(fid):
    """Read Qt style QString.  

    The first 32-bit unsigned number indicates the length of the string (in bytes).  
    If this number equals 0xFFFFFFFF, the string is null.

    Strings are stored as unicode.
    """

    length, = struct.unpack('<I', fid.read(4))
    if length == int('ffffffff', 16): return ""

    if length > (os.fstat(fid.fileno()).st_size - fid.tell() + 1) :
        print(length)
        raise Exception('Length too long.')

    # convert length from bytes to 16-bit Unicode words
    length = int(length / 2)

    data = []
    for i in range(0, length):
        c, = struct.unpack('<H', fid.read(2))
        data.append(c)

    if sys.version_info >= (3,0):
        a = ''.join([chr(c) for c in data])
    else:
        a = ''.join([unichr(c) for c in data])
    
    return a
    
# Define read_header function
def read_header(fid):
    """Reads the Intan File Format header from the given file."""

    # Check 'magic number' at beginning of file to make sure this is an Intan
    # Technologies RHD2000 data file.
    magic_number, = struct.unpack('<I', fid.read(4)) 
    if magic_number != int('0xD69127AC', 16): raise Exception('Unrecognized file type.')

    header = {}
    # Read version number.
    version = {}
    (version['major'], version['minor']) = struct.unpack('<hh', fid.read(4)) 
    header['version'] = version

    print('')
    print('Reading Intan Technologies RHS2000 Data File, Version {}.{}'.format(version['major'], version['minor']))
    print('')

    # Read information of sampling rate and amplifier frequency settings.
    header['sample_rate'], = struct.unpack('<f', fid.read(4))
    (header['dsp_enabled'],
     header['actual_dsp_cutoff_frequency'],
     header['actual_lower_bandwidth'],
     header['actual_lower_settle_bandwidth'],
     header['actual_upper_bandwidth'],
     header['desired_dsp_cutoff_frequency'],
     header['desired_lower_bandwidth'],
     header['desired_lower_settle_bandwidth'],
     header['desired_upper_bandwidth']) = struct.unpack('<hffffffff', fid.read(34))


    # This tells us if a software 50/60 Hz notch filter was enabled during
    # the data acquisition.
    notch_filter_mode, = struct.unpack('<h', fid.read(2))
    header['notch_filter_frequency'] = 0
    if notch_filter_mode == 1:
        header['notch_filter_frequency'] = 50
    elif notch_filter_mode == 2:
        header['notch_filter_frequency'] = 60
    
    (header['desired_impedance_test_frequency'], header['actual_impedance_test_frequency']) = struct.unpack('<ff', fid.read(8))
    (header['amp_settle_mode'], header['charge_recovery_mode']) = struct.unpack('<hh', fid.read(4))
    
    frequency_parameters = {}
    frequency_parameters['amplifier_sample_rate'] = header['sample_rate']
    frequency_parameters['board_adc_sample_rate'] = header['sample_rate']
    frequency_parameters['board_dig_in_sample_rate'] = header['sample_rate']
    frequency_parameters['desired_dsp_cutoff_frequency'] = header['desired_dsp_cutoff_frequency']
    frequency_parameters['actual_dsp_cutoff_frequency'] = header['actual_dsp_cutoff_frequency']
    frequency_parameters['dsp_enabled'] = header['dsp_enabled']
    frequency_parameters['desired_lower_bandwidth'] = header['desired_lower_bandwidth']
    frequency_parameters['desired_lower_settle_bandwidth'] = header['desired_lower_settle_bandwidth']
    frequency_parameters['actual_lower_bandwidth'] = header['actual_lower_bandwidth']
    frequency_parameters['actual_lower_settle_bandwidth'] = header['actual_lower_settle_bandwidth']
    frequency_parameters['desired_upper_bandwidth'] = header['desired_upper_bandwidth']
    frequency_parameters['actual_upper_bandwidth'] = header['actual_upper_bandwidth']
    frequency_parameters['notch_filter_frequency'] = header['notch_filter_frequency']
    frequency_parameters['desired_impedance_test_frequency'] = header['desired_impedance_test_frequency']
    frequency_parameters['actual_impedance_test_frequency'] = header['actual_impedance_test_frequency']
    
    header['frequency_parameters'] = frequency_parameters
    
    (header['stim_step_size'],
     header['recovery_current_limit'],
     header['recovery_target_voltage']) = struct.unpack('fff', fid.read(12))

    note1 = read_qstring(fid)
    note2 = read_qstring(fid)
    note3 = read_qstring(fid)
    header['notes'] = { 'note1' : note1, 'note2' : note2, 'note3' : note3}
    
    (header['dc_amplifier_data_saved'],
     header['eval_board_mode']) = struct.unpack('<hh', fid.read(4))
    
    header['ref_channel_name'] = read_qstring(fid)

    
    # Create structure arrays for each type of data channel.
    header['spike_triggers'] = []
    header['amplifier_channels'] = []
    header['board_adc_channels'] = []
    header['board_dac_channels'] = []
    header['board_dig_in_channels'] = []
    header['board_dig_out_channels'] = []

    # Read signal summary from data file header.
    number_of_signal_groups, = struct.unpack('<h', fid.read(2))

    for signal_group in range(1, number_of_signal_groups + 1):
        signal_group_name = read_qstring(fid)
        signal_group_prefix = read_qstring(fid)
        (signal_group_enabled, signal_group_num_channels, signal_group_num_amp_channels) = struct.unpack('<hhh', fid.read(6))

        if (signal_group_num_channels > 0) and (signal_group_enabled > 0):
            for signal_channel in range(0, signal_group_num_channels):
                new_channel = {'port_name' : signal_group_name, 'port_prefix' : signal_group_prefix, 'port_number' : signal_group}
                new_channel['native_channel_name'] = read_qstring(fid)
                new_channel['custom_channel_name'] = read_qstring(fid)
                (new_channel['native_order'], new_channel['custom_order'],
                 signal_type, channel_enabled, new_channel['chip_channel'],
                 command_stream, new_channel['board_stream']) = struct.unpack('<hhhhhhh', fid.read(14)) # ignore command_stream
                new_trigger_channel = {}
                (new_trigger_channel['voltage_trigger_mode'],
                 new_trigger_channel['voltage_threshold'],
                 new_trigger_channel['digital_trigger_channel'],
                 new_trigger_channel['digital_edge_polarity']) = struct.unpack('<hhhh', fid.read(8))
                (new_channel['electrode_impedance_magnitude'],
                 new_channel['electrode_impedance_phase']) = struct.unpack('<ff', fid.read(8))

                if channel_enabled:
                    if signal_type == 0:
                        header['amplifier_channels'].append(new_channel)
                        header['spike_triggers'].append(new_trigger_channel)
                    elif signal_type == 1:
                        raise Exception('Wrong signal type for the rhs format')
                        #header['aux_input_channels'].append(new_channel)
                    elif signal_type == 2:
                        raise Exception('Wrong signal type for the rhs format')
                        #header['supply_voltage_channels'].append(new_channel)
                    elif signal_type == 3:
                        header['board_adc_channels'].append(new_channel)
                    elif signal_type == 4:
                        header['board_dac_channels'].append(new_channel)
                    elif signal_type == 5:
                        header['board_dig_in_channels'].append(new_channel)
                    elif signal_type == 6:
                        header['board_dig_out_channels'].append(new_channel)
                    else:
                        raise Exception('Unknown channel type.')

    # Summarize contents of data file.
    header['num_amplifier_channels'] = len(header['amplifier_channels'])
    header['num_board_adc_channels'] = len(header['board_adc_channels'])
    header['num_board_dac_channels'] = len(header['board_dac_channels'])
    header['num_board_dig_in_channels'] = len(header['board_dig_in_channels'])
    header['num_board_dig_out_channels'] = len(header['board_dig_out_channels'])

    return header

# Define find_channel_in_group function
def find_channel_in_group(channel_name, signal_group):
    for count, this_channel in enumerate(signal_group):
        if this_channel['custom_channel_name'] == channel_name:
            return True, count
    return False, 0

# Define find_channel_in_header function
def find_channel_in_header(channel_name, header):

    # Look through all present signal groups
    
    # 1. Look through amplifier_channels
    if 'amplifier_channels' in header:
        channel_found, channel_index = find_channel_in_group(channel_name, header['amplifier_channels'])
        if channel_found:
            return True, 'amplifier_channels', channel_index
    
    # 2. Look through board_adc_channels
    if 'board_adc_channels' in header:
        channel_found, channel_index = find_channel_in_group(channel_name, header['board_adc_channels'])
        if channel_found:
            return True, 'board_adc_channels', channel_index
        
    # 3. Look through board_dac_channels
    if 'board_dac_channels' in header:
        channel_found, channel_index = find_channel_in_group(channel_name, header['board_dac_channels'])
        if channel_found:
            return True, 'board_dac_channels', channel_index
    
    # 4. Look through board_dig_in_channels
    if 'board_dig_in_channels' in header:
        channel_found, channel_index = find_channel_in_group(channel_name, header['board_dig_in_channels'])
        if channel_found:
            return True, 'board_dig_in_channels', channel_index
    
    # 5. Look through board_dig_out_channels
    if 'board_dig_out_channels' in header:
        channel_found, channel_index = find_channel_in_group(channel_name, header['board_dig_out_channels'])
        if channel_found:
            return True, 'board_dig_out_channels', channel_index
    
    return False, '', 0
    
    
# Define get_bytes_per_data_block function
def get_bytes_per_data_block(header):
    """Calculates the number of bytes in each 128-sample datablock."""
    N = 128 # n of amplifier samples
    # Each data block contains N amplifier samples.
    bytes_per_block = N * 4  # timestamp data
    
    bytes_per_block += N * 2 * header['num_amplifier_channels']

    # DC amplifier voltage (absent if flag was off)
    # bytes_per_block += N * 2 * header['dc_amplifier_data_saved']
    if header['dc_amplifier_data_saved'] > 0:
        bytes_per_block += N * 2 * header['num_amplifier_channels']

    # Stimulation data, one per enabled amplifier channels
    bytes_per_block += N * 2 * header['num_amplifier_channels']

    # Board analog inputs are sampled at same rate as amplifiers
    bytes_per_block += N * 2 * header['num_board_adc_channels']

    # Board analog outputs are sampled at same rate as amplifiers
    bytes_per_block += N * 2 * header['num_board_dac_channels']

    # Board digital inputs are sampled at same rate as amplifiers
    if header['num_board_dig_in_channels'] > 0:
        bytes_per_block += N * 2

    # Board digital outputs are sampled at same rate as amplifiers
    if header['num_board_dig_out_channels'] > 0:
        bytes_per_block += N * 2

    return bytes_per_block
    
# Define read_one_data_block function
def read_one_data_block(data, header, indices, fid):
    """Reads one 128-sample data block from fid into data, at the location indicated by indices."""

    data['t'][indices['amplifier']:
    (indices['amplifier']+128)] = np.array(struct.unpack('<' + 'i' * 128, fid.read(128*4)))

    if header['num_amplifier_channels'] > 0:
        tmp = np.fromfile(fid, dtype='uint16', count=128 * header['num_amplifier_channels'])
        data['amplifier_data'][range(header['num_amplifier_channels']),
        indices['amplifier']:(indices['amplifier']+128)] = tmp.reshape(header['num_amplifier_channels'], 128)

        # check if dc amplifier voltage was saved
        if header['dc_amplifier_data_saved']:
            tmp = np.fromfile(fid, dtype='uint16', count=128 * header['num_amplifier_channels'])
            data['dc_amplifier_data'][range(header['num_amplifier_channels']),
            indices['amplifier']:(indices['amplifier'] + 128)] = tmp.reshape(header['num_amplifier_channels'], 128)

        # get the stimulation data
        tmp = np.fromfile(fid, dtype='uint16', count=128 * header['num_amplifier_channels'])
        data['stim_data_raw'][range(header['num_amplifier_channels']),
        indices['amplifier']:(indices['amplifier'] + 128)] = tmp.reshape(header['num_amplifier_channels'], 128)

    if header['num_board_adc_channels'] > 0:
        tmp = np.fromfile(fid, dtype='uint16', count=128 * header['num_board_adc_channels'])
        data['board_adc_data'][range(header['num_board_adc_channels']),
        indices['board_adc']:(indices['board_adc'] + 128)] = tmp.reshape(header['num_board_adc_channels'], 128)

    if header['num_board_dac_channels'] > 0:
        tmp = np.fromfile(fid, dtype='uint16', count=128 * header['num_board_dac_channels'])
        data['board_dac_data'][range(header['num_board_dac_channels']),
        indices['board_dac']:(indices['board_dac'] + 128)] = tmp.reshape(header['num_board_dac_channels'], 128)

    if header['num_board_dig_in_channels'] > 0:
        data['board_dig_in_raw'][indices['board_dig_in']:(indices['board_dig_in'] + 128)] = np.array(struct.unpack('<' + 'H' * 128, fid.read(256)))

    if header['num_board_dig_out_channels'] > 0:
        data['board_dig_out_raw'][indices['board_dig_out']:(indices['board_dig_out'] + 128)] = np.array(struct.unpack('<' + 'H' * 128, fid.read(256)))
        
# Define data_to_result function
def data_to_result(header, data, data_present):
    """Moves the header and data (if present) into a common object."""
    
    result = {}
    result['t'] = data['t']
    
    stim_parameters = {}
    stim_parameters['stim_step_size'] = header['stim_step_size']
    stim_parameters['charge_recovery_current_limit'] = header['recovery_current_limit']
    stim_parameters['charge_recovery_target_voltage'] = header['recovery_target_voltage']
    stim_parameters['amp_settle_mode'] = header['amp_settle_mode']
    stim_parameters['charge_recovery_mode'] = header['charge_recovery_mode']
    result['stim_parameters'] = stim_parameters
    
    result['stim_data'] = data['stim_data']
    result['spike_triggers'] = header['spike_triggers']
    result['notes'] = header['notes']
    result['frequency_parameters'] = header['frequency_parameters']
    
    if header['dc_amplifier_data_saved']:
        result['dc_amplifier_data'] = data['dc_amplifier_data']
        
    if header['num_amplifier_channels'] > 0:
        if data_present:
            result['compliance_limit_data'] = data['compliance_limit_data']
            result['charge_recovery_data'] = data['charge_recovery_data']
            result['amp_settle_data'] = data['amp_settle_data']
    
    if header['num_board_dig_out_channels'] > 0:
        result['board_dig_out_channels'] = header['board_dig_out_channels']
        if data_present:
            result['board_dig_out_data'] = data['board_dig_out_data']
        
    if header['num_board_dig_in_channels'] > 0:
        result['board_dig_in_channels'] = header['board_dig_in_channels']
        if data_present:
            result['board_dig_in_data'] = data['board_dig_in_data']
        
    if header['num_board_dac_channels'] > 0:
        result['board_dac_channels'] = header['board_dac_channels']
        if data_present:
            result['board_dac_data'] = data['board_dac_data']
        
    if header['num_board_adc_channels'] > 0:
        result['board_adc_channels'] = header['board_adc_channels']
        if data_present:
            result['board_adc_data'] = data['board_adc_data']
            
    if header['num_amplifier_channels'] > 0:
        result['amplifier_channels'] = header['amplifier_channels']
        if data_present:
            result['amplifier_data'] = data['amplifier_data']
            
    return result

# Define plot_channel function
def plot_channel(channel_name, result):
    # Find channel that corresponds to this name
    channel_found, signal_type, signal_index = find_channel_in_header(channel_name, result)
    
    # Plot this channel
    if channel_found:
        fig, ax = plt.subplots()
        ax.set_title(channel_name)
        ax.set_xlabel('Time (s)')
        
        if signal_type == 'amplifier_channels':
            ylabel = 'Voltage (microVolts)'
            signal_data_name = 'amplifier_data'
            
        elif signal_type == 'board_adc_channels':
            ylabel = 'Voltage (Volts)'
            signal_data_name = 'board_adc_data'
            
        elif signal_type == 'board_dac_channels':
            ylabel = 'Voltage (Volts)'
            signal_data_name = 'board_dac_data'
            
        elif signal_type == 'board_dig_in_channels':
            ylabel = 'Digital In Events (High or Low)'
            signal_data_name = 'board_dig_in_data'
            
        elif signal_type == 'board_dig_out_channels':
            ylabel = 'Digital Out Events (High or Low)'
            signal_data_name = 'board_dig_out_data'
            
        else:
            raise Exception('Plotting not possible; signal type ', signal_type, ' not found')

        ax.set_ylabel(ylabel)
        
        ax.plot(result['t'], result[signal_data_name][signal_index,:])
        ax.margins(x=0, y=0)
        
    else:
        raise Exception('Plotting not possible; channel ', channel_name, ' not found')
        
# Define load_file function
def load_file(filename):
    # Start timing
    tic = time.time()
    
    # Open file
    fid = open(filename, 'rb')
    filesize = os.path.getsize(filename)
    
    # Read file header
    header = read_header(fid)
    
    # Output a summary of recorded data
    print('Found {} amplifier channel{}.'.format(header['num_amplifier_channels'], plural(header['num_amplifier_channels'])))
    print('Found {} board ADC channel{}.'.format(header['num_board_adc_channels'], plural(header['num_board_adc_channels'])))
    print('Found {} board DAC channel{}.'.format(header['num_board_dac_channels'], plural(header['num_board_dac_channels'])))
    print('Found {} board digital input channel{}.'.format(header['num_board_dig_in_channels'], plural(header['num_board_dig_in_channels'])))
    print('Found {} board digital output channel{}.'.format(header['num_board_dig_out_channels'], plural(header['num_board_dig_out_channels'])))
    print('')
    
    # Determine how many samples the data file contains
    bytes_per_block = get_bytes_per_data_block(header)
    print('{} bytes per data block'.format(bytes_per_block))
    # How many data blocks remain in this file?
    data_present = False
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining > 0:
        data_present = True
        
    if bytes_remaining % bytes_per_block != 0:
        raise Exception('Something is wrong with file size : should have a whole number of data blocks')
        
    num_data_blocks = int(bytes_remaining / bytes_per_block)
    
    # Calculate how many samples of each signal type are present
    num_amplifier_samples = 128 * num_data_blocks
    num_board_adc_samples = 128 * num_data_blocks
    num_board_dac_samples = 128 * num_data_blocks
    num_board_dig_in_samples = 128 * num_data_blocks
    num_board_dig_out_samples = 128 * num_data_blocks

    record_time = num_amplifier_samples / header['sample_rate']

    # Output a summary of contents of header file
    if data_present:
        print('File contains {:0.3f} seconds of data.  Amplifiers were sampled at {:0.2f} kS/s.'.format(record_time, header['sample_rate'] / 1000))
    else:
        print('Header file contains no data.  Amplifiers were sampled at {:0.2f} kS/s.'.format(header['sample_rate'] / 1000))

    if data_present:
        # Pre-allocate memory for data
        print('')
        print('Allocating memory for data...')

        data = {}
        data['t'] = np.zeros(num_amplifier_samples, dtype=np.int)

        data['amplifier_data'] = np.zeros([header['num_amplifier_channels'], num_amplifier_samples], dtype=np.uint)
        
        if header['dc_amplifier_data_saved']:
            data['dc_amplifier_data'] = np.zeros([header['num_amplifier_channels'], num_amplifier_samples], dtype=np.uint) * header['dc_amplifier_data_saved']
          
        data['stim_data_raw'] = np.zeros([header['num_amplifier_channels'], num_amplifier_samples], dtype=np.int)
        data['stim_data'] = np.zeros([header['num_amplifier_channels'], num_amplifier_samples], dtype=np.int)

        data['board_adc_data'] = np.zeros([header['num_board_adc_channels'], num_board_adc_samples], dtype=np.uint)
        data['board_dac_data'] = np.zeros([header['num_board_dac_channels'], num_board_dac_samples], dtype=np.uint)

        # by default, this script interprets digital events (digital inputs and outputs) as booleans
        # if unsigned int values are preferred(0 for False, 1 for True), replace the 'dtype=np.bool' argument with 'dtype=np.uint' as shown
        # the commented line below illustrates this for digital input data; the same can be done for digital out

        #data['board_dig_in_data'] = np.zeros([header['num_board_dig_in_channels'], num_board_dig_in_samples], dtype=np.uint)
        data['board_dig_in_data'] = np.zeros([header['num_board_dig_in_channels'], num_board_dig_in_samples], dtype=np.bool)
        data['board_dig_in_raw'] = np.zeros(num_board_dig_in_samples, dtype=np.uint)
        data['board_dig_out_data'] = np.zeros([header['num_board_dig_out_channels'], num_board_dig_out_samples], dtype=np.bool)
        data['board_dig_out_raw'] = np.zeros(num_board_dig_out_samples, dtype=np.uint)

        # Read sampled data from file.
        print('Reading data from file...')

        # Initialize indices used in looping
        indices = {}
        indices['amplifier'] = 0
        indices['board_adc'] = 0
        indices['board_dac'] = 0
        indices['board_dig_in'] = 0
        indices['board_dig_out'] = 0

        print_increment = 10
        percent_done = print_increment
        for i in range(num_data_blocks):
            read_one_data_block(data, header, indices, fid)
            # Increment all indices in 128
            indices = {k: v + 128 for k, v in indices.items()}         

            fraction_done = 100 * (1.0 * i / num_data_blocks)
            if fraction_done >= percent_done:
                print('{}% done...'.format(percent_done))
                percent_done = percent_done + print_increment

        print('100% done...')

        # Make sure we have read exactly the right amount of data
        bytes_remaining = filesize - fid.tell()
        if bytes_remaining != 0: raise Exception('Error: End of file not reached.')


    fid.close()

    if (data_present):
        print('Parsing data...')

        # Extract digital input channels to separate variables.
        for i in range(header['num_board_dig_in_channels']):
            data['board_dig_in_data'][i, :] = np.not_equal(np.bitwise_and(data['board_dig_in_raw'],
                                                                          (1 << header['board_dig_in_channels'][i][
                                                                              'native_order'])), 0)

        # Extract digital output channels to separate variables.
        for i in range(header['num_board_dig_out_channels']):
            data['board_dig_out_data'][i, :] = np.not_equal(np.bitwise_and(data['board_dig_out_raw'],
                                                                           (1 << header['board_dig_out_channels'][i][
                                                                               'native_order'])), 0)

        # Extract stimulation data
        data['compliance_limit_data'] = np.bitwise_and(data['stim_data_raw'], 32768) >= 1 # get 2^15 bit, interpret as True or False
        data['charge_recovery_data'] = np.bitwise_and(data['stim_data_raw'], 16384) >= 1 # get 2^14 bit, interpret as True or False
        data['amp_settle_data'] = np.bitwise_and(data['stim_data_raw'], 8192) >= 1 # get 2^13 bit, interpret as True or False
        data['stim_polarity'] = 1 - (2*(np.bitwise_and(data['stim_data_raw'], 256) >> 8)) # get 2^8 bit, interpret as +1 for 0_bit or -1 for 1_bit
        
        curr_amp = np.bitwise_and(data['stim_data_raw'], 255) # get least-significant 8 bits corresponding to the current amplitude
        data['stim_data'] = curr_amp * data['stim_polarity'] # multiply current amplitude by the correct sign

        # # Scale voltage levels appropriately.
        data['amplifier_data'] = np.multiply(0.195,
                                             (
                                             data['amplifier_data'].astype(np.int32) - 32768))  # units = microvolts
        data['stim_data'] = np.multiply(header['stim_step_size'], data['stim_data'] / 1.0e-6)

        if header['dc_amplifier_data_saved']:
            data['dc_amplifier_data'] = np.multiply(-0.01923,
                                                    (data['dc_amplifier_data'].astype(
                                                        np.int32) - 512))  # units = volts

        data['board_adc_data'] = np.multiply(0.0003125, (data['board_adc_data'].astype(np.int32) - 32768)) # units = volts
        data['board_dac_data'] = np.multiply(0.0003125, (data['board_dac_data'].astype(np.int32) - 32768)) # units = volts

        # Check for gaps in timestamps.
        num_gaps = np.sum(np.not_equal(data['t'][1:] - data['t'][:-1], 1))
        if num_gaps == 0:
            print('No missing timestamps in data.')
        else:
            print('Warning: {0} gaps in timestamp data found.  Time scale will not be uniform!'.format(num_gaps))

        # Scale time steps (units = seconds).
        data['t'] = data['t'] / header['sample_rate']

        # If the software notch filter was selected during the recording, apply the
        # same notch filter to amplifier data here.
        if header['notch_filter_frequency'] > 0 and header['version']['major'] < 3:
            print_increment = 10
            percent_done = print_increment
            for i in range(header['num_amplifier_channels']):
                data['amplifier_data'][i, :] = notch_filter(data['amplifier_data'][i, :], header['sample_rate'],
                                                            header['notch_filter_frequency'], 10)
                if fraction_done >= percent_done:
                    print('{}% done...'.format(percent_done))
                    percent_done = percent_done + print_increment
    else:
        data = []

    # Move variables to result struct.
    result = data_to_result(header, data, data_present)

    print('Done!  Elapsed time: {0:0.1f} seconds'.format(time.time() - tic))
    
    return result, data_present

# Define function print_all_channel_names
def print_all_channel_names(result):
    
    # Print all amplifier_channels
    if 'amplifier_channels' in result:
        print_names_in_group(result['amplifier_channels'])
    
    # Print all board_adc_channels
    if 'board_adc_channels' in result:
        print_names_in_group(result['board_adc_channels'])
          
    # Print all board_dac_channels
    if 'board_dac_channels' in result:
        print_names_in_group(result['board_dac_channels'])
    
    # Print all board_dig_in_channels
    if 'board_dig_in_channels' in result:
        print_names_in_group(result['board_dig_in_channels'])
    
    # Print all board_dig_out_channels
    if 'board_dig_out_channels' in result:
        print_names_in_group(result['board_dig_out_channels'])
    
# Define function print_names_in_group
def print_names_in_group(signal_group):
    for this_channel in signal_group:
        print(this_channel['custom_channel_name'])