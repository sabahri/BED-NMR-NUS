'''
Compiling most of what I've done into one place to finally implement
'''

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as pl
from scipy.signal import find_peaks
import nmrglue as ng
import operator 
import os
import sys
import math
import cmath
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import warnings
import shutil
import contextlib
np.random.seed = 1234

########################################################
######### SER Raw Data Reduction for Simulation ########
########################################################

def fetch_fid(data_dir, filename):

    abs_data_dir = os.path.abspath(data_dir)
    ser_path = os.path.join(abs_data_dir, filename)
    
    if not os.path.exists(ser_path):
        raise FileNotFoundError(f"SER file not found at {ser_path}")
        
    return(ser_path)

def process_binary_chunks(input_file, output_file, chunk_size=2048, chunks_to_remove=None):
    """
    Read binary file in chunks and optionally remove specific chunks.
    
    Args:
        input_file: Path to input binary file
        output_file: Path to output binary file
        chunk_size: Size of each chunk in bytes (default 2048)
        chunks_to_remove: List of chunk indices to remove (0-based), e.g. [0,2] removes first and third chunks
    """
    if chunks_to_remove is None:
        chunks_to_remove = []
    
    # Convert to set for O(1) lookup
    chunks_to_remove = set(chunks_to_remove)
    
    with open(input_file, 'rb') as fin, open(output_file, 'wb') as fout:
        chunk_num = 0
        while True:
            # Read chunk_size bytes
            chunk = fin.read(chunk_size)
            if not chunk:  # End of file
                break
                
            # If this isn't a chunk we want to remove, write it
            if chunk_num not in chunks_to_remove:
                fout.write(chunk)
                
            chunk_num += 1
    
    return chunk_num  # Return total number of chunks processed


def analyze_binary_file(filename, chunk_size=2048):
    """
    Analyze a binary file and print information about its chunks.
    Not really needed, just for troubleshooting purposes
    """

    with open(filename, 'rb') as f:
        # Get total file size
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        f.seek(0)  # Return to start
        
        # Calculate number of chunks
        full_chunks = file_size // chunk_size
        remainder = file_size % chunk_size
        
        print(f"File size: {file_size} bytes")
        print(f"Number of {chunk_size}-byte chunks: {full_chunks}")
        if remainder:
            print(f"Remainder bytes: {remainder}")
            
        # Read and print first few bytes of each chunk
        print("\nFirst few bytes of each chunk:")
        for i in range(full_chunks + (1 if remainder else 0)):
            f.seek(i * chunk_size)
            chunk = f.read(min(16, chunk_size))  # Read first 16 bytes of chunk
            hex_preview = ' '.join(f'{b:02x}' for b in chunk)
            #print(f"Chunk {i}: {hex_preview}...")


def create_ser(input_file, output_file, t_sampled, chunk_size=2048):
    try:
        with open(input_file, 'rb') as fin, open(output_file, 'wb') as fout:
            # Print some debug info
            #print(f"Reading from: {input_file}")
            #print(f"Writing to: {output_file}")
            #print(f"Number of time points to sample: {len(t_sampled)}")
            
            for idx in t_sampled:
                fin.seek(idx * chunk_size)
                chunk = fin.read(chunk_size)
                if chunk:
                    fout.write(chunk)
                else:
                    print(f"Warning: Could not read chunk at index {idx}")
                    
            # Verify file was written
            fout.flush()
            #print(f"Output file size: {os.path.getsize(output_file)} bytes")
    except IOError as e:
        raise IOError(f"Error processing binary files: {str(e)}")

def write_nuslist(t_acquired, output_path):
    """
    Save NUS sampling schedule to a file with each timepoint on a new line.
    
    Args:
        ts (numpy.ndarray): Array of timepoints in the NUS schedule
        output_path (str): Path where the schedule file should be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure ts is a numpy array and convert to integers if needed
        t_acquired = np.asarray(t_acquired, dtype=int)
        
        # Sort timepoints if not already sorted
        t_acquired = np.sort(t_acquired)
        
        # Write timepoints to file, one per line
        with open(output_path, 'w') as f:
            for t in t_acquired:
                f.write(f"{t}\n")
                
        return True
        
    except Exception as e:
        print(f"Error saving NUS schedule: {str(e)}")
        return False

########################################################
######### Exponential Sampling Initiation ##############
########################################################

def initiate_exp_acqu(decay_rate, num_samples, D):
	# Randomly sample time points according to inverse CDF of exponential
    num_samples = num_samples - 1
    t_range = np.arange(1,D)
    t_range = t_range[:,None]
    t_prob = np.exp(- decay_rate * t_range)

    t_prob[0] = 0
    t_prob /= t_prob.sum()
    t_cdf = t_prob.cumsum()

    us = (np.random.rand() + np.arange(0,num_samples)) / num_samples

    ts_0 = (us[:,None] <= t_cdf[None,:]).argmax(1)
    ts_0 = np.concatenate(([0], ts_0))
    return(ts_0)

###################################################################################
######### Wrapper for nmrglue operations on ser_nus and acqu2s files ##############
###################################################################################


@contextlib.contextmanager
def temp_ser_file(custom_ser_path):
    """
    Context manager that temporarily renames a custom ser file to 'ser' for nmrglue operations.
    
    Args:
        custom_ser_path (str): Path to your custom ser file
        
    Yields:
        str: Directory containing the temporarily renamed ser file
        
    Example:
        >>> with temp_ser_file('path/to/custom_ser') as temp_dir:
        >>>     # nmrglue operations here - will use temp_dir/ser
        >>>     dic, data = ng.bruker.read(temp_dir)
    """
    # Get the directory and filename
    dir_path = os.path.dirname(custom_ser_path)
    if not dir_path:
        dir_path = '.'
    
    # Path for the standard 'ser' file
    standard_ser_path = os.path.join(dir_path, 'ser')
    
    # Store original ser file if it exists
    original_exists = False
    if os.path.exists(standard_ser_path):
        original_exists = True
        backup_path = standard_ser_path + '.backup'
        shutil.move(standard_ser_path, backup_path)
    
    try:
        # Copy custom file to standard ser name
        shutil.copy2(custom_ser_path, standard_ser_path)
        
        # Yield the directory for nmrglue operations
        yield dir_path
        
    finally:
        # Clean up: remove temporary ser file
        if os.path.exists(standard_ser_path):
            os.remove(standard_ser_path)
            
        # Restore original ser file if it existed
        if original_exists:
            shutil.move(backup_path, standard_ser_path)

def process_custom_ser(custom_ser_path, nmrglue_func):
    """
    Process a custom ser file using an nmrglue function that expects 'ser'.
    
    Args:
        custom_ser_path (str): Path to your custom ser file
        nmrglue_func (callable): Function that processes the ser file
        
    Returns:
        The result of nmrglue_func
        
    Example:
        >>> def my_nmrglue_processing(path):
        >>>     dic, data = ng.bruker.read(path)
        >>>     return dic, data
        >>> 
        >>> result = process_custom_ser('path/to/custom_ser', my_nmrglue_processing)
    """
    with temp_ser_file(custom_ser_path) as temp_dir:
        result = nmrglue_func(temp_dir)
    return result

@contextlib.contextmanager
def temp_acqu_params(dir_path, td1, nus_td):
    """
    Context manager that temporarily modifies TD and NusTD in acqu2s file.
    """
    acqu2s_path = os.path.join(dir_path, 'acqu2s')
    backup_path = None
    
    try:
        # Backup original acqu2s if it exists
        if os.path.exists(acqu2s_path):
            backup_path = acqu2s_path + '.backup'
            shutil.copy2(acqu2s_path, backup_path)
            
            # Read and modify the file
            with open(acqu2s_path, 'r') as f:
                lines = f.readlines()
            
            # Create modified content
            modified_lines = []
            for line in lines:
                # Print line for debugging
                #print(f"Reading line: {line.strip()}")
                if '##$TD=' in line:  # Changed to match exact format
                    modified_lines.append(f'##$TD= {nus_td}\n')  # Use nus_td instead of td1
                    #print(f"Modified TD line to: ##$TD= {nus_td}")
                else:
                    modified_lines.append(line)
            
            # Write modified content
            with open(acqu2s_path, 'w') as f:
                f.writelines(modified_lines)
        
        yield dir_path
        
    finally:
        # Restore original file if it existed
        if backup_path and os.path.exists(backup_path):
            shutil.move(backup_path, acqu2s_path)

################################################################################
########### Feature Matrix, Posterior Mean, and Posterior Inverse ##############
################################################################################

def update_X(freqs, t_acquired, D):

    N = len(t_acquired)
    Phi = np.exp(1j*freqs[None,:] * t_acquired[:,None]) / np.sqrt(N)

    return(Phi)

def update_m_post(Phi, reg, y):

    N, D = Phi.shape
    Phi_c = np.conjugate(Phi)
    reg = reg / N
    
    post_var = np.linalg.inv(np.matmul(Phi_c.T, Phi) + reg * np.eye(D))
    post_mean = np.matmul(post_var, np.matmul(Phi_c.T, y[:,None]))

    return(post_mean)
    
def update_ys_post(Phi, mpost):

    N, D = Phi.shape
    ys_post = np.matmul(Phi, mpost)
    ys_post = ys_post[:,None]

    return(ys_post)

def update_ts(t_old, t_new):

    t_full = np.sort(np.append(t_old, t_new))

    return(t_full)

def update_ys(ys_old, t_old, t_new):

    t_full = np.append(t_old, t_new)
    
    ys_new = st1[t_new] + sigma * np.random.randn(len(t_new))
    ys_updated = np.append(ys_old, ys_new)
    
    t_ind = np.argsort(t_full)
    t_full = t_full[t_ind]
    ys_updated = ys_updated[t_ind]

    return(ys_updated)

def list_unsampled_t(t_acquired):

    unsampled = np.array([x for x in t_range if x not in t_acquired])

    return(unsampled)

def rms(arr, start, end):

    sum_squares = 0
    for i in range(start, end):
        sum_squares += arr[i]**2
        
    RMS = np.sqrt(sum_squares/np.abs(end - start))

    return(RMS)

###############################################
########### Relaxation Estimation #############
###############################################

def exp_decay(t, A, T2):
    # Why is this impacting the fit's weighting towards the later points?
    A = max_amp

    return(A * np.exp(-t/T2))

def find_all_maxima(signal_values, height_threshold):

    amplitudes = np.abs(signal_values)
    # Find local maxima
    peaks, _ = find_peaks(amplitudes, height=[height_threshold,np.max(amplitudes)])
    # Include absolute maximum by checking first point
    if amplitudes[0] > amplitudes[1]:
        peaks = np.concatenate(([0], peaks))

    return peaks

# Dynamic Peak Detector
def dynamic_peaks(spectrum, N_num, rms_noise, h):

    max_peaks_expected = 1.5*N_num
    spec_peaks, _ = find_peaks((spectrum.ravel()).real, height=h)

    if len(spec_peaks) > max_peaks_expected:
        h = h + 3*rms_noise
    elif len(spec_peaks) < max_peaks_expected / 2:
        h = h - rms_noise

    return(spec_peaks, h)

def fit_T2_and_correlate(t_acquired, y, spectrum, ppmthresh):
	# This needs to be converted to 2D
    amplitudes = np.abs(y)
    tthresh = 0.2*np.max(amplitudes)   
    tpeaks = find_all_maxima(amplitudes, tthresh)

    # Fit exponential to detected peaks
    weights = np.exp(t_acquired[tpeaks]/100) 

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        popt, _ = curve_fit(exp_decay, t_acquired[tpeaks], amplitudes[tpeaks], p0=[np.max(amplitudes),50], 
                    sigma=1/weights)
    
    # Extracting T2 from exponential fit
    T2_est = popt[1]
    # Signal envelope
    envelope = max_amp*np.exp(-np1/T2_est)
    # ridge_freqs is the m_post calculations
    om, _ = find_peaks((spectrum.ravel()).real, height=ppmthresh)  # <-- make ppmthresh dynamic
    
    return(T2_est, envelope, om)

def T2_kernel(t_star, t_acquired, y, spectrum):

    ts_star = np.append(t_acquired, t_star)
    envelope = (fit_T2_and_correlate(t_acquired, y, spectrum,5)[1])[:,None]
    condensed_envelope = envelope[ts_star]

    return(condensed_envelope)


###############################################
########### Entropy Calculations ##############
###############################################

def Q(reg, var, t_star, t_acquired, D, y, spectrum, alpha):

    N = len(t_acquired)
    if N != N1:
        ts_star = np.append(t_acquired, t_star)
        # Why are we scaling the regularization parameter here? I don't remember.
        reg = reg / len(ts_star)
        # alpha is a scaling factor used to make entropy calculations meaningful
        freqs = 2 * alpha * np.pi * np.arange(0,D) / D
        X_star = np.exp(1j*freqs[None,:] * ts_star[:,None])/ np.sqrt(len(ts_star)) #/ D

        X_star_c = np.conjugate(X_star)

        S_post_star = np.linalg.inv(X_star_c.T @ X_star + reg * np.eye(D))
        cond_num_star = np.linalg.cond(S_post_star)
        SK = X_star @ S_post_star @ X_star_c.T
        
        condensed_envelope = T2_kernel(t_star, t_acquired, y, spectrum)
        #SK = SK @ condensed_envelope
        # vdM operates in the original measurement space, while
        # reg operates in the normalized feature space, so reg is indeed scaled
        # while vdM is not
        vdM = (var)*np.eye(N+1)
        Sigma_predictive = SK + vdM
    
        Sigma_old = Sigma_predictive[:-1,:-1]
        cond_num_sigma = np.linalg.cond(Sigma_old)
        Sigma_u = Sigma_predictive[-1,-1]
        C = Sigma_predictive[-1,:-1]
        C = C[:,None]
        
        C_c = np.conjugate(C)
        Sigma_tilda = Sigma_u - np.matmul(C_c.T, np.matmul(np.linalg.inv(Sigma_old), C))
        # Entropy Calculation for a Multivariate Gaussian
        logabsdet = np.log(np.abs(Sigma_tilda))
        ent = 0.5 * (1 + np.log(2*np.pi)) + 0.5*logabsdet

        return(ent)

def find_max_ent(entropy_array, t_old, nt_bloc):

    unsampled = list_unsampled_t(t_old)
    # (n,1) vs. (n,)
    unsampled = unsampled[:,0]
    entropy_array = entropy_array[:,0]
    Q_star_dict = {}

    for a in range(len(unsampled)):
        Q_star_dict[unsampled[a]] = entropy_array[a]
    
    res = dict(sorted(Q_star_dict.items(), key=operator.itemgetter(1), reverse=True)[:nt_bloc])
    ts_update = np.sort(list(res.keys()))
    ts_update = np.unique(ts_update)

    return(ts_update)

###############################################
########### Some Processing Fxns ##############
###############################################

def update_ZF(t_sampled, y, D):
	t_zf = np.zeros((len(t_sampled),))
	y_zf = np.zeros((len(t_sampled),))

	t_sampled_zf = np.append(ts, t_zf)
	y_zf = np.append(ys, y_zf)

	modes_zf = D*2

	return(t_zf, y_zf, modes_zf)

def tppi_transpose(data):
# transpose and convert indirect real points in complex points with imag=0
    m = data.shape[1]
    n = data.shape[0]
    tmpdata = np.zeros((m,n),dtype=complex)
# reorder:
    for x in range(m):
        for y in range(n):
            tmpdata[x][y] += complex(data[y][x],0)
    return tmpdata

def tppi_after_correction(data):
    # Removes 2nd half of indirect dimension because 2 real points correspond to 1 freq. point (Nyquist)
    # Also mirroring image for whatever reason
    m = data.shape[0]
    n = data.shape[1]
    q = int(m/2)
    tmpdata = np.zeros((q,n),dtype=data.dtype)
    for y in range(q):
        for x in range(n):
            tmpdata[y][x] = data[q-1-y][x]
    return(tmpdata)

def my_win(data):
    m = data.shape[0]
    n = data.shape[1]
    for y in range(m):
        for x in range(n):
            data[y][x] *= math.exp(-(x*3)/(n))
    return data

##########################################
######### Plotting Function ##############
##########################################

def plot_spectrum(data, udic, axlims_x, axlims_y, stage=""):
    """Plot 2D NMR spectrum with proper axes and contours"""
    # Set up contour parameters 
    cmap = matplotlib.cm.Reds_r    # contour map (colors to use for contours)
    contour_start =  1.5*(10**5)           #  contour level start value
    contour_num = 64                # number of contour levels
    contour_factor = 1.1          # scaling factor between contour levels
    
    # Calculate contour levels
    cl = contour_start * contour_factor ** np.arange(contour_num) 
    
    # Get ppm scales from universal dictionary
    uc_1h = ng.fileiobase.uc_from_udic(udic, dim=1)
    ppm_1h = uc_1h.ppm_scale()
    ppm_1h_0, ppm_1h_1 = uc_1h.ppm_limits()
    
    uc_15n = ng.fileiobase.uc_from_udic(udic, dim=0)
    ppm_15n = uc_15n.ppm_scale()
    ppm_15n_0, ppm_15n_1 = uc_15n.ppm_limits()
    
    # Create the figure
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    
    # Plot contours
    ax.contour(data, cl, cmap=cmap,extent=(ppm_1h_0, ppm_1h_1, ppm_15n_0, ppm_15n_1))
    
    # Decorate axes
    ax.set_ylabel("15N (ppm)")
    ax.set_xlabel("1H (ppm)")
    ax.set_title(f"Protein 2D hNH Spectrum - {stage}")
    
    # Set proper spectrum limits
    ax.set_xlim(axlims_x[0],axlims_x[1])
    ax.set_ylim(axlims_y[0],axlims_y[1])
    
    plt.show()

# Noise variance
sigma = 0.002 * (1 + 0*1j)
sigma2 = np.conjugate(sigma)*sigma
sigma2 = 0.6

# define prior over weights (i.e. Fourier spectrum amplitudes)

chi = 1.0*(1 + 0*1j)
chi2 = np.conjugate(chi)*chi
chi2 = 10000

# define Bayesian posterior mean and covariance
l = sigma2 / chi2 
l = 0.005

##########################################
########### Running DIANA ################
##########################################

def live_nus(ser_path, t_acquired, chunk_size, D, reg, noise_var, nt_bloc, num_iters):
    """
    Execute Live NUS procedure on binary NMR data.
    
    Args:
        ser_path (str): Path to the binary ser file
        t_acquired (np.array): Initially acquired timepoints
        chunk_size (int): Size of each FID chunk in bytes
        D (int): Total number of points in full dataset
        reg (float): Regularization parameter for ridge regression
        noise_var (float): Noise variance estimate
        nt_bloc (int): Number of points to acquire in each iteration
        num_iters (int): Number of iterations to run
        
    Returns:
        tuple: (final_timepoints, final_reconstruction)
    """
    
    # Initialize storage
    t_sampled = t_acquired.copy()
    freqs = 2 * np.pi * np.arange(0, D) / D
    
    # Read initial FIDs
    ys_sampled = []
    with open(ser_path, 'rb') as f:
        for t in t_sampled:
            f.seek(t * chunk_size)
            fid_data = f.read(chunk_size)
            # Convert binary FID to complex number
            # Assuming each FID point is 8 bytes (4 for real, 4 for imag)
            fid_complex = np.frombuffer(fid_data, dtype=np.complex128)
            ys_sampled.append(fid_complex[0])  # Take first point of FID
    
    ys_sampled = np.array(ys_sampled)
    
    for i in range(num_iters):
        print(f"Iteration {i+1}/{num_iters}")
        
        # Get unsampled timepoints
        t_unsampled = list_unsampled_t(t_sampled)
        if len(t_unsampled) == 0:
            break
            
        # Update feature matrix and get reconstruction
        X = update_X(freqs, t_sampled, D)
        m_post = update_m_post(X, reg, ys_sampled)
        ys_post = update_ys_post(X, m_post)
        
        # Calculate entropy for unsampled points
        Q_star = []
        for t in t_unsampled:
            entropy = Q(reg, noise_var, t, t_sampled, D, ys_sampled, m_post.real, 0.008)
            Q_star.append(entropy)
        Q_star = np.array(Q_star)
        
        # Find points with maximum entropy
        t_opt = find_max_ent(Q_star[:,None], t_sampled, nt_bloc)
        
        # Read new FIDs
        new_ys = []
        with open(ser_path, 'rb') as f:
            for t in t_opt:
                f.seek(t * chunk_size)
                fid_data = f.read(chunk_size)
                fid_complex = np.frombuffer(fid_data, dtype=np.complex64)
                new_ys.append(fid_complex[0])
        
        # Update sampled points and FIDs
        ys_sampled = np.append(ys_sampled, new_ys)
        t_sampled = update_ts(t_sampled, t_opt)
        
        # Optional: Create intermediate ser file for visualization
        temp_ser = f"temp_iter_{i}.ser"
        create_ser(ser_path, temp_ser, t_sampled, chunk_size)
        
        print(f"Points sampled: {len(t_sampled)}/{D}")
        
    return t_sampled, m_post

# Example usage:
# t_sampled, reconstruction = live_nus_binary(
#     ser_path="path/to/ser",
#     t_acquired=initial_points,
#     chunk_size=2048,  # Typical Bruker FID size
#     D=total_points,
#     reg=0.005,
#     noise_var=0.6,
#     nt_bloc=16,
#     num_iters=10
# )

'''
def live_nus(t_acquired, D, y, reg, noise_var, nt_bloc, num_iters):
    # Execute Live NUS procedure

    #num_iters = int((D-len(t_acquired)) / nt_bloc) + 1

    t_sampled = t_acquired
    freqs = 2 * np.pi * np.arange(0,D) / D
    ys_sampled = y 
    signal_to_noise = np.zeros((num_iters, len(peaks1)))

    for i in range(0, num_iters):

        t_unsampled = list_unsampled_t(t_acquired)
        X = update_X(freqs, t_acquired, D)
        m_post = update_m_post(X, var, ys_sampled)
        ys_post = update_ys_post(X, m_post)


        print("Iteration number:", i+1)
        print("Number of sampled points:", t_sampled.shape[0])
        print("Number of unsampled points:", t_unsampled.shape[0])
        print("Feature matrix size:", X.shape)
        print("Estimated T2:", T2_estimate)

        if t_unsampled.shape[0] > 0:
            Q_star = np.array([Q(l, noise_var, t, t_sampled, K, ys_sampled, m_post, 0.008) for t in t_unsampled])

            # Update parameters
            t_opt = find_max_ent(Q_star, t_sampled, t_bloc)
        
            t_opt_indices = np.array([np.where(t_unsampled == t)[0][0] for t in t_opt])
            Q_opt = Q_star[t_opt_indices]
        
            #pthresh = dynamic_peaks(m_post, 8, Sk1_rms, pthresh)
            pl.scatter(t_unsampled, Q_star, c = 'lime', lw = 0.5, edgecolor = 'k')
            pl.scatter(t_opt, Q_opt, c = 'r', lw = 0.5, edgecolor = 'k')

            pl.set_xlim(0,D)
            #axs[2].set_ylim(0,2)
        
            for i in t_sampled:
                pl.axvline(x=i, c = 'r', linestyle = '--', lw = '0.5', alpha = 0.66)
        
            ys_sampled = update_ys(ys_sampled, t_sampled, t_opt)
            t_sampled = update_ts(t_sampled, t_opt)

        pl.show()
        '''