import numpy as np
import scipy as sp 
from joblib import Parallel, delayed

import os
import glob
import fathon
from natsort import natsorted
from fathon import fathonUtils as fu


sjnr_to_analyze =  ['100307', '102816', '133019', '149741', '153732', '154532']

this_dfa_frequency_bands = ['broadband','alpha', 'delta', 'beta', 'gamma', 'theta']
gf_domain = False

nsim = 600
fs = 500
chunk = 50  

min_window = 2
max_window = 30


# Per-subject-group passband edges (individual alpha peaks, chosen by eye).
# Group A (100307, 133019, 149741, 153732) has lower individual alpha peak
# than group B (102816, 154532); theta/alpha/beta edges adjusted accordingly.
band_edges_hz_group_a = {
    'theta': (3, 6),
    'alpha': (6, 11),
    'beta':  (11, 25),
    'gamma': (25, 40),
}
band_edges_hz_group_b = {
    'theta': (3, 7),
    'alpha': (7, 13),
    'beta':  (13, 25),
    'gamma': (25, 40),
}

# Cutoff for delta/broadband_above split
delta_CUTOFF_HZ = 3

subjects_group_a = {'100307', '133019', '149741', '153732', 'noise-100307',   'noise-133019', 'noise-149741', 'noise-153732'}
subjects_group_b = {'102816', '154532', 'noise-102816', 'noise-154532'}


CYCLES_OF_LOWER_EDGE = 2


def design_filter(frequencyband, sjnr):
    """Design FIR filter for a given band and subject group.
    
    Returns (fir_taps, num_fir_taps) for filtered bands,
    or (None, 0) for broadband (no filtering).
    """
    if frequencyband == 'broadband':
        return None, 0

    if sjnr in subjects_group_a:
        band_edges = band_edges_hz_group_a
    else:
        band_edges = band_edges_hz_group_b

    if frequencyband == 'delta':
        # Lowpass at delta_CUTOFF_HZ
        reference_frequency_hz = delta_CUTOFF_HZ
        num_fir_taps = int(CYCLES_OF_LOWER_EDGE * fs / reference_frequency_hz)
        if num_fir_taps % 2 == 0:
            num_fir_taps += 1
        fir_taps = sp.signal.firwin(num_fir_taps, delta_CUTOFF_HZ,
                                     fs=fs, pass_zero='lowpass')

    else:
        # Standard bandpass
        low_edge_hz, high_edge_hz = band_edges[frequencyband]
        num_fir_taps = int(CYCLES_OF_LOWER_EDGE * fs / low_edge_hz)
        if num_fir_taps % 2 == 0:
            num_fir_taps += 1
        fir_taps = sp.signal.firwin(num_fir_taps, (low_edge_hz, high_edge_hz),
                                     fs=fs, pass_zero='bandpass')

    return fir_taps, num_fir_taps




winSizes = np.logspace(np.log10(min_window * fs), np.log10(max_window * fs),
                       dtype='int', num=20)
revSeg = True
polOrd = 3


def DFA_H(tseries, fir_taps, num_fir_taps):
    if fir_taps is not None:
        filtered = sp.signal.lfilter(fir_taps, 1, tseries)
        # Discard causal filter ramp-up
        filtered = filtered[num_fir_taps:]
    else:
        filtered = tseries

    env = np.abs(sp.signal.hilbert(filtered))
    del filtered

    a = fu.toAggregated(env)

    pydfa = fathon.DFA(a)

    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)

    H, H_intercept = pydfa.fitFlucVec(logBase=10)

    return H


def make_output_path(population_label, frequencyband, sjnr):
    """Construct output filename."""
    base_directory = '/Users/marcoaqil/Desktop/miami_mac/Graph-Stochastic-Wilson-Cowan-Model/test'

    filename = (f'DFAexp_{population_label}_{frequencyband}_hcpmeg_10min_20khzsim'
                f'_fftfilterdown500hz_{min_window}{max_window}sec_polyorder{polOrd}'
                f'_adaptaps_{sjnr}.npy')
    return os.path.join(base_directory, filename)


def load_and_project_simulations(file_paths, eigenvectors, num_output_nodes, num_samples_total, dtype):
    """Load simulation chunks, project to node space via eigenvectors."""
    sims = np.empty((num_output_nodes, num_samples_total), dtype=dtype)
    samples_loaded = 0
    for chunk_start in range(0, nsim, chunk):
        print(chunk_start)
        block = np.concatenate(
            [np.load(path)['arr_0'] for path in file_paths[chunk_start:chunk_start + chunk]],
            axis=-1
        )
        
        if not gf_domain:
            block = eigenvectors @ block

        sims[:, samples_loaded:samples_loaded + block.shape[1]] = block
        samples_loaded += block.shape[1]
        del block
    return sims


def run_dfa_bands(sims, sjnr, population_label, frequency_bands_to_run):
    """Run DFA for all requested bands on a single loaded dataset."""
    for frequencyband in frequency_bands_to_run:
        fir_taps, num_fir_taps = design_filter(frequencyband, sjnr)

        print(f"  Running DFA: {population_label} {frequencyband} ({sjnr})")
        all_H = Parallel(n_jobs=14, verbose=10)(
            delayed(DFA_H)(sims[vx], fir_taps, num_fir_taps) for vx in range(sims.shape[0])
        )

        output_path = make_output_path(population_label, frequencyband, sjnr)
        np.save(output_path, all_H)
        print(f"  Saved: {output_path}")


# Main loop: iterate over subjects (outer), bands (inner)
# to avoid reloading data for each band
for sjnr in sjnr_to_analyze:
    if 'noise' in sjnr:
        subject_number = sjnr.split('-')[1]
        subject_folder = f'/Users/marcoaqil/Desktop/miami_mac/Downloads/{subject_number}'
        default_number_of_nodes = 8000

        if os.path.isdir(subject_folder):
            number_of_nodes = len(np.load(os.path.join(subject_folder, 'eigenvalues.npy')))
        else:
            number_of_nodes = default_number_of_nodes

        sims = np.random.default_rng().normal(0, 1, size=(number_of_nodes, int(fs*nsim)))
        run_dfa_bands(sims, sjnr, 'noise', this_dfa_frequency_bands)
        del sims

    else:
        eigenvectors = np.load(
            f'/Users/marcoaqil/Desktop/miami_mac/Downloads/{sjnr}/eigenvectors.npy')

        # Infer sizes from first E file
        paths_excitatory = natsorted(glob.glob(
            f'/Users/marcoaqil/Desktop/miami_mac/Graph-Stochastic-Wilson-Cowan-Model'
            f'/test/sim/{sjnr}_long/beta_E_activity*.npz'))
        tmp = np.load(paths_excitatory[0])['arr_0']
        num_output_nodes = eigenvectors.shape[0]
        num_samples_per_sim = tmp.shape[1]
        num_samples_total = nsim * num_samples_per_sim
        data_dtype = tmp.dtype
        del tmp

        # Load E data once, run all bands
        print(f"\nLoading E data for {sjnr}...")
        sims = load_and_project_simulations(
            paths_excitatory, eigenvectors, num_output_nodes, num_samples_total, data_dtype)
        
        if gf_domain:
            pop_label_E = 'beta_E'
            pop_label_I = 'beta_I'
        else:
            pop_label_E = 'E'
            pop_label_I = 'I'

        run_dfa_bands(sims, sjnr, pop_label_E, this_dfa_frequency_bands)
        del sims

        # Load I data once, run all bands
        paths_inhibitory = natsorted(glob.glob(
            f'/Users/marcoaqil/Desktop/miami_mac/Graph-Stochastic-Wilson-Cowan-Model'
            f'/test/sim/{sjnr}_long/beta_I_activity*.npz'))
        print(f"\nLoading I data for {sjnr}...")
        sims = load_and_project_simulations(
            paths_inhibitory, eigenvectors, num_output_nodes, num_samples_total, data_dtype)
        run_dfa_bands(sims, sjnr, pop_label_I, this_dfa_frequency_bands)
        del sims