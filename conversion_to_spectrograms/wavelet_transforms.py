import numpy as np
import mne
import scipy


def butterworth_filter_pass(edf, channels_data, target_channel, source_freq, low_pass_freq, high_pass_freq):
    filtered = [None] * target_channel

    for index, cd in enumerate(channels_data):
        filtered[index] = mne.filter.filter_data(cd, source_freq, low_pass_freq, high_pass_freq, method='iir')

    return filtered


def rescale_morlet_plz(sample, test_config):
    """
    Rescale from shape (MAX_MORLET_FREQ, SECTOR_LENGTH) to shape
    (MORLET_FREQ_STEPS, SECTOR_LENGTH_STEPS)
    """

    NW = test_config.sector_length_steps
    FW = (test_config.sector_length  // test_config.sector_length_steps)

    if test_config.sector_length  == test_config.sector_length_steps:
        return sample

    sample = np.reshape(sample, (test_config.max_morlet_freq, NW, FW)).mean(axis=2)

    if test_config.max_morlet_freq != test_config.morlet_freq_steps:
        raise RuntimeError('Incomplete code')

    return sample


def morlet_wavelet_pass(channel_splitted_data, test_config, w = 6.0):
    """
    Performs wavelet transform over the given data. Returns 2D matrixes
    representing morlet transform application result for each of 4 channels for
    each of N samples.

    channel_splitted_data contains 4 channels, each has a set of splitted
    samples in it.
    """

    t, dt = np.linspace(0, test_config.sector_length / test_config.source_freq, test_config.sector_length, retstep=True)
    freq = np.linspace(1, test_config.max_morlet_freq, test_config.max_morlet_freq)
    fs = 1 / dt
    widths = w * fs / (2 * freq * np.pi)

    FW = (test_config.max_morlet_freq// test_config.morlet_freq_steps)
    FH = (test_config.sector_length // test_config.sector_length_steps)


    return t[::FH], freq[::FW], [
        [
            rescale_morlet_plz(scipy.signal.cwt(channel_splitted_data[channel][index], scipy.signal.morlet2, widths, w=w), test_config)
            for index in range(len(channel_splitted_data[channel]))
        ]
        for channel in range(test_config.num_target_channels)
    ]

def transpose_morlet_channel_data(morlet_channel_data, test_config):
    """
    Perform transposition of channel data so order changes from
    morlet_channel_data[channel][index] to morlet_channel_data[index][channel]
    """
    return [
        [
            morlet_channel_data[channel][index]
            for channel in range(test_config.num_target_channels)
        ]
        for index in range(len(morlet_channel_data[0]))
    ]

def abs_morlet_data(morlet):
    return np.abs(morlet)


def convert_person_segments_to_spectrograms(segments, test_config):

    t, freq, morlets = morlet_wavelet_pass(segments, test_config)

    person_spectrograms = transpose_morlet_channel_data(morlets, test_config)
    person_spectrograms = abs_morlet_data(person_spectrograms )

    return person_spectrograms


def normalize_spectrograms(spectrograms):
    return spectrograms / (spectrograms.max() - spectrograms.min()) - 0.5


def group_spectrograms_by_phoneme(labels, spectrograms):
    spectrograms_by_phoneme = [[] for _ in range(len(set(labels)))]

    for i, (spectrogram, label) in enumerate(zip(spectrograms, labels)):
        spectrograms_by_phoneme[label].append((spectrogram))

    return spectrograms_by_phoneme