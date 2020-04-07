import numpy as np

from imlib.radial.misc import radial_bins
from imlib.array.misc import midpoints_of_series


def radial_spike_histogram(
    angle_timeseries,
    spikes_timeseries,
    bin_width,
    bin_occupancy=None,
    normalise=False,
    degrees=True,
):
    """
    From a timeseries of angles and spikes, calculate a radial spiking
    histogram

    :param angle_timeseries: array like tigitmeseries of angles
    :param spikes_timeseries: array like timeseries of spikes, with N spikes
    per timepoint
    :param bin_width: Size of bin used for histogram
    :param bin_occupancy: Array like timeseries of temporal occupancy of bins.
    If specified, the relative spike rates will be returned.
    :param normalise: Normalise the resulting histogram
    :param degrees: Use degrees, rather than radians
    :return: Histogram bin centers (in radians) and the spikes (or spike rate)
    per radial bin

    """
    spikes_per_bin, bins = np.histogram(
        angle_timeseries,
        weights=spikes_timeseries,
        bins=radial_bins(bin_width, degrees=degrees),
        density=normalise,
    )
    if bin_occupancy is not None:
        spikes_per_bin = np.divide(spikes_per_bin, bin_occupancy)
    if degrees:
        bin_centers = np.deg2rad(midpoints_of_series(bins))
    return bin_centers, spikes_per_bin
