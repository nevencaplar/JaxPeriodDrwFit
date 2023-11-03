#!/usr/bin/env python3

import numpy as np
from eztao.carma import DRW_term
from eztao.ts import gpSimRand, gpSimByTime
from scipy.spatial import KDTree
import pandas as pd


def find_sky_overlaps(pointings_database, parameters):
    """
    Find the sky indices of the pointings database that overlap with the transient.
    """
    pointings_sky_pos = np.column_stack((pointings_database['_ra'].values, pointings_database['_dec'].values))
    transient_sky_pos = np.column_stack((parameters['ra'].values, parameters['dec'].values))

    transient_sky_pos_3D = np.vstack([np.cos(transient_sky_pos[:, 0]) * np.cos(transient_sky_pos[:, 1]),
                                      np.sin(transient_sky_pos[:, 0]) * np.cos(transient_sky_pos[:, 1]),
                                      np.sin(transient_sky_pos[:, 1])]).T
    pointings_sky_pos_3D = np.vstack([np.cos(pointings_sky_pos[:, 0]) * np.cos(pointings_sky_pos[:, 1]),
                                      np.sin(pointings_sky_pos[:, 0]) * np.cos(pointings_sky_pos[:, 1]),
                                      np.sin(pointings_sky_pos[:, 1])]).T
    # law of cosines to compute 3D distance
    max_3D_dist = np.sqrt(2. - 2. * np.cos(survey_radius))
    survey_tree = KDTree(pointings_sky_pos_3D)
    overlap_indices = survey_tree.query_ball_point(x=transient_sky_pos_3D, r=max_3D_dist)

    return overlap_indices, transient_sky_pos_3D, pointings_sky_pos_3D


if __name__ == '__main__':
    print('Hello World!')
    # amp is RMS amplitude of the process and tau is the decorrelation timescale
    amp = 0.2
    tau = 100

    # Create the kernel
    DRW_kernel = DRW_term(np.log(amp), np.log(tau))

    # specify how many lightcurves and observational details
    num_light_curves = 100
    snr = 10
    duration_in_days = 365 * 10
    num_observations = 200

    # Generate `num_light_curves` lightcurves
    # t, y, yerr are np.ndarray with shape = (num_light_curves, num_observations)
    t_multi, y_multi, yerr_multi = gpSimRand(
        carmaTerm=DRW_kernel,
        SNR=snr,
        duration=duration_in_days,
        N=num_observations,
        nLC=num_light_curves)

    np.save('/astro/users/ncaplar/data/t_multi.npy', t_multi)
    np.save('/astro/users/ncaplar/data/y_multi.npy', y_multi)
    np.save('/astro/users/ncaplar/data/yerr_multi.npy', yerr_multi)

    survey_fov_sqdeg = 9.6
    survey_fov_sqrad = survey_fov_sqdeg*(np.pi/180.0)**2
    survey_radius = np.sqrt(survey_fov_sqrad/np.pi)
    print('survey_radius:' + str(survey_radius))

    datadir = '/astro/users/ncaplar/github/redback/redback/tables'
    pointings_database_name = 'rubin_baseline_v3.0_10yrs.tar.gz'
    pointings_database = pd.read_csv(datadir + "/" + pointings_database_name, compression='gzip')

    parameters = pd.DataFrame(np.array([[0, 1, 2], [0.2813573, 1, 2], [-0.0978646, 1, 2]]).T,
                              columns=['id', 'ra', 'dec'])

    parameters_deg = parameters.copy()

    parameters['ra'] = parameters_deg['ra'] * np.pi/180
    parameters['dec'] = parameters_deg['dec'] * np.pi/180

    time_space_overlap, transient_sky_pos_3D, pointings_sky_pos_3D = \
        find_sky_overlaps(pointings_database, parameters)

    overlapping_database_iter = pointings_database.iloc[time_space_overlap[0]]
    single_filter_single_position = \
        overlapping_database_iter[overlapping_database_iter['filter'] == 'lsstr']['expMJD']
    t_input = single_filter_single_position.values - np.min(single_filter_single_position.values)

    t_multi_LSST, y_multi_LSST, yerr_multi_LSST = gpSimByTime(
        carmaTerm=DRW_kernel, SNR=snr, t=t_input,
        nLC=num_light_curves, log_flux=True)

    Per = 200
    phi = 0.23
    per_amp = 0.5
    y_per = per_amp * np.sin(t_multi_LSST[0]/Per + phi)

    y_multi_LSST += y_per

    np.save('/astro/users/ncaplar/data/t_multi_LSST.npy', t_multi_LSST)
    np.save('/astro/users/ncaplar/data/y_multi_LSST.npy', y_multi_LSST)
    np.save('/astro/users/ncaplar/data/yerr_multi_LSST.npy', yerr_multi_LSST)

    print('Finished')
