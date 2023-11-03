#!/usr/bin/env python3
import numpy as np
# import time
import JaxPeriodDrwFit
# import dill as pickle
# import cloudpickle as pickle

from tape.ensemble import Ensemble
from tape.utils import ColumnMapper
if __name__ == '__main__':
    print('Hello World!')

    ##########
    # creating ensamble
    # TODO: do this better
    """
    t_true = np.loadtxt('/epyc/users/suberlak/2023_DRW_SINUSOID_COMBINE/data/t_true.txt')
    data_all = \
        np.load('/epyc/users/suberlak/2023_DRW_SINUSOID_COMBINE/data/COMB_001_054_009.npy', allow_pickle=True)
    """

    # generated in create_data script, to avoid epyc failure
    t_multi = np.load('/astro/users/ncaplar/data/t_multi.npy')
    y_multi = np.load('/astro/users/ncaplar/data/y_multi.npy')
    yerr_multi = np.load('/astro/users/ncaplar/data/yerr_multi.npy')

    id, t, y, yerr, filter = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    for i in range(5):
        # data = data_all[()].get(i)

        # get time for a single lightcurve
        t_true = t_multi[i]
        # sample 100 points from 200
        downsample_int = np.sort(np.random.choice(np.arange(len(t_true)), 100))
        # extract 100 times from 200
        t_single = t_true[downsample_int]

        id = np.append(id, np.full(len(downsample_int), i))
        filter_single = np.full(len(t_single), 'r')
        t = np.append(t, t_single)
        filter = np.append(filter, filter_single)

        # create custom errors
        y_err_single = np.full(len(t_single), 0.001)
        yerr = np.append(yerr, np.full(len(t_single), 0.001))

        # extract measurements; 100 from each lightcurve
        # y_pre = data['y_tot'][downsample_int]
        y_pre = y_multi[i][downsample_int]

        # create noise and add to lightcurves
        noise = np.random.normal(0, y_err_single)
        y = np.append(y, y_pre + noise)

    # columns assigned manually
    manual_colmap = ColumnMapper().assign(
        id_col="id", time_col="t", flux_col="y", err_col="yerr", band_col="filter"
    )

    ens = Ensemble()
    ens.from_source_dict({'id': id, "t": t, 'y': y, 'yerr': yerr, 'filter': filter},
                         column_mapper=manual_colmap)
    single_lc = ens.compute("source")[id == 0]
    # comment out line below if trying to run ensamble.batch
    # ens.client.close()
    ##########
    JaxPeriodDrwFit_instance = JaxPeriodDrwFit.JaxPeriodDrwFit()
    res = ens.batch(JaxPeriodDrwFit_instance.optimize_map, 't', 'y', 'yerr',
                    compute=True, meta=None, n_init=100)
    print(res)
    np.save(res)
    ens.client.close()
    """
    # https://github.com/nevencaplar/epyc_notebooks/blob/main/tiny_lsst.ipynb
    # has an example where I managed to run `something'

    t = single_lc['t'].values
    y = single_lc['y'].values
    yerr = single_lc['yerr'].values

    # This block shows that the code works on a single lightcurve
    # And it is faster second time
    t1 = time.time()
    test_single_lc_res = JaxPeriodDrwFit_instance.optimize_map(t, y, yerr, n_init=100)
    t2 = time.time()
    print(f'Execution time for single lc is {t2 - t1} sec')
    print('Best result is:' + str(test_single_lc_res))
    t1 = time.time()
    test_single_lc_res = JaxPeriodDrwFit_instance.optimize_map(t, y, yerr, n_init=100)
    t2 = time.time()
    print(f'Execution time for second run with single lc is {t2 - t1} sec')
    print('Best result is:' + str(test_single_lc_res))

    # ensamble batch does not work
    # but it works in a notebook?!?
    # what I thought is the minimal example is here, it is because it is not possible to
    # pickle jitted funtion
    # https://github.com/google/jax/issues/5043
    # https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
    pickle.dumps(JaxPeriodDrwFit_instance.optimize_map)
    """
    print('OK')
