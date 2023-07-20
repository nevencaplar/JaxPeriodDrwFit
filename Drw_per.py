#!/usr/bin/env python3
import numpy as np
import time
import JaxPeriodDrwFit

from tape.ensemble import Ensemble
from tape.utils import ColumnMapper
if __name__ == '__main__':
    print('Hello World!')

    ##########
    # creating ensamble
    # TODO: do this better
    t_true = np.loadtxt('/epyc/users/suberlak/2023_DRW_SINUSOID_COMBINE/data/t_true.txt')
    data_all = \
        np.load('/epyc/users/suberlak/2023_DRW_SINUSOID_COMBINE/data/COMB_001_054_009.npy', allow_pickle=True)
    id, t, y, yerr, filter = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    for i in range(100):
        data = data_all[()].get(i)
        downsample_int = np.sort(np.random.choice(np.arange(len(t_true)), 100))
        t_single = t_true[downsample_int]
        id = np.append(id, np.full(len(downsample_int), i))
        filter_single = np.full(len(t_single), 'r')
        t = np.append(t, t_single)
        filter = np.append(filter, filter_single)

        y_err_single = np.full(len(t_single), 0.001)
        yerr = np.append(yerr, np.full(len(t_single), 0.001))

        y_pre = data['y_tot'][downsample_int]
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
    ens.client.close()
    ##########

    JaxPeriodDrwFit_instance = JaxPeriodDrwFit.JaxPeriodDrwFit()

    t = single_lc['t'].values
    y = single_lc['y'].values
    yerr = single_lc['yerr'].values
    t1 = time.time()
    test_single_lc_res = JaxPeriodDrwFit_instance.optimize_map(100, t, y, yerr)
    t2 = time.time()
    print(f'Execution time for single lc is {t2 - t1} sec')
    print('Best result is:' + str(test_single_lc_res))
    t1 = time.time()
    test_single_lc_res = JaxPeriodDrwFit_instance.optimize_map(100, t, y, yerr)
    t2 = time.time()
    print(f'Execution time for second run with single lc is {t2 - t1} sec')
    print('Best result is:' + str(test_single_lc_res))
