#!/usr/bin/env python3
import numpy as np
# import time
import JaxPeriodDrwFit
from tqdm import tqdm
# import dill as pickle
# import cloudpickle as pickle

"""
# data created create_data.py script
# analyzed in Drw_per_no_tape.py

COMB_002_ = using custom created ezTao data; 24 min on single core (but running same data 100 times!)
COMB_003_ = using custom created ezTao data; running both combined kernels and drw
COMB_004_ = using custom created ezTao data; running both combined kernels and drw; done on LSST ligtcurve
            generated from a random position in redback
"""

# from tape.ensemble import Ensemble
# from tape.utils import ColumnMapper

JaxPeriodDrwFit_instance = JaxPeriodDrwFit.JaxPeriodDrwFit()
JaxPeriodDrwFit_instance_drw = JaxPeriodDrwFit.JaxPeriodDrwFit()

if __name__ == '__main__':
    print('Hello World!')

    """
    Thus each file COMB_xxx_yyy_zzz.npy  corresponds to different xxx tau value,
    yyy sfinf value,  zzz DRW realization.
    Each is a dictionary of 100 light curves to which a different sinusoidal
    signal with different value of amplitude, period  has been added.
    All light curves here are "ideal", i.e. no noise,  evaluated at times t_true.txt
    """

    for yyy in tqdm(range(0, 1)):
        # for zzz in range(0, 5):
        for zzz in range(0, 1):
            formatted_yyy = f'{yyy:03}'
            formatted_zzz = f'{yyy:03}'
            # t_true = np.loadtxt('/epyc/users/suberlak/2023_DRW_SINUSOID_COMBINE/data/t_true.txt')
            # data_all = \
            #    np.load('/epyc/users/suberlak/2023_DRW_SINUSOID_COMBINE/data/COMB_001_'
            #            + formatted_yyy + '_' + formatted_zzz + '.npy', allow_pickle=True)

            # generated in create_data script, to avoid epyc failure
            t_multi = np.load('/astro/users/ncaplar/data/t_multi_LSST.npy')
            y_multi = np.load('/astro/users/ncaplar/data/y_multi_LSST.npy')
            yerr_multi = np.load('/astro/users/ncaplar/data/yerr_multi_LSST.npy')

            id, t, y, yerr, filter = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

            # array in which we will save the results
            res_s_par_combo = np.zeros((100, 5))
            res_s_par_combo_drw = np.zeros((100, 3))

            for i in range(100):
                # data = data_all[()].get(i)

                # get time for a single lightcurve
                t_true = t_multi[i]

                downsample_int = np.sort(np.random.choice(np.arange(len(t_true)), 100))
                t_single = t_true[downsample_int]

                # id = np.append(id, np.full(len(downsample_int), i))
                # filter_single = np.full(len(t_single), 'r')
                # t = np.append(t, t_single)
                # filter = np.append(filter, filter_single)

                y_err_single = np.full(len(t_single), 0.001)
                # yerr = np.append(yerr, np.full(len(t_single), 0.001))

                # y_pre = data['y_tot'][downsample_int]
                y_pre = y_multi[i][downsample_int]

                noise = np.random.normal(0, y_err_single)
                # y = np.append(y, y_pre + noise)
                y_single = y_pre + noise

                # t1 = time.time()
                test_single_lc_res = JaxPeriodDrwFit_instance.optimize_map(t_single, y_single, y_err_single,
                                                                           n_init=100)
                test_single_lc_res_drw = JaxPeriodDrwFit_instance_drw.optimize_map_drw(t_single,
                                                                                       y_single,
                                                                                       y_err_single,
                                                                                       n_init=100)
                # Put here the analysis without the period component
                # Put here the analysis with only the period component
                # t2 = time.time()
                # print(f'Execution time for single lc is {t2 - t1} sec')
                res_s_par_combo[i] = test_single_lc_res
                res_s_par_combo_drw[i] = test_single_lc_res_drw
     
    np.save('/astro/users/ncaplar/data/res_tests/COMB_004_'
            + formatted_yyy+'_'+formatted_zzz+'_run_0', res_s_par_combo)
    np.save('/astro/users/ncaplar/data/res_tests/COMB_004_'
            + formatted_yyy+'_'+formatted_zzz+'_drw_run_0', res_s_par_combo_drw)

    # print(f'Best result for loop index {i} is: {test_single_lc_res}')
