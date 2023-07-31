#!/usr/bin/env python3
import numpy as np
# import time
import JaxPeriodDrwFit
from tqdm import tqdm
# import dill as pickle
# import cloudpickle as pickle

# from tape.ensemble import Ensemble
# from tape.utils import ColumnMapper

JaxPeriodDrwFit_instance = JaxPeriodDrwFit.JaxPeriodDrwFit()

if __name__ == '__main__':
    print('Hello World!')

    """
    Thus each file COMB_xxx_yyy_zzz.npy  corresponds to different xxx tau value,
    yyy sfinf value,  zzz DRW realization.
    Each is a dictionary of 100 light curves to which a different sinusoidal
    signal with different value of amplitude, period  has been added.
    All light curves here are "ideal", i.e. no noise,  evaluated at times t_true.txt
    """

    for yyy in tqdm(range(0, 99)):
        for zzz in range(0, 5):
            formatted_yyy = f'{yyy:03}'
            formatted_zzz = f'{yyy:03}'
            t_true = np.loadtxt('/epyc/users/suberlak/2023_DRW_SINUSOID_COMBINE/data/t_true.txt')
            data_all = \
                np.load('/epyc/users/suberlak/2023_DRW_SINUSOID_COMBINE/data/COMB_001_'
                        + formatted_yyy + '_' + formatted_yyy + '.npy', allow_pickle=True)
            id, t, y, yerr, filter = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            res_s_par_combo = np.zeros((100, 5))

            for i in range(100):
                data = data_all[()].get(i)
                downsample_int = np.sort(np.random.choice(np.arange(len(t_true)), 100))
                t_single = t_true[downsample_int]
                # id = np.append(id, np.full(len(downsample_int), i))
                # filter_single = np.full(len(t_single), 'r')
                # t = np.append(t, t_single)
                # filter = np.append(filter, filter_single)

                y_err_single = np.full(len(t_single), 0.001)
                # yerr = np.append(yerr, np.full(len(t_single), 0.001))

                y_pre = data['y_tot'][downsample_int]
                noise = np.random.normal(0, y_err_single)
                # y = np.append(y, y_pre + noise)
                y_single = y_pre + noise

                # t1 = time.time()
                test_single_lc_res = JaxPeriodDrwFit_instance.optimize_map(t_single, y_single, y_err_single,
                                                                           n_init=100)
                # t2 = time.time()
                # print(f'Execution time for single lc is {t2 - t1} sec')
                res_s_par_combo[i] = test_single_lc_res
                np.save('/astro/users/ncaplar/data/res_tests/COMB_001_'
                        + formatted_yyy+'_'+formatted_zzz+'_run_0', res_s_par_combo)

    # print(f'Best result for loop index {i} is: {test_single_lc_res}')
