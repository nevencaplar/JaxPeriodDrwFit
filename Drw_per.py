#!/usr/bin/env python3
import numpy as np
import JaxPeriodDrwFit
import jax.numpy as jnp
print('Hello World!')

JaxPeriodDrwFit_instance = JaxPeriodDrwFit.JaxPeriodDrwFit()
# single_lc_res = JaxPeriodDrwFit_instance.optimize_map(1, t, y, yerr)
t = jnp.array([0, 1, 2, 3])
y = jnp.array([0, 1, 67, 3])
yerr = jnp.array([0, 1, 2, 89])

theta_init = [np.log10(1.0), np.log10(4.3), np.log10(100), np.log10(0.25)]

test_buildgp = JaxPeriodDrwFit_instance.build_gp(theta_init, t, y, yerr)
test_neg_log_likelihood =\
    JaxPeriodDrwFit_instance.neg_log_likelihood(theta_init, t, y, yerr)
test_single_lc_res = JaxPeriodDrwFit_instance.optimize_map(1, t, y, yerr)
print(test_neg_log_likelihood)
print(test_single_lc_res)
