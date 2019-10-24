import numpy as np
from scipy import sparse

import matplotlib.pyplot as plt

from computational_bounds import generate_lower_bounds, complex_to_real, real_to_complex_design

# Test in the real case
n = 100

weights = np.ones(n)
A = sparse.diags([np.ones(n-1), -2 * np.ones(n), np.ones(n-1)], [-1, 0, 1]).tocsc()
b = np.zeros(n)
z_hat = np.zeros(n)
z_hat[n//4:(3*n)//4] = 1
t_max = np.ones(n)

obj_value, nu_value, init_design, init_field = generate_lower_bounds(z_hat, weights, A, b, t_max)
plt.figure()
plt.plot(np.linspace(0, 1, n), init_design)
plt.show()

# Simple test in the imaginary case
z_hat_t, weights_t, A_t, b_t, t_max_t, idx_constrained_t = complex_to_real(z_hat, weights, A, b, t_max)

obj_value_t, nu_value_t, init_design_t, init_field_t = generate_lower_bounds(z_hat_t, weights_t, A_t, b_t, t_max_t, idx_constrained=idx_constrained_t)
assert np.all(init_design_t[:n] == init_design_t[n:])
assert np.isclose(obj_value, obj_value_t)

print("Nice.")