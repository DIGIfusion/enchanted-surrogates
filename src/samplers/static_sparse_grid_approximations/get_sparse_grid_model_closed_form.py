import numpy as np
import chaospy as cp
from itertools import product

from config.config import *

def eps(Q_pred, Q_ref):

	N = Q_ref.shape[0]

	assert Q_ref.shape[0] == Q_pred.shape[0]

	eps = np.sqrt( 1/N * np.sum( (Q_ref - Q_pred)**2/(Q_ref + Q_pred)**2 ) )

	return eps

def get_indices_reduced_basis(spectral_coeff, target_perc):

	coeff_sq 			= spectral_coeff[1:]**2
	ind_coeff_sq_sorted = np.argsort(coeff_sq)[::-1]
	cummulative_energy 	= np.cumsum(coeff_sq[ind_coeff_sq_sorted])/np.sum(coeff_sq[ind_coeff_sq_sorted])

	target_index 	= (np.abs(cummulative_energy - target_perc)).argmin()
	target_indices 	= ind_coeff_sq_sorted[:target_index] + 1

	target_indices = np.insert(target_indices, 0, 0, axis=0)

	return target_indices


def get_1D_poly(deg_1D, left_1D, right_1D, var_1D):

	distr 	= cp.Uniform(left_1D, right_1D)
	poly 	= cp.generate_expansion(deg_1D, distr, normed=True)


	target_monomial = poly(q0=var_1D)[deg_1D]

	return target_monomial

def get_ND_poly(deg_ND, left_ND, right_ND, var_ND):

	temp = get_1D_poly(deg_ND[0], left_ND[0], right_ND[0], var_ND[0])

	for i in range(1, deg_ND.shape[0]):
	 	temp = temp * get_1D_poly(deg_ND[i], left_ND[i], right_ND[i], var_ND[i])

	return temp

def get_reduced_model_in_closed_form(target_coeff, target_basis, left_bounds, right_bounds, var_ND):

	reduced_model = target_coeff[0] * get_ND_poly(target_basis[0, :], left_bounds, right_bounds, var_ND)

	for i in range(1, target_coeff.shape[0]):
	
		reduced_model = reduced_model + target_coeff[i] * get_ND_poly(target_basis[i, :], left_bounds, right_bounds, var_ND)

	return reduced_model

if __name__ == '__main__':

	var_ND  = cp.variable(dim)

	spectral_coeff 	= np.load('results/spectral_coeff.npy')
	spectral_basis 	= np.load('results/spectral_basis.npy')

	reduced_model_full = get_reduced_model_in_closed_form(spectral_coeff, spectral_basis, left_stoch_boundary, right_stoch_boundary, var_ND)