from sg_lib.grid.grid import *
from sg_lib.algebraic.multiindex import *

from config.config import *
    
if __name__ == '__main__':
	
	## objects setup
	Grid_obj 				= Grid(dim, level, level_to_nodes, left_bounds, right_bounds, weights)	
	Multiindex_obj 			= Multiindex(dim)

	## sparse grid multi-index set 
	multiindex_set = Multiindex_obj.get_std_total_degree_mindex(level)

	## first, we get the grid points in [0, 1]^dim
	std_sg_points = Grid_obj.get_std_sg_surplus_points(multiindex_set)

	print("\033[1m no points for dim = {} and level = {} is n = {}\033[0m".format(dim, level, std_sg_points.shape[0]))

	## we then map the grid points to our domain of interest
	mapped_sg_points = Grid_obj.map_std_sg_surplus_points(std_sg_points, left_stoch_boundary, right_stoch_boundary)

	print("\033[1m mapped sparse grid points \033[0m")
	print(mapped_sg_points)