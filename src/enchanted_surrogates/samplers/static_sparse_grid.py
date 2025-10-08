import warnings
import os
import pysgpp
# from scipy.stats import sobol_indices, uniform, entropy
import pandas as pd
import pickle
from enchanted_surrogates.samplers.base_sampler import Sampler
import numpy as np
from scipy.stats import sobol_indices, uniform, entropy



class StaticSparseGrid(Sampler):
    def __init__(self, bounds, parameters, do_brute_force_sobol_indicies = False, brute_force_sobol_indicies_num_samples=1e6, budget=np.inf, test_dir = None, *args, **kwargs):
        self.do_write_batch_info = kwargs.get('do_write_batch_info', True)
        self.test_dir = test_dir
        self.bounds = np.array(bounds)
        self.parameters = parameters
        self.initial_level = kwargs.get('initial_level', 2)
        self.final_level = kwargs.get('final_level', 3)
        self.level = self.initial_level
        self.base_run_dir = None
        self.budget = budget
        
        # if type(parameters[0]) == type([]):
        #     print('CONVERTING LIST TO TUPLE')
        #     self.parameters = [tuple(pa) for pa in parameters]
        #     print(self.parameters, type(self.parameters[0]))
        
        self.do_brute_force_sobol_indicies = do_brute_force_sobol_indicies
        if self.do_brute_force_sobol_indicies:
            # make uniform distributions for each input, this is used later to calculate sobel indicies for UQ
            self.dists = []
            for b in self.bounds:
                assert b[1] > b[0]
                self.dists.append(uniform(loc=b[0], scale=b[1]-b[0]))
            self.brute_force_sobol_indicies_num_samples = float(brute_force_sobol_indicies_num_samples)
        
        self.dim=len(parameters)
        
        self.do_surplus_based=True
                
        ### The test sampler is used to fill the space and run the surrogate model so that we can calcualte the integral and sobel indicies to brute_check for convergence in UQ situations.
        brute_check_sampler_kwargs = kwargs.get('brute_check_sampler')
        if type(brute_check_sampler_kwargs) != type(None):
            self.do_brute_check=True
            brute_check_sampler_kwargs['parameters'] = self.parameters
            brute_check_sampler_kwargs['bounds'] = self.bounds
            self.brute_check_sampler = getattr(importlib.import_module(f"samplers.{brute_check_sampler_kwargs['type']}"),brute_check_sampler_kwargs['type'])(**brute_check_sampler_kwargs) 
            self.old_brute_check_predictions = None
        else:
            self.do_brute_check=False
        
        self.batch_number = 0
        self.num_samples_by_batch = []        
        
        self.train = {}
        self.bounds_points = {}
        self.parent_points = {}
        
        self.grid_increase = None    
        self.all_batch_info = []
    
    def point_transform_box2unit(self, box_point):
        # min max normalisation
        unit_point = tuple()
        for i, bco in enumerate(box_point):
            min = self.bounds[i][0]
            max = self.bounds[i][1]
            assert max > min
            uco = (bco - min) / (max - min)
            unit_point = unit_point + (uco,)
        return unit_point

    def point_transform_unit2box(self, unit_point):
        # min max normalisation
        box_point = tuple()
        for i, uco in enumerate(unit_point):
            min = self.bounds[i][0]
            max = self.bounds[i][1]
            assert max > min
            bco = uco*(max - min) + min
            box_point = box_point + (bco,)
        return box_point
    
    def points_transform_unit2box(self, unit_points):
        unit_points = np.array(unit_points)
        return unit_points * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
    
    def points_transform_box2unit(self, box_points):
        return (box_points - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    def get_initial_parameters(self):
        # returns the initial set of parameters for initial trianing of surrogate model
        # self.grid = pysgpp.Grid.createPolyBoundaryGrid(self.dim, self.poly_basis_degree)
        print('INITIAL LEVEL:',self.initial_level)
        self.grid = pysgpp.Grid.createModLinearGrid(self.dim)
        self.gridStorage = self.grid.getStorage()
        self.gridGen = self.grid.getGenerator()
        self.gridGen.regular(self.initial_level)
        print('INITIAL GRID SIZE:', self.gridStorage.getSize())
        # array containing labels in order they come from;- for i in range(gridStorage.getSize()): gp = gridStorage.getPoint(i)
        self.alpha = pysgpp.DataVector(self.gridStorage.getSize())
        
        batch_samples = []
        boundary_counter = 0
        print('initial samples',self.gridStorage.getSize())
        for i in range(self.gridStorage.getSize()):
            gp = self.gridStorage.getPoint(i)
            unit_point = tuple()
            for j in range(self.dim):
                unit_point = unit_point + (gp.getStandardCoordinate(j),)
                
            if 1.0 in unit_point or 0.0 in unit_point:
                pass
                boundary_counter += 1
                
            box_point = self.point_transform_unit2box(unit_point)
        
            param_dict = {}
            for j, param in enumerate(self.parameters):
                param_dict[param]=box_point[j]
        
            batch_samples.append(param_dict)
        
        print('boundary counter',boundary_counter)
        self.num_samples_by_batch.append(len(batch_samples))    
        self.batch_number += 1
        self.submitted += len(batch_samples)
        return batch_samples
        
    def get_next_samples(
        self,
        base_run_dir:str=None,
        *args,
        **kwargs) -> list[dict[str, float]]:
        # returns a list of parameters that are the next batch to be labeled, based on the training samples, models used and selection criteria.
        """
        args
        train, dict: A dictionary where the keys are a tuple of inputs (x0,x1,x2) (*in the order of the origional parameters) and the value is a label
        """
        #
        
        if self.batch_number == 0:
            return self.get_initial_parameters()
        
        else:
            if not self.base_run_dir:
                raise RuntimeError('base_run_dir IS NOT SET IN SAMPLER. THIS IS PASSED IN THE CONFIG FOR THE EXECUTOR. THE EXECUTOR MUST THEN PASS IT TO THE SAMPLER SO IT CAN GRAB DATA FOR TRAINING. ENSURE THE EXECUTOR HAS THIS LINE IN start_runs --> self.sampler.base_run_dir = self.base_run_dir ')
            previous_batch_dir = os.path.join(self.base_run_dir, f'batch_{self.batch_number-1}')
            new_data_df = pd.read_csv(os.path.join(previous_batch_dir, f'enchanted_dataset_batch_{self.batch_number-1}.csv'))
            output_col = [col for col in new_data_df.columns if 'output' in col]
            if len(output_col) > 1:
                raise RuntimeError('StaticSparseGrid SAMPLER REQUIRES EXACTLY ONE OUTPUT VARIABLE. THE single_code_run IN THE RUNNER SHOULD RETURN A DICTIONARY OF OUTPUTS WHERE ONLY ONE HAS output IN THE KEY, eg \{growthrate_output\:5\}. THIS IS THE ONE THAT WILL BE USED FOR ACTIVE LEARNING PUTPOSES')
            train_df = new_data_df[self.parameters + output_col]
            new_train = {
                tuple(row[col] for col in self.parameters): float(row[output_col].iloc[0])
                for _, row in train_df.iterrows()
            }

            self.train.update(new_train)
            
            for i in range(self.gridStorage.getSize()):
                gp = self.gridStorage.getPoint(i)
                unit_point = ()
                for j in range(self.dim):
                    unit_point = unit_point + (gp.getStandardCoordinate(j),)
                box_point = self.point_transform_unit2box(unit_point)

                if box_point not in self.train:
                    raise RuntimeError('BOX POINT NOT IN TRAIN')
                
                if box_point not in self.train.keys() and box_point not in self.bounds_points and box_point not in self.parent_points:
                    if self.infer_bounds and not gp.isInnerPoint() or self.infer_parents and not gp.isLeaf():
                        # If it is both then default to boundary point
                        if self.infer_bounds and not gp.isInnerPoint():
                            print('appending bounds points')
                            self.bounds_points[box_point] = np.mean(list(self.train.values()))#self.surrogate_predict([box_point])[0]
                        # TODO: infer parents WOn't work as surrogate predict has no alpha value, put alpha set up into surrogate predict
                        # elif self.infer_parents and not gp.isLeaf():
                        #     self.parent_points[box_point] = self.surrogate_predict([box_point])[0]
                        #     print('appending parent points')
                            
                if box_point in self.train.keys():
                    self.alpha[i] = self.train[box_point]
                elif box_point in self.bounds_points.keys():
                    self.alpha[i] = self.bounds_points[box_point]
                elif box_point in self.parent_points.keys():
                    self.alpha[i] = self.parent_points[box_point] 
            
            # This changes alpha from the training point values into the surpluses
            pysgpp.createOperationHierarchisation(self.grid).doHierarchisation(self.alpha)        
            if self.do_write_batch_info:
                # Now the surrogate is trained we can write batch info
                print('WRITING batch INFO. LEVEL:',self.level)
                self.write_batch_info(previous_batch_dir)
            else:
                self.save_grid(previous_batch_dir)
            
            if self.level == self.final_level:
                # self.budget = 0 # this will stop the executor from asking for more samples because base_sampler.has_budget() will return false 
                # using return None to stop sampling as I need sampler to request samples for one batch more than the config allows so that I can write the batch info on the last batch
                self.merge_batch_info()
                return None # this needs to be allowed by the executor
            grid_size_before_refinement = self.gridStorage.getSize()

            self.level += 1
            self.grid = pysgpp.Grid.createModLinearGrid(self.dim)
            self.gridStorage = self.grid.getStorage()
           
            self.gridGen = self.grid.getGenerator()
            self.gridGen.regular(self.level)
                    
            grid_size_after_refinement = self.gridStorage.getSize()
            print('level:',self.level,'new grid size:', grid_size_after_refinement)
            self.grid_increase = grid_size_after_refinement - grid_size_before_refinement
            # print("refinement step {}, new grid size: {}".format(refnum+1, gridStorage.getSize()))  
            self.alpha.resizeZero(self.gridStorage.getSize())
            
            batch_samples = []
            for i in range(self.gridStorage.getSize()):
                gp = self.gridStorage.getPoint(i)
                unit_point = ()
                for j in range(self.dim):
                    unit_point = unit_point + (gp.getStandardCoordinate(j),)
                
                box_point = self.point_transform_unit2box(unit_point)

                if box_point not in self.train.keys() and box_point not in self.bounds_points and box_point not in self.parent_points:
                    print('NEW POINT',box_point, unit_point)
                    print('IS BOUNDARY', not gp.isInnerPoint())
                    print('is parent', not gp.isLeaf())                
                    param_dict = {}
                    for j, param in enumerate(self.parameters):
                        param_dict[param]=box_point[j]                    
                    print('adding to batch samples')
                    batch_samples.append(param_dict)
            
            self.num_samples_by_batch.append(len(self.train)+len(batch_samples))
            print('len batch_samples',len(batch_samples))
            # batch_run_dirs = self.make_run_dirs(batch_dir, len(batch_samples))  
            self.batch_number += 1
            self.submitted += len(batch_samples)
            return batch_samples
    
    def is_boundary_point(self, unit_point):
        if 0 in unit_point or 1 in unit_point:
            return True
        else:
            return False
    
    # currently only works with point in the unit cube.     
    def surrogate_predict(self, box_points):
        unit_points = self.points_transform_box2unit(np.array(box_points))
        unit_points_dm = pysgpp.DataMatrix(unit_points)
        opEval = pysgpp.createOperationMultipleEval(self.grid, unit_points_dm)
        results = pysgpp.DataVector(len(unit_points))
        opEval.eval(self.alpha, results)
        ans = np.array(results.array()) # sometimes this line breaks python. 
        # ans = np.array([results.get(i) for i in range(len(positions))])
        return ans
    
    def quadrature_integral(self):
        op_quad = pysgpp.createOperationQuadrature(self.grid)
        unit_integral = op_quad.doQuadrature(self.alpha)
        return unit_integral
    
    def quadrature_function_integral(self, function):
        # this would be useful for calculating expectation values, variances and sobol indicies, even with non uniform probability distirbutions
        # The function can be a data look up table that is multiplied by a prob density for expectation or squared for variance etc
        alpha = pysgpp.DataVector(self.gridStorage.getSize())
        for i in range(self.gridStorage.getSize()):
            gp = self.gridStorage.getPoint(i)
            unit_point = ()
            for j in range(self.dim):
                unit_point = unit_point + (gp.getStandardCoordinate(j),)
            box_point = self.point_transform_unit2box(unit_point)
            alpha[i] = function(box_point)
        # compute surpluses
        pysgpp.createOperationHierarchisation(self.grid).doHierarchisation(alpha)
        op_quad = pysgpp.createOperationQuadrature(self.grid)
        unit_integral = op_quad.doQuadrature(alpha)
        return unit_integral
    
    def lookup_function(self, box_point):
        box_point = tuple(box_point)
        if box_point in self.train.keys():
            return self.train[box_point]
        elif box_point in self.bounds_points.keys():
            return self.bounds_points[box_point]
        elif box_point in self.parent_points.keys():
            return self.parent_points[box_point]
        else: 
            self.point_not_in_train_count += 1
            # get closeest point,
            # all_points = list(self.train.keys()) + list(self.parent_points.keys()) + list(self.bounds_points.keys())
            # all_points_dict = self.train | self.parent_points | self.bounds_points
            # closest_point = all_points[np.argmin(np.sum(np.abs(np.array(all_points) - box_point)**2))]
            messages=[f'this point {box_point} was not in train, bounds_points or parent_points',
                      f'len train: {len(self.train)}',
                      f'grid size: {self.gridStorage.getSize()}',
                    #   f'closeest point: {closest_point}',
                      f'number of times this issue has occured: {self.point_not_in_train_count}', 
                      'returning simulation value for closest point',
                      f'train:{self.train}']
            message = '\n'.join(messages)
            # print(f'this point {box_point} was not in train, bounds_points or parent_points')
            raise KeyError(message)
            # # raise ValueError(f'this point {box_point} was not in train, bounds_points or parent_points')
            # return all_points_dict[closest_point]
    
    def quadrature_expectation(self):
        # assumes uniform input distributions
        return self.quadrature_integral()
    
    def quadrature_variance(self):
        # assumes uniform probability distirbutions for inputs
        # VAR(f(x)) = EXP(f(x)**2) - EXP(f(x))**2
        fxsquared = lambda box_point: self.lookup_function(box_point)**2
        EXPfxsquared = self.quadrature_function_integral(fxsquared)
        EXPfx = self.quadrature_expectation()
        VARfx = EXPfxsquared - EXPfx**2
        return VARfx
    
    def integral_approx(self, function_values):
        bounds_array = np.array(self.bounds).T
        space_volume = np.prod(bounds_array[1]-bounds_array[0])
        integral_estimate = np.mean(function_values) * space_volume
        return integral_estimate
    
    # Assumes the inputs used to get the function values were sampled from their uncertainty distributions.
    def expectation_approx(self, function_values, num_bins=200):
        # assumes uniform input distributions
        heights, edges = np.histogram(function_values, bins=num_bins)
        weights = heights/sum(heights)
        centers = (edges[:-1] + edges[1:]) / 2
        expectation = np.average(centers,weights=weights)
        return expectation
    
    def confidance_interval(self, function_values, percentage, num_bins=200):
        decimal = percentage/100
        heights, edges = np.histogram(function_values, bins=num_bins, density=True)
        width = edges[1]-edges[0]
        centers = (edges[:-1] + edges[1:]) / 2
        area = 0
        area_limit = (1-decimal)/2
        i = 0
        while area<area_limit:
            area += heights[i]*width
            i+=1
        lower_interval = centers[i]
        
        area = 0
        i = 0
        while area<area_limit:
            area += heights[-i]*width
            i+=1
        upper_interval = centers[-i]
        return lower_interval, upper_interval
        
    def double_sigma_approx(self, function_values):
        double_sigma_estimate = 2*(np.sqrt(self.expectation_approx(function_values**2) - self.expectation_approx(function_values)**2))
        return double_sigma_estimate
    
    def sobel_indicies_approx(self):
        func = lambda positions: self.surrogate_predict(positions.T)
        sobol = sobol_indices(func=func,  n=2**int(np.log2(self.brute_force_sobol_indicies_num_samples)), dists=self.dists)
        return sobol.first_order, sobol.total_order
    
    def relative_entropy(self, samples_p, samples_q, num_bins=200):
        """
        Calculate the relative entropy (Kullback-Leibler divergence) between two distributions.

        Parameters:
        samples_p (list or np.array): Samples from the first distribution.
        samples_q (list or np.array): Samples from the second distribution.

        Returns:
        float: The relative entropy between the two distributions.
        """
        if type(samples_p) == type(None) or type(samples_q) == type(None):
            return np.nan
        # Convert samples to numpy arrays
        samples_p = np.array(samples_p)
        samples_q = np.array(samples_q)
        # Calculate the probability density functions
        p_values, _ = np.histogram(samples_p, bins=num_bins, density=True)
        q_values, _ = np.histogram(samples_q, bins=num_bins, density=True)
        # Add a small value to avoid division by zero and log of zero
        p_values += 1e-10
        q_values += 1e-10
        # Calculate the relative entropy
        rel_entropy = entropy(p_values, q_values)
        return rel_entropy
    
    def write_batch_info(self, batch_dir, fname='batch_info.csv', save_grid=True): #Necessary
        if self.do_write_batch_info:
            print(f'+++ \n Write BATCH: {self.batch_number}')
            print('+++ \n NUM INNER LEAF POINTS', len(self.train))            
            print('+++ \n NUM PARENT POINTS', len(self.parent_points))
            print('+++ \n NUM bounds POINTS', len(self.bounds_points))
            # df = pd.DataFrame({"num_samples":[self.num_samples_by_batch[-1]], "num_parents":[len(self.parent_points)], "num_bounds":[len(self.bounds_points)], "quad_expectation":[self.quadrature_expectation()],"quad_double_sigma":[2*np.sqrt(self.quadrature_variance())]})
            quad_exp = self.quadrature_expectation()
            quad_std = 2*np.sqrt(self.quadrature_variance())
            df = pd.DataFrame({"num_samples":[len(self.train)], "num_parents":[len(self.parent_points)], "num_bounds":[len(self.bounds_points)], "mean":[quad_exp],"std":[quad_std]})
            if self.grid_increase != None:
                df['mean_recent_surplus'] = [np.mean( np.abs(np.array(self.alpha.array())[-self.grid_increase : ]) )]
            else:
                df['mean_recent_surplus'] = [np.mean( np.abs(np.array(self.alpha.array())) )]
            df['mean_surplus'] = [np.mean(np.abs(np.array(self.alpha.array())))]
            df['max_surplus'] = [np.max(np.abs(np.array(self.alpha.array())))]
            df['do_surplus_based'] = self.do_surplus_based 
            if self.test_dir != None:
                x_test, y_test = self.get_test_set(self.test_dir)
                y_pred = self.surrogate_predict(x_test)
                residuals = y_test - y_pred
                me = np.mean(np.abs(residuals))
                df["mean_error"]=[me]
                df["expectation_error"] = [np.abs(np.mean(y_test)-quad_exp)]
                df["std_error"] = [np.abs(np.sqrt(np.var(y_test))-quad_std)]
            if self.do_brute_check:
                print('DOING BRUTE CHECK')
                predictions = self.surrogate_predict(self.brute_check_sampler.samples)
                expectation = self.expectation_approx(predictions)
                double_sigma = self.double_sigma_approx(predictions)
                entropy_diff = self.relative_entropy(self.old_brute_check_predictions, predictions, num_bins=200)
                self.old_brute_check_predictions=predictions
                print('EXPECTATION', expectation)
                df["brute_expectation"]=[expectation]
                df["brute_double_sigma"]=[double_sigma]
                df["brute_entropy_diff"]=[entropy_diff]
                del predictions
                
            if self.do_brute_force_sobol_indicies:
                print('doing brute sobol indicies')
                sobol_first_order, sobol_total_order = self.sobel_indicies_approx()
                for d, sfo in enumerate(sobol_first_order):
                    df[f'brute_sobol_first_order_{d}'] = [sfo]
                for d, sto in enumerate(sobol_total_order):
                    df[f'brute_sobol_total_order_{d}']= [sto]
            df.to_csv(os.path.join(batch_dir,fname), header=True, index=False)
            self.all_batch_info.append(df)
            del df
        if save_grid:
            self.save_grid(batch_dir)
    
    def merge_batch_info(self):
        assert self.base_run_dir
        dfs = []
        for dir_ in os.listdir(self.base_run_dir):
            if 'batch_' in dir_:
                batch_dir = os.path.join(self.base_run_dir, dir_)
                dfi = pd.read_csv(os.path.join(batch_dir, 'batch_info.csv'))
                dfs.append(dfi)
        df = pd.concat(dfs)
        df.to_csv(os.path.join(self.base_run_dir,'batch_info.csv'), header=True, index=False)
        
    def save_grid(self, batch_dir):
        print('pickeling points')        
        with open(os.path.join(batch_dir,'train_points.pkl'), 'wb') as file:
            pickle.dump(self.train, file)
        with open(os.path.join(batch_dir,'infered_bound_points.pkl'), 'wb') as file:
            pickle.dump(self.bounds_points, file)
        with open(os.path.join(batch_dir,'infered_parent_points.pkl'), 'wb') as file:
            pickle.dump(self.parent_points, file)
        
        print('SAVING GRID')
        with open(os.path.join(batch_dir,'pysgpp_grid.txt'), 'w') as file:
            file.write(self.grid.serialize())
        print('SAVING SURPLUSES')
        self.alpha.toFile(os.path.join(batch_dir, 'surpluses.mat'))
        
    
    # def post_run(self, base_dir):
    #     df = pd.concat(self.all_batch_info, ignore_index=True)
    #     df.to_csv(os.path.join(base_dir, 'all_batch_info.csv'))
        
    #     fig = plt.figure()
    #     plt.plot(np.array(self.num_samples_by_batch), df['quad_expectation'].to_numpy().astype('float'))
    #     plt.fill_between(self.num_samples_by_batch, df['quad_expectation'].to_numpy().astype('float')-df['quad_double_sigma'].to_numpy().astype('float'),df['quad_expectation'].to_numpy().astype('float')+df['quad_double_sigma'].to_numpy().astype('float'), color='grey', label='2sigma')
    #     plt.ylabel('quad_expectation')
    #     plt.xlabel('Number of Parent Function Evaluations')
    #     plt.legend()
    #     fig.savefig(os.path.join(base_dir,'quad_expectation'), dpi=300)
        
    #     if self.do_brute_check:
    #         predictions = self.surrogate_predict(self.brute_check_sampler.samples)
    #         fig = plt.figure()
    #         bars = plt.hist(predictions, bins=200, density=True)
    #         heights, edges = bars[0], bars[1]
    #         plt.vlines([self.expectation_approx(predictions)],0, max(heights),color='red', label=r'Expectation and 95% confidance interval')
    #         lower_interval, upper_interval = self.confidance_interval(predictions,percentage=95)
    #         plt.hlines(0, lower_interval, upper_interval, color='red', linewidth=3)
    #         plt.xlabel('function output')
    #         plt.ylabel('probability density')
    #         plt.title(f'Output Distribution, {self.brute_check_sampler.num_samples} samples')
    #         plt.legend()
    #         fig.savefig(os.path.join(base_dir, 'output_distribution'), dpi=400)
    #         hist_data_heights = pd.DataFrame({'heights': heights})
    #         hist_data_heights.to_csv(os.path.join(base_dir, 'output_distribution_hist_data_heights.csv'), index=False)
    #         hist_data_edges = pd.DataFrame({'edges': edges})
    #         hist_data_edges.to_csv(os.path.join(base_dir, 'output_distribution_hist_data_edges.csv'), index=False)
            
    #         fig = plt.figure()
    #         plt.plot(np.array(self.num_samples_by_batch), df['brute_expectation'].to_numpy().astype('float'))
    #         plt.fill_between(self.num_samples_by_batch, df['brute_expectation'].to_numpy().astype('float')-df['brute_double_sigma'].to_numpy().astype('float'),df['brute_expectation'].to_numpy().astype('float')+df['brute_double_sigma'].to_numpy().astype('float'), color='grey', label='2sigma')
    #         plt.ylabel('brute_expectation')
    #         plt.xlabel('Number of Parent Function Evaluations')
    #         plt.legend()
    #         fig.savefig(os.path.join(base_dir,'approx_expectation'), dpi=300)
            
    #         fig = plt.figure()
    #         plt.plot(self.num_samples_by_batch, df['brute_entropy_diff'].to_numpy().astype('float'))
    #         plt.ylabel('Entropy Difference')
    #         plt.xlabel('Number of Parent Function Evaluations')
    #         fig.savefig(os.path.join(base_dir, 'approx_entropy_diff'),dpi=300)
            
    #     if self.do_brute_force_sobol_indicies:
    #         fig = plt.figure()
    #         for i in range(self.dim):
    #             plt.plot(self.num_samples_by_batch, df[f'brute_sobol_first_order_{i}'])
    #             plt.ylabel('First order sobol indicie')
    #             plt.xlabel('Number of Parent Function Evaluations')
    #             fig.savefig(os.path.join(base_dir, 'approx_sobol_first_order'),dpi=300)
            
    #         fig = plt.figure()
    #         for i in range(self.dim):
    #             plt.plot(self.num_samples_by_batch, df[f'brute_sobol_total_order_{i}'])
    #             plt.ylabel('Total order sobol indicie')
    #             plt.xlabel('Number of Parent Function Evaluations')
    #             fig.savefig(os.path.join(base_dir, 'approx_sobol_total_order'),dpi=300)
    
    def get_test_set(self, test_dir):
        print('RETRIVING TEST SET FROM', test_dir)        
        if os.path.exists(os.path.join(test_dir,'merged_runner_return.csv')):        
            df_test = pd.read_csv(os.path.join(test_dir,'merged_runner_return.csv'))
            print('got runner_return.csv')
        elif os.path.exists(os.path.join(test_dir, 'merged_runner_return.txt')):
            df_test = pd.read_csv(os.path.join(test_dir, 'merged_runner_return.txt'))
            print('got runner_return.txt')
        # elif os.path.exists(os.path.join(test_dir, 'runner_return.txt')):
        #     df_test = pd.read_csv(os.path.join(test_dir, 'runner_return.txt'))
        #     print('got runner_return.txt')    
        else:
            print('NO RUNNER RETURN FOUND, BEGINNIGN PARSING')
            finished_result = find_files(test_dir, 'GENE.finished')
            stopped_result = find_files(test_dir, 'stopped_by_monitor')
            result = finished_result + stopped_result
            run_dirs = [os.path.dirname(path) for path in result]
            if len(result) == 0:        
                raise FileNotFoundError('NO RUNNER RETURN PATH WAS FOUND IN:',test_dir,'\n ALSO THERE SEEM TO BE NO FINNISHED OR EARLY STOPPED GENE RUNS IN:',test_dir)
            else:
                outputs = []
                for i, run_dir in enumerate(run_dirs):
                    if i % 10 == 0:
                        print('NUMBER OF RUN_DIR PARSED:',i)
                    outputs.append(parse_run_dir(run_dir, parameters))
                with open(os.path.join(test_dir, 'merged_runner_return.txt'), 'w') as file:
                    lines = [runner_return_headder] + outputs
                    lines = [line+'\n' for line in lines]
                    file.writelines(lines)
                df_test = pd.read_csv(os.path.join(test_dir, 'merged_runner_return.txt'))
                
        test_x = np.array(df_test.iloc[:,0:-1].astype('float'))
        # print('debug l tx', len(test_x))
        test_y = np.array(df_test.iloc[:,-1].astype('float'))
        return test_x, test_y

          
    def check_UQ_convergence():
        None
    
    
    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None

# # import pysgpp library
# import pysgpp
# import inspect
# f = lambda x0, x1: mmg.evaluate([x0, x1])
# # f = lambda x0, x1: 16.0 * (x0-1)*x0 * (x1-1)*x1*x1

# poly_basis_degree = 3
# grid = pysgpp.Grid.createPolyBoundaryGrid(dim, poly_basis_degree)
# # grid = pysgpp.Grid.createLinearBoundaryGrid(dim)

# gridStorage = grid.getStorage()

# # create regular sparse grid, level 3
# initial_level = 3
# gridGen = grid.getGenerator()

# gridGen.regular(initial_level)

# print("number of initial grid points:    {}".format(gridStorage.getSize()))

# alpha = pysgpp.DataVector(gridStorage.getSize())
# print("length of alpha vector:           {}".format(alpha.getSize()))
# # Obtain function values and refine adaptively 5 times
# num_refinement_steps =100

# x0, x1, x0_leaf, x1_leaf = [], [], [], []

# #We don't want to run the function for every point so a wrapper function should brute_check to see if the point has been ran and if it has return that value
# def dummy_runner(f, samples, labeled_samples=None):
#     if type(labeled_samples) != type(None):
#         # print('LABELED SAMPLES',labeled_samples.items())
#         for k, v in samples.items():
#             if k in labeled_samples.keys():
#                 samples[k] = labeled_samples[k]
#             else:
#                 samples[k] = f(*k)
#     else:
#         for k, v in samples.items():
#             samples[k] = f(*k)
#     return samples
            
# labeled_samples=None
# samples = {}
# # I want to know if there are any samples that have never been a leaf
# was_leaf = []

# samples_s = []
# is_leaf_s = []
# for refnum in range(num_refinement_steps):
#     print('REFINEMENT STEP', refnum+1)
#     # make samples dict
#     for i in range(gridStorage.getSize()):
#         gp = gridStorage.getPoint(i)
#         x0 = gp.getStandardCoordinate(0)
#         x1 = gp.getStandardCoordinate(1)
#         samples[(x0,x1)] = None
#         if i > len(was_leaf):
#             # print('APPENDING')
#             was_leaf.append(False)
#     was_leaf.append(False)
    
#     # label samples
#     samples = dummy_runner(f, samples, labeled_samples)
#     samples_s.append(samples.copy())
#     labeled_samples = samples.copy()
    
#     # set function values in alpha
#     is_leaf = []
#     for i in range(gridStorage.getSize()):
#         gp = gridStorage.getPoint(i)
#         # print(dir(gp))
#         # break
#         x0 = gp.getStandardCoordinate(0)
#         x1 = gp.getStandardCoordinate(1)
#         # print('function value',f(x0,x1), type(f(x0,x1)))
#         # print('dict value',samples[(x0,x1)], type(samples[(x0,x1)]))
#         alpha[i] = samples[(x0,x1)]
#         if gp.isLeaf():
#             was_leaf[i] = True
#             is_leaf.append(True)
#         else:
#             is_leaf.append(False)     
#     is_leaf_s.append(is_leaf)
#     # break
#     pysgpp.createOperationHierarchisation(grid).doHierarchisation(alpha)
#     gridGen.refine(pysgpp.SurplusRefinementFunctor(alpha, 1))
#     print("refinement step {}, new grid size: {}".format(refnum+1, gridStorage.getSize()))    
#     alpha.resizeZero(gridStorage.getSize())
    



##!!!!!!!!!!!EVALUATING AN INTEGRAL USING ANALYTICALLY CALCULATED COEFFICIENTS THAT ASSUME THE HEIRACIAL BASIS FUNCTION IS AN ACCURATE INTERPOLATOR!!!!!!!!!!!

# import pysgpp
# from pysgpp import Grid, DataVector, DataMatrix

# # Define the dimensionality of the problem
# dim = 2

# # Create a piecewise linear grid
# grid = Grid.createLinearGrid(dim)
# grid_gen = grid.getGenerator()
# level = 3
# grid_gen.regular(level)

# # Get the grid points
# grid_storage = grid.getStorage()
# num_points = grid_storage.getSize()

# # Create a DataVector to store the function values at the grid points
# alpha = DataVector(num_points)

# # Define a function to integrate
# def f(x):
#     return x[0] * x[1]

# # Evaluate the function at the grid points
# for i in range(num_points):
#     gp = grid_storage.getPoint(i)
#     coords = [gp.getStandardCoordinate(d) for d in range(dim)]
#     alpha[i] = f(coords)

# # Create an operation to compute the quadrature weights
# op_quad = pysgpp.createOperationQuadrature(grid)

# # Compute the integral (quadrature)
# integral = op_quad.doQuadrature(alpha)

# # Print the result
# print("Integral:", integral)

# # To get the quadrature weights for each point, you can use the following:
# weights = DataVector(num_points)
# op_quad.getQuadratureWeights(weights)

# # Print the quadrature weights
# for i in range(num_points):
#     gp = grid_storage.getPoint(i)
#     coords = [gp.getStandardCoordinate(d) for d in range(dim)]
#     print(f"Point: {coords}, Weight: {weights[i]}")