import numpy as np
import scipy.stats as stats
from sympy import Symbol
from scipy.spatial import Delaunay, ConvexHull
from sympy import Matrix,Array
from sympy import MutableDenseNDimArray as MArray

class EI_below_hull:

    def __init__(self,
                design_X,
                design_comp,
                model,
                hull_function,
                xi=0.0,
                mode='min'
                ):
        """
        Compute the Expected Improvement at (X_i, comp_i). 
        
        design_X: Compositions of the training data set
        
        design_comp: Compositions of the design data set
        
        model: BG.regression
        
        mode: minimize or maximize the current Y profile, default is min

        xi: parameter for EI, read Brochu's paper for details, a safe value of 0.0 is good
        """
        self.design_X = design_X
        self.design_comp = design_comp
        self.model = model
        self.hull_function = hull_function
        self.xi = xi
        self.mode = mode 

    def EI(self):
        
        m_s, v_s = self.model.predict(self.design_X)[:2] 
        m_s = m_s.flatten() # mean 
        v_s = v_s.flatten() # variance
        # fmin: convex hull
        fmins = self.hull_function(self.design_comp).flatten()
        self.predictive_mean = m_s
        self.predictive_variance = v_s
        # self.design_X, m_s, v_s, design_comp, and fmins are for the same configurations 
        # variance too small will not be important
        if isinstance(v_s, np.ndarray):
            v_s[v_s<1e-10] = 1e-10
        elif v_s< 1e-10:
            v_s = 1e-10

        # GPy's model.predict returns the variance; EI needs the standard deviation
        s_s = np.sqrt(v_s)
        if self.mode == 'min':
            # find the function minimum.
            u = (fmins - m_s - self.xi) / s_s
        elif self.mode == 'max':
            # find the function maximum.
            u = (m_s - fmins - self.xi) / s_s
        else:
            print('I do not know what to do with mode %s' %self.mode)
        self.ei = s_s * (u * stats.norm.cdf(u) + stats.norm.pdf(u))
        
        return (self.ei)

class EI_hull_area:

    def __init__(self,
                pool,
                model,
                known_hull=None,
                known_hull_area=0,
                budget=5,
                xi=0.0,full_cov=False
                ):
        """
        Compute the Expected Improvement given the hull vertices. 
        
        design_X: Compositions of the training data set
        
        design_comp: Compositions of the design data set
        
        model: BG.regression

        pool: bgotools.set_pool.set_pool object 

        For C components, pool.design_comp contains C-1 independent fractions.
        Use my_nd_hull_funcs for a multicomponent known_hull; Area denotes volume.
        
        budget: number of configuration to select for next sets of exp. 

        xi: parameter for EI, read Brochu's paper for details, a safe value of 0.0 is good
        """
        self.pool = pool 
        self.design_X = self.pool.design_X
        self.design_comp = np.asarray(self.pool.design_comp) # one or more independent fractions
        self.design_positions = {index: row for row, index in enumerate(self.pool.design_index)}
        if len(self.design_positions) != len(self.design_X):
            raise ValueError('design_index must uniquely identify every design row')
        if not set(self.pool.train_index).issubset(self.design_positions):
            raise ValueError('The design pool must include the training configurations')
        self.model = model #BGO model object
        self.xi = xi
        self.m = budget 
        self.full_cov = full_cov # request covariance for each candidate subset
        # provide known_hull_func or known_hull_area
        if known_hull is not None:
            self.known_hull = known_hull
            self.known_hull.get_bottom_hull() # bgotools.my_hull_funcs.my_hull_funcs object
            self.known_hull.shoelace_area()
            self.known_hull_area,var_area = self.known_hull.Area,self.known_hull.nu_Area
        else:
            self.known_hull_area = known_hull_area

        from bgotools.my_hull_funcs import my_hull_funcs, my_nd_hull_funcs
        from itertools import combinations
        self.my_hull_funcs = my_hull_funcs if self.design_comp.ndim == 1 else my_nd_hull_funcs
        predictive_mean, predictive_variance = self.model.predict(self.design_X)[:2] 

        self.predictive_mean = predictive_mean.flatten()
        self.predictive_variance = predictive_variance
        # self.design_X, m_s, v_s, design_comp, and fmins are for the same configurations 
        if self.design_comp.ndim == 1:
            self.predicted_hull = self.my_hull_funcs(self.design_comp,self.predictive_mean,
                                            nu_Y=self.predictive_variance)
        else:
            self.predicted_hull = self.my_hull_funcs(
                np.column_stack((self.design_comp,self.predictive_mean)), E_nu=self.predictive_variance)
        self.predicted_hull.get_bottom_hull()
        self.predicted_hull.shoelace_area()
        # remove configurations which have been calculated 
        self.predicted_hull_configs = []
        for index in (self.predicted_hull.hull.vertices if self.design_comp.ndim == 1
                      else self.predicted_hull.bt_hull_index_all):
            index = list(self.pool.design_index)[index]
            if index not in self.pool.train_index:
                self.predicted_hull_configs.append(index)
        # all combinations of the predicted_hull_configs list up to a length of budget
        res = []
        if self.m <= len(self.predicted_hull_configs):
            
            for l in range(self.m+1):
                c = [list(i) for i in combinations(self.predicted_hull_configs, l)]
                res.extend(c)
        else:
            
            for l in range(len(self.predicted_hull_configs)+1):
                c = [list(i) for i in combinations(self.predicted_hull_configs, l)]
                res.extend(c)
        self.config_combinations = [ele for ele in res if ele != []]

    def EI(self):
        areas = []
        areas_var = []
        sub_hull_save = [] 
        for config_subset in self.config_combinations:
            # construct new hull using the subsets 
            new_configs = list(self.pool.train_index) + config_subset
            sub_index = [self.design_positions[index] for index in new_configs]
            sub_design_X = self.pool.design_X[sub_index]
            sub_design_comp = self.design_comp[sub_index]
            if self.full_cov:
                sub_mean, sub_variance = self.model.predict(sub_design_X,full_cov=True)[:2]
                if np.shape(sub_variance) != (len(sub_index),len(sub_index)):
                    raise ValueError('full_cov=True must return a square subset covariance matrix')
            else:
                sub_mean = self.predictive_mean[sub_index]
                sub_variance = self.predictive_variance[sub_index]
            sub_mean = sub_mean.flatten()
            
            # self.design_X, m_s, v_s, design_comp, and fmins are for the same configurations 
            if self.design_comp.ndim == 1:
                sub_hull = self.my_hull_funcs(sub_design_comp,sub_mean,nu_Y=sub_variance)
            else:
                sub_hull = self.my_hull_funcs(
                    np.column_stack((sub_design_comp,sub_mean)), E_nu=sub_variance)
            sub_hull_save.append(sub_hull)
            sub_hull.get_bottom_hull()
            sub_hull.shoelace_area()
            areas.append(sub_hull.Area) 
            # Binary nu_Area is variance; multicomponent nu_Area is already std.
            areas_var.append(float(sub_hull.nu_Area if self.design_comp.ndim == 1 else sub_hull.var_Area))

        self.all_sub_hulls = sub_hull_save
        self.areas = np.array(areas)
        self.areas_var = np.array(areas_var)
        # mean and variance of the predicted hull 
        m_s = self.areas
        v_s = self.areas_var
        # to maximize the area:
        # find the function maximum.
        u = np.divide(m_s - self.known_hull_area - self.xi, np.sqrt(v_s),
                      out=np.zeros_like(m_s), where=v_s > 0)
    
        self.ei = np.sqrt(v_s) * (u * stats.norm.cdf(u) + stats.norm.pdf(u))
        self.ei[v_s == 0] = np.maximum(m_s - self.known_hull_area - self.xi, 0)[v_s == 0]
        
        #return (self.ei,self.config_subset)
        
class EI_global_min:

    def __init__(self,
                design_X,
                design_comp,
                model,
                hull_function,
                xi=0.0,
                mode='min'
                ):
        """
        Compute the Expected Improvement at (X_i, comp_i). 
        
        design_X: Compositions of the training data set
        
        design_comp: Compositions of the design data set
        
        model: BG.regression
        
        mode: minimize or maximize the current Y profile, default is min

        xi: parameter for EI, read Brochu's paper for details, a safe value of 0.01 is good
        """
        self.design_X = design_X
        self.design_comp = design_comp
        self.model = model
        self.hull_function = hull_function
        self.xi = xi
        self.mode = mode 

    def EI(self):
        
        m_s, v_s = self.model.predict(self.design_X)[:2] 
        m_s = m_s.flatten() # mean 
        v_s = v_s.flatten() # variance
        # fmin: convex hull
        fmins = self.hull_function(self.design_comp).flatten()
        self.predictive_mean = m_s
        self.predictive_variance = v_s
        # self.design_X, m_s, v_s, design_comp, and fmins are for the same configurations 
        # variance too small will not be important
        if isinstance(v_s, np.ndarray):
            v_s[v_s<1e-10] = 1e-10
        elif v_s< 1e-10:
            v_s = 1e-10

        # GPy's model.predict returns the variance; EI needs the standard deviation
        s_s = np.sqrt(v_s)
        if self.mode == 'min':
            # find the function minimum.
            u = (fmins - m_s - self.xi) / s_s
        elif self.mode == 'max':
            # find the function maximum.
            u = (m_s - fmins - self.xi) / s_s
        else:
            print('I do not know what to do with mode %s' %self.mode)
        self.ei = s_s * (u * stats.norm.cdf(u) + stats.norm.pdf(u))
        
        return (self.ei)
