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

        if self.mode == 'min':
            # find the function minimum.
            u = (fmins - m_s - self.xi) / v_s
        elif self.mode == 'max':
            # find the function maximum.
            u = (m_s - fmins - self.xi) / v_s
        else:
            print('I do not know what to do with mode %s' %self.mode)
        self.ei = v_s * (u * stats.norm.cdf(u) + stats.norm.pdf(u))
        
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
        
        budget: number of configuration to select for next sets of exp. 

        xi: parameter for EI, read Brochu's paper for details, a safe value of 0.0 is good
        """
        self.pool = pool 
        self.design_X = self.pool.design_X
        self.design_comp = self.pool.design_comp #1d
        self.model = model #BGO model object
        self.xi = xi
        self.m = budget 
        self.full_cov = full_cov # tell whether to print the matrix
        # provide known_hull_func or known_hull_area
        try:
            self.known_hull = known_hull
            self.known_hull.get_bottom_hull() # bgotools.my_hull_funcs.my_hull_funcs object
            self.known_hull.shoelace_area()
            self.known_hull_area,var_area = self.known_hull.Area,self.known_hull.nu_Area
        except:
            self.known_hull_area = known_hull_area

        from bgotools.my_hull_funcs import my_hull_funcs
        from itertools import combinations
        self.my_hull_funcs = my_hull_funcs
        predictive_mean, predictive_variance = self.model.predict(self.design_X)[:2] 

        self.predictive_mean = predictive_mean.flatten()
        self.predictive_variance = predictive_variance
        # self.design_X, m_s, v_s, design_comp, and fmins are for the same configurations 
        self.predicted_hull = my_hull_funcs(self.design_comp,self.predictive_mean,
                                        nu_Y=self.predictive_variance)
        self.predicted_hull.get_bottom_hull()
        self.predicted_hull.shoelace_area()
        # remove configurations which have been calculated 
        self.predicted_hull_configs = []
        for index in self.predicted_hull.hull.vertices:
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
            sub_design_X = self.pool.design_X[new_configs]
            sub_design_comp = self.pool.design_comp[new_configs]
            #sub_mean, sub_variance = self.model.predict(sub_design_X,full_cov=self.full_cov)[:2] 
            sub_mean = self.predictive_mean[new_configs]
            sub_variance = self.predictive_variance[new_configs]
            sub_mean = sub_mean.flatten()
            
            # self.design_X, m_s, v_s, design_comp, and fmins are for the same configurations 
            sub_hull = self.my_hull_funcs(sub_design_comp,sub_mean,
                                            nu_Y=sub_variance)
            sub_hull_save.append(sub_hull)
            sub_hull.get_bottom_hull()
            sub_hull.shoelace_area()
            areas.append(sub_hull.Area) 
            areas_var.append(sub_hull.nu_Area)

        self.all_sub_hulls = sub_hull_save
        self.areas = np.array(areas)
        self.areas_var = np.array(areas_var)
        # mean and variance of the predicted hull 
        m_s = self.areas
        v_s = self.areas_var
        # to maximize the area:
        # find the function maximum.
        u = (m_s - self.known_hull_area - self.xi) / v_s
    
        self.ei = v_s * (u * stats.norm.cdf(u) + stats.norm.pdf(u))
        
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

        if self.mode == 'min':
            # find the function minimum.
            u = (fmins - m_s - self.xi) / v_s
        elif self.mode == 'max':
            # find the function maximum.
            u = (m_s - fmins - self.xi) / v_s
        else:
            print('I do not know what to do with mode %s' %self.mode)
        self.ei = v_s * (u * stats.norm.cdf(u) + stats.norm.pdf(u))
        
        return (self.ei)
