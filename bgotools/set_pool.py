
'''
process the input data (pd.DataFrame)
set the training data pool and the design data pool for BGO. 
'''
import numpy as np 
import pandas as pd 


class set_pool:

    def __init__(self, 
                data,
                train_index,
                design_index,
                selected_corrs,
                Y_name,
                ternary = False):

        # data should include comp, corr(i)
        self.data = data
        # a list of indices of configurations that will be used to train the BG regression
        self.train_index = train_index
        # a list of indices of configurations that will be predicted by the trained BG model
        self.design_index = design_index
        # a list of 0 or 1 to turn off/on the correlation functions while training the BG model
        # e.g. np.ones(len(corrs)) to select all the correlation functions
        self.selected_corrs = selected_corrs
        self.ternary_check = ternary
        corrs_idx = np.array(range(len(self.selected_corrs)))
        corrs_boollist = list(map(bool,selected_corrs))
        selected_corrs_idx = corrs_idx[corrs_boollist]
        indices = [i for i, s in enumerate(data.columns) if 'corr' in s]
        self.corrs_all = data[data.columns[indices]]
        self.corrs_trunct = self.corrs_all[self.corrs_all.columns[selected_corrs_idx]]

        self.train_X = np.array(self.corrs_trunct.iloc[self.train_index])
        self.train_Y = np.array(self.data.iloc[self.train_index][Y_name])
        self.train_Y_nd = np.array([[Y] for Y in self.train_Y])
        self.design_X = np.array(self.corrs_trunct.iloc[self.design_index])

        self.space = []
        for corr_i in self.corrs_all.columns[selected_corrs_idx]: 
            space_i = {}
            space_i['name'] = corr_i
            space_i['type'] = 'discrete'
            space_i['domain'] = (0,1)
            self.space.append(space_i)
        if ternary: 

            self.train_comp = data.iloc[train_index][['comp(a)','comp(b)']]
            self.design_comp = data.iloc[design_index][['comp(a)','comp(b)']]

        else: 

            self.train_comp = data.iloc[train_index]['comp(a)']
            self.design_comp = data.iloc[design_index]['comp(a)']
