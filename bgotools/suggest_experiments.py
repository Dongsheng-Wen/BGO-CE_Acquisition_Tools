import numpy as np 
import pandas as pd 

class suggest_next:
    # suggest the next experiments 

    # criteria:
    # 1. new configurations must not be inclused in the observed pool
    # 2. new configurations must not have the same composition
    # further cost function maybe applied 

    def __init__(self,data,ei,batch_size=8,ei_tol=0.001,
        already_exist_idx=[],ternary=False,cost_function=None):

        # given a list of ei
        self.ei = ei 
        # screen the ei for ei larger than max*0.01
        self.ei_trunct_idx = np.nonzero((self.ei)>=max(self.ei)*ei_tol)
        # screened ei list
        self.ei_trunct = self.ei[self.ei_trunct_idx]

        self.ternary = ternary
        # cost_function depending on some variables # on the way!
        self.cost_function = cost_function

        self.already_exist_idx = []
        self.next_exps = []
        self.batch_size = batch_size
        try:
            data_ei = pd.DataFrame(data={
                'comp(a)': data['comp(a)'],
                'comp(b)': data['comp(b)'],
                'ei': self.ei
                })

        except:
            data_ei = pd.DataFrame(data={
                'comp(a)': data['comp(a)'],
                'ei': self.ei
                })

        self.data_ei = data_ei 
        self.data_trunct = self.data_ei.iloc[self.ei_trunct_idx]

    def compute_batch(self,):

        self.next_exps = []
        self.repeated_comps = []
        sorted_data_ei = self.data_ei.iloc[self.ei_trunct_idx].sort_values(by = 'ei', ascending = False)

        for idx in sorted_data_ei.index:
            try:
                comp_i = ([sorted_data_ei[sorted_data_ei.index==idx]['comp(a)'].values[0],
                           sorted_data_ei[sorted_data_ei.index==idx]['comp(b)'].values[0]])
            except:
                comp_i = sorted_data_ei[sorted_data_ei.index==idx]['comp(a)'].values[0]
            if (comp_i not in self.repeated_comps) and (idx not in self.already_exist_idx):
                self.next_exps.append(idx)
                self.repeated_comps.append(comp_i)
                self.already_exist_idx.append(idx)
            if len(self.next_exps)>=self.batch_size:
                break

    def compute_batch_w_cost(self):
        print('on the way!')

