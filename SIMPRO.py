import pandas as pd 
import numpy as np
from scipy.stats import wasserstein_distance as wd
from scipy.stats import chisquare
from scipy import stats
import warnings
from tqdm import tqdm


class simpro:
    def __init__(self, og, syn):
        self.og = og
        self.syn = syn
        
    def cal_marginal_indicators(self):
        def extract_numeric_columns(dataset):
            d = []
            for col in dataset.columns:
                if pd.api.types.is_numeric_dtype(dataset[col]):
                    d.append(dataset[col])
            return pd.DataFrame(d).T
        
        og1 = extract_numeric_columns(self.og['d1'])
        syn1 = extract_numeric_columns(self.syn['d1'])
        column_list = list(og1.columns)
        
        if 'd2' in self.og:
            og2 = extract_numeric_columns(self.og['d2'])
            syn2 = extract_numeric_columns(self.syn['d2'])
    
            for col in og2.columns:
                if col not in column_list:
                    column_list.append(col)


            column_list.remove('user_id')
            if 'log_id' in column_list:
                column_list.remove('log_id')


            p_values = {}
            w_dis = {}
            for col in column_list:
                if col in og1.columns and col in syn1.columns:
                    p_values[col] = stats.ks_2samp(og1[col], syn1[col])[1]
                    w_dis[col] = wd(og1[col], syn1[col])
                elif col in og1.columns and col in syn2.columns:
                    p_values[col] = stats.ks_2samp(og1[col], syn2[col])[1]
                    w_dis[col] = wd(og1[col], syn2[col])
                elif col in og2.columns and col in syn1.columns:
                    p_values[col] = stats.ks_2samp(og2[col], syn1[col])[1]
                    w_dis[col] = wd(og2[col], syn1[col])
                elif col in og2.columns and col in syn2.columns:
                    p_values[col] = stats.ks_2samp(og2[col], syn2[col])[1]
                    w_dis[col] = wd(og2[col], syn2[col])
        else:
            p_values = {}
            w_dis = {}
            for col in column_list:
                if col in og1.columns and col in syn1.columns:
                    p_values[col] = stats.ks_2samp(og1[col], syn1[col])[1]
                    w_dis[col] = wd(og1[col], syn1[col])
                
        marginals = {}
        marginals['p-values'] = p_values
        marginals['w-distance'] = w_dis
        self.marginal_indicators = marginals
        
    def cal_conditional_indicators(self):
        def extract_numeric_columns(dataset):
            d = []
            for col in dataset.columns:
                if pd.api.types.is_numeric_dtype(dataset[col]):
                    d.append(dataset[col])
            return pd.DataFrame(d).T
        og1 = extract_numeric_columns(self.og['d1'])
        syn1 = extract_numeric_columns(self.syn['d1'])
        column_list = list(og1.columns)
        
        og2 = extract_numeric_columns(self.og['d2'])
        syn2 = extract_numeric_columns(self.syn['d2'])
        warnings.filterwarnings("ignore")



        def find_cross_party_feature_correlation(og_col1, og_col2, syn_col1, syn_col2, corr_type = "p"):
            og_table = pd.crosstab(og_col1, og_col2)
            syn_table = pd.crosstab(syn_col1, syn_col2)
            all_columns = set(og_table.columns).union(set(syn_table.columns))
            all_rows = set(og_table.index).union(set(syn_table.index))
            for col in all_columns:
                if col not in og_table.columns:
                    og_table[col] = 0
                if col not in syn_table.columns:
                    syn_table[col] = 0
            for row in all_rows:
                if row not in og_table.index:
                    og_table.loc[row] = [0] * len(og_table.columns)
                if row not in syn_table.index:
                    syn_table.loc[row] = [0] * len(syn_table.columns)   
            og_prob = og_table.div(og_table.sum(axis = 1), axis = 0)
            og_prob.fillna(0, inplace=True)
            syn_prob = syn_table.div(syn_table.sum(axis = 1), axis = 0)
            syn_prob.fillna(0, inplace=True)
            row_totals = og_table.sum(axis=1)
            probabilities = row_totals / sum(row_totals)
            corr_table = np.zeros(og_table.shape[0])
                
            for i in range(og_table.shape[0]):
                if corr_type == 'p':
                    if sum(syn_prob.iloc[i,:]) == 0 or sum(og_prob.iloc[i,:]) == 0:
                        corr_table[i] = 0
                    else:
                        corr_table[i] = chisquare(syn_prob.iloc[i,:], og_prob.iloc[i,:])[1]
                else:
                    corr_table[i] = wd(og_prob.iloc[i,:], syn_prob.iloc[i,:])
            return np.matmul(corr_table, probabilities) 
        
        
        total_syn = pd.merge(syn1, syn2, left_on = 'user_id', right_on = 'user_id').drop_duplicates()
        
        total_og = pd.merge(og1, og2, left_on = 'user_id', right_on = 'user_id').drop_duplicates()
        total_og_num = extract_numeric_columns(total_og)
        total_syn_num = extract_numeric_columns(total_syn)
        num_cols = [i for i in total_og_num.columns if i in total_syn_num.columns if i != 'user_id']
        
        p_values = {}
        w_dis = {}
        with tqdm(total = len(num_cols) ** 2) as pbar:
        
            for col1 in num_cols:
                    for col2 in num_cols:
                        p_values[f"{col1}, {col2}"] = find_cross_party_feature_correlation(total_og_num[col1], total_og_num[col2], total_syn_num[col1], total_syn_num[col2], corr_type = 'p')
                        w_dis[f"{col1}, {col2}"] = find_cross_party_feature_correlation(total_og_num[col1], total_og_num[col2], total_syn_num[col1], total_syn_num[col2], corr_type = 'w')
                        pbar.update(1)
                        
        conditionals = {}
        conditionals['p-values'] = {key: value for key, value in p_values.items() if not pd.isna(value)}
        conditionals['w-distance'] = {key: value for key, value in w_dis.items() if not pd.isna(value)}
        self.conditional_indicators = conditionals