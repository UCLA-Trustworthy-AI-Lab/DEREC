import pandas as pd
import numpy as np
import shutil
import os
from realtabformer import REaLTabFormer
from pathlib import Path
from scipy.stats import wasserstein_distance
from scipy.stats import chisquare
import warnings


class data_clean_room:
    def __init__(self, dataset1, dataset2, identifier):
        self.og1 = dataset1.copy()
        self.og2 = dataset2.copy()
        self.identifier = identifier
        self.derec_parent = None
        self.derec_parent_small = None
        for col in self.og1.columns:
            if col in self.og2.columns and col != self.identifier:
                self.og1.rename(columns={col: col + '_1'}, inplace=True)
                self.og2.rename(columns={col: col + '_2'}, inplace=True)
        
    def derec(self):
        def get_parent_cols(dataset, identifier):
            d_parent_cols = [identifier]
            grouped_d = dataset.groupby(identifier)
    
            for column in dataset.columns.difference([identifier]):
                result = grouped_d[column].apply(lambda x: x.nunique() == 1)
                if True in result.value_counts():    
                    if result.value_counts()[True] / len(result) >= 0.95:
                        d_parent_cols.append(column)        
                else:
                    if result.value_counts()[False] / len(result) < 0.05:
                        d_parent_cols.append(column) 
                        
            return d_parent_cols
        
        d1_parent_cols = get_parent_cols(self.og1, self.identifier)
        d2_parent_cols = get_parent_cols(self.og2, self.identifier)
        d1_parent = self.og1[d1_parent_cols]
        d2_parent = self.og2[d2_parent_cols]
        unique_d1_parent = d1_parent.drop_duplicates(subset = [self.identifier])
        unique_d2_parent = d2_parent.drop_duplicates(subset = [self.identifier])
        self.derec_parent = unique_d1_parent.merge(unique_d2_parent, on = self.identifier)
        d1_parent_cols.remove(self.identifier)
        d2_parent_cols.remove(self.identifier)


        d1_child_cols = self.og1.columns.difference(d1_parent_cols)
        d2_child_cols = self.og2.columns.difference(d2_parent_cols)


        self.derec_child_1 = self.og1[d1_child_cols]
        self.derec_child_2 = self.og2[d2_child_cols]

    def sampling(self, n = 1000):
        if self.derec_parent is None:
            self.derec()
        
        if len(self.derec_parent) <= n:
            unique_id = self.derec_parent[self.identifier].sample(len(self.derec_parent), random_state = 1018)
        else:
            unique_id = self.derec_parent[self.identifier].sample(n, random_state = 1018)

        self.derec_parent_small = self.derec_parent[self.derec_parent[self.identifier].isin(unique_id)]
        self.derec_child_1_small = self.derec_child_1[self.derec_child_1[self.identifier].isin(unique_id)]
        self.derec_child_2_small = self.derec_child_2[self.derec_child_2[self.identifier].isin(unique_id)]
        
    def synthesize(self, parent_synthetic_size = '', child_synthetic_size = ''):
        if self.derec_parent_small is None:
            self.sampling()
        
        
        def synthesize_with_rtf_1(parent, child, identifier, parent_synthetic_size = '', child_synthetic_size = ''):
            if parent_synthetic_size == '':
                parent_synthetic_size = len(parent)
            if child_synthetic_size == '':
                child_synthetic_size = len(child)
            join_on = identifier
    
            parent_model = REaLTabFormer(model_type="tabular", epochs = 1, batch_size = 5, train_size = 0.8)
            parent_model.fit(parent.drop(join_on, axis=1), num_bootstrap=5)
    
            save_directory = f"{os.path.dirname(os.path.abspath(__file__))}/fine_tuned_model_1"
    
    
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
    
            pdir = Path(save_directory)
            parent_model.save(pdir)
            parent_model_path = sorted([p for p in pdir.glob("id*") if p.is_dir()], key=os.path.getmtime)[-1]
    
    
            child_model_1 = REaLTabFormer(
                model_type="relational",
                parent_realtabformer_path=parent_model_path, epochs=10, batch_size = 5, train_size = 0.8)
    
            child_model_1.fit(
                df = child,
                in_df = parent,
                join_on = join_on, num_bootstrap = 10)
    
    
            parent_samples = parent_model.sample(parent_synthetic_size)
    
    
            parent_samples.index.name = join_on
            parent_samples = parent_samples.reset_index()
    
            child_samples = child_model_1.sample(n_samples = child_synthetic_size,
                input_unique_ids=parent_samples[join_on],
                input_df=parent_samples.drop(join_on, axis=1),
                output_max_length = None,
                gen_batch = 1)
    
            child_samples.index.name = identifier
            return parent_samples, child_samples
        
        def synthesize_with_rtf_2(parent, child, identifier, parent_synthetic_size = '', child_synthetic_size = ''):
            if parent_synthetic_size == '':
                parent_synthetic_size = len(parent)
            if child_synthetic_size == '':
                child_synthetic_size = len(child)
            join_on = identifier
    
            parent_model = REaLTabFormer(model_type="tabular", epochs = 1, batch_size = 5, train_size = 0.8)
            parent_model.fit(parent.drop(join_on, axis=1), num_bootstrap=5)
    
            save_directory = f"{os.path.dirname(os.path.abspath(__file__))}/fine_tuned_model_2"
    
    
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
    
            pdir = Path(save_directory)
    
            parent_model.save(pdir)
            parent_model_path = sorted([p for p in pdir.glob("id*") if p.is_dir()], key=os.path.getmtime)[-1]
    
    
            child_model_1 = REaLTabFormer(
                model_type="relational",
                parent_realtabformer_path=parent_model_path, epochs=10, batch_size = 5, train_size = 0.8)
    
            child_model_1.fit(
                df = child,
                in_df = parent,
                join_on = join_on, num_bootstrap = 10)
    
    
            parent_samples = parent_model.sample(parent_synthetic_size)
    
    
            parent_samples.index.name = join_on
            parent_samples = parent_samples.reset_index()
    
            child_samples = child_model_1.sample(n_samples = child_synthetic_size,
                input_unique_ids=parent_samples[join_on],
                input_df=parent_samples.drop(join_on, axis=1),
                output_max_length = None,
                gen_batch = 1)
    
            child_samples.index.name = identifier
            return parent_samples, child_samples
        
        self.derec_parent_syn, self.derec_child_1_syn = synthesize_with_rtf_1(self.derec_parent_small, self.derec_child_1_small, self.identifier, parent_synthetic_size = parent_synthetic_size, child_synthetic_size = child_synthetic_size)
        dummy, self.derec_child_2_syn = synthesize_with_rtf_2(self.derec_parent_small, self.derec_child_2_small, self.identifier, parent_synthetic_size = parent_synthetic_size, child_synthetic_size = child_synthetic_size)
        
        subset1 = self.derec_parent_syn[[col for col in self.derec_parent_syn.columns if col in self.og1.columns]]
        subset2 = self.derec_parent_syn[[col for col in self.derec_parent_syn.columns if col in self.og2.columns]]
        self.syn1 = pd.merge(subset1, self.derec_child_1_syn, left_on = self.identifier, right_on = self.identifier)
        self.syn2 = pd.merge(subset2, self.derec_child_2_syn, left_on = self.identifier, right_on = self.identifier)
                
    def simpro(self, syn_d1 = '', syn_d2 = '', method = "p"):
        warnings.filterwarnings("ignore")
        if syn_d1 != '' or syn_d2 != '':
            raise AssertionError("Please use a dataframe format for your synthetic dataset input")
        def get_num_cols(dataset):
            d = []
            for col in dataset.columns:
                if pd.api.types.is_numeric_dtype(dataset[col]):
                    d.append(dataset[col])
            return pd.DataFrame(d).T


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
                    corr_table[i] = wasserstein_distance(og_prob.iloc[i,:], syn_prob.iloc[i,:])
            return np.matmul(corr_table, probabilities) 
        
        if syn_d1 == '' and syn_d2 == '' and self.derec_parent_syn is None:
            raise AssertionError("Please synthesize some data or input your own synthetic datasets!")
        elif syn_d1 == '' and syn_d2 == '' and self.derec_parent_syn is not None:
            total_syn = pd.merge(self.syn1, self.syn2, left_on = self.identifier, right_on = self.identifier).drop_duplicates()
        elif syn_d1 != '' and syn_d2 != '':
            total_syn = pd.merge(syn_d1, syn_d2, left_on = self.identifier, right_on = self.identifier).drop_duplicates()
        elif syn_d1 != '' and syn_d2 == '':
            total_syn = pd.merge(syn_d1, self.syn2, left_on = self.identifier, right_on = self.identifier).drop_duplicates()
        elif syn_d1 == '' and syn_d2 != '':
            total_syn = pd.merge(self.syn1, syn_d2, left_on = self.identifier, right_on = self.identifier).drop_duplicates()
        
        total_og = pd.merge(self.og1, self.og2, left_on = self.identifier, right_on = self.identifier).drop_duplicates()
        total_og_num = get_num_cols(total_og)
        total_syn_num = get_num_cols(total_syn)
        num_cols = [i for i in total_og_num.columns if i in total_syn_num.columns if i != self.identifier]
        
        matrix = []
        name_list = []
        for col1 in num_cols:
                for col2 in num_cols:
                    print(col1, col2)
                    name_list.append(col1 + ', ' + col2)
                    output = find_cross_party_feature_correlation(total_og_num[col1], total_og_num[col2], total_syn_num[col1], total_syn_num[col2], corr_type = method)
                    matrix.append(output)

        if method == 'p':
            name = 'P-values'
        else:
            name = 'W-Distance'
                    
        return pd.DataFrame(matrix, index = name_list, columns = [name]).sort_index().dropna()
