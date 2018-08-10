import numpy as np
import pandas as pd

from os import path

import itertools
from collections import Counter
import random
import time

from dask.distributed import Client, progress
import dask.delayed as delayed
import dask.dataframe as dd

def order_pk(order_products, rec_products, k=10):
    '''
    Calculate precision@k for a set of ordered products vs. recommended products.
    
    Parameters
    ----------
    order_products: array-like
        List of product IDs for a given order
    rec_products: array-like
        Sorted list of recommended product IDs (highest to lowest)
    k: int, array-like (default = 10)
        K threshold at which to evaluate precision
    ----------
    Returns: precision score (float)
    '''
    if isinstance(k, (int,np.integer)):
        k = [k]
    precisions = {}
    for k_ in k:
        # Get top K recommendations
        top_n_recs = rec_products[:k_]

        # Number of recommendations in order
        n_hit = len(set(top_n_recs).intersection(order_products))

        # Discount precision denominator for imbalances between # of recommendations and # of ordered products
        m = min(len(order_products), len(top_n_recs))
        # Proportion of correct recommendations
        precisions['p@{:02d}'.format(k_)] = (n_hit / m)

    return precisions


def order_map(order_products, rec_products, n_range=10):
    '''
    Calculate the MAP score for a given set of orderered products and recommended products.
    
    Paramters
    ---------
    order_products: array-like (int)
        List of product IDs in a given order
    rec_products: array-like (int)
        List of recommended product IDs
    n_range: int, array-like
        Value or list of values at which to evaluate MAP. Scoring is calculated relative to the first n recommendations.
    ---------
    Returns: dict(string: float)
    '''
    if isinstance(n_range, (int, np.integer)):
        n_range = [n_range]
    order_product_set = set(order_products)
    order_eval = {}
    for n in n_range:
        top_n_recs = rec_products[:n]
        # Minimum of number of products in order or rec; avoids unfair scoring for length mismatches
        m = min(len(order_product_set), n)
        n_hit = 0
        cum_prec = 0
        for i, rec in enumerate(top_n_recs):
            # Check if recommendation is contained in order
            hit = int(rec in order_product_set)
            # Increment hit count
            n_hit += hit
            # Cumulative precision of hits, dicounted for rank
            cum_prec += hit*n_hit/(i+1)

        # Mean precision according to number of products/recs
        ap = cum_prec/m
        order_eval['map@{:02d}'.format(n)] = ap
    
    return order_eval


def order_tna(user_id, order_products, rec_products, sample_size=100, n_range=10):
    '''
    Calculate Top-N Accuracy for a given user and set of order products and recommended products.
    
    Parameters
    ----------
    user_id: int
        User ID for which order is being evaluated. Required to pull list of non-ordered products for said user.
    order_products: array-like (int)
        List of products in a given order
    rec_products: array-like (int)
        List of recommended products (must be sorted by rank)
    sample_size: int (default: 100)
        Number of random non-purchased products to combine with each order product 
        when evaluating recommendation relevancy/effectiveness.
    n_range: int, array-like (default: 10)
        N value(s) at which to evaluate Top-N Accuracy.
    ----------
    Returns: dict(string: float)
    '''
    if isinstance(n_range, (int, np.integer)):
        n_range = [n_range]
        
    non_ordered_products_set = get_non_ordered_products(user_id)
    
    order_eval = {}
    hits_at_n = Counter()
    for product_id in set(order_products):
        # get 100 non purchased products
        non_ordered_sample = set(random.sample(non_ordered_products_set, k=100))
        # combine with product of interest
        tna_sample = non_ordered_sample.union([product_id])
        # rank products according to recs
        ranked_products = rec_products[rec_products.isin(tna_sample)]
        # top_n eval
        for n in n_range:
            if ranked_product_id in ranked_products.values[:n]:
                hits_at_n[n] += 1
    
    n_products = len(order_products)
    for n in n_range:
        order_eval['tna@{:02d}'.format(n)] = hits_at_n[n]/n_products
    
    return order_eval

def get_non_ordered_products(user_id):
    ordered_products = ord_prod.loc[user_id]
    non_ordered_product_set = set(full_product_list).difference(ordered_products)
    return non_ordered_product_set


class ModelEvaluator():
    N_CORES = 8
    
    def __init__(self, eval_data, method='precision@k', n_range=[5,10]):
        '''
        Model evaluator class to perform scoring on recommender systems.
        Utilizes dask distributed processing with the assumption a default client has already been configured.
        
        Parameters
        ----------
        method: string (default: 'precision@k')
            Scoring method to perform. Options are 'precision@k' (default), or 'map'
        eval_data: pandas.DataFrame
            Data on which to perform evaluation. Assumes columns of ['user_id', 'order_id', 'product_id],
            with a row instance for each product ordered in a given order & user.
        n_range: int, array-like (default: [5,10])
            N value(s) at which to perform evaluation (i.e precision@10)
        ----------
        '''
        # Set n_range
        if isinstance(n_range, (int, np.integer)):
            n_range = [n_range]
        self.n_range = n_range
        # Set list of unique products
        self.product_list = None
        # Set method
        self.method = method
        # Set eval data
        self.eval_data = None
        self.update_data(eval_data=eval_data)
    
    def update_data(self, eval_data):
        '''
        Update the data on which to perform evaluation
        
        Parameters
        ----------
        eval_data: pandas.DataFrame
            Order-product data consisting columns for 'user_id' (int), 'order_id' (int), and 'product_id' (int).
        ----------
        '''
        # Store eval data with User ID as index
        self.eval_data = eval_data[['user_id', 'order_id', 'product_id']].copy().set_index('user_id').sort_index()
        # List of unique users in evaluation dataset
        self.unique_users = np.array(self.eval_data.index.unique())
        # Number of unique users in evaluation dataset
        self.nunique_users = self.unique_users.shape[0]
    
    def evaluate_model(self, model, method=None, eval_data=None, return_full=False):
        '''
        Evaluate a given recommender system. Evaluation data can be provided again to overwrite existing dataset.
        
        Parameters
        ----------
        model: RecommenderSystem object (custom class)
            Recommender system model to be evaluated. Assume model to have been fitted prior to evaluation.
        eval_data: pandas.DataFrame, optional
            Evaluation data to overwrite existing dataset. See update_data for details.
        method: string, optional
            Method with which to evaluate model performance. Will overwrite existing method.
            Available options are 'precision@k' or 'map'
        return_full: bool, optional (default: False)
            Whether to include the full user-order scoring results versus just the aggreggated scores.
        ----------
        Returns: dict (return_full = False), or tuple(dict, pandas.DataFrame) (return_full = True)
        '''
        # Update data if provided
        if eval_data is not None:
            self.update_data(eval_data=eval_data)
        # Update method if provided
        if method is not None:
            self.method = method
        # Unset verbose in model
        model.set_params(verbose=False)    
        if self.method == 'precision@k':
            eval_results = self.eval_pk(model)
        elif self.method == 'map':
            eval_results = self.eval_map(model)
        else:
            raise ValueError('method {} not recognized.'.format(self.method))
        
        if return_full:
            return eval_results
        else:
            return eval_results[0]
    
    def eval_pk(self, model):
        '''
        Evaluate Precision@K for a given model
        '''
        dd_test = dd.from_pandas(self.eval_data, npartitions=self.N_CORES, sort=True)
        
        # Precompute user recommendations. Assumes n_range is within reasonable bounds.
        user_recs = model.recommend(user_id=self.unique_users, n_rec=max(self.n_range))
        
        # Apply order_pk function to each order per user
        dd_order_pk = dd_test.groupby(['user_id', 'order_id'])['product_id']\
                                     .apply(lambda x: order_pk(x, user_recs.loc[x.name[0]], k=self.n_range),\
                                            meta=('precison@k', float))
        dd_order_pk = dd_order_pk.compute()
        # Get results
        order_pk_results = dd_order_pk.unstack()
        order_pk_mean = order_pk_results.mean()
        
        # Convert to summary dict
        global_metrics = {'model_name': model.MODEL_NAME, **order_pk_mean.to_dict()}
        
        return global_metrics, order_pk_results
    
    def eval_map(self, model):
        '''
        Evaluate MAP for a given model
        '''
        dd_test = dd.from_pandas(self.eval_data, npartitions=self.N_CORES, sort=True)
        
        # Precompute user recommendations. Assumes n_range is within reasonable bounds.
        user_recs = model.recommend(user_id=self.unique_users, n_rec=max(self.n_range))
        
        # Apply order_pk function to each order per user
        dd_order_map = dd_test.groupby(['user_id', 'order_id'])['product_id']\
                                     .apply(lambda x: order_map(x, user_recs.loc[x.name[0]], self.n_range),\
                                            meta=('precison@k', float))
        dd_order_map = dd_order_map.compute()
        # Get results
        order_map_results = dd_order_map.unstack()
        order_map_mean = order_map_results.mean()
        
        # Convert to summary dict
        global_metrics = {'model_name': model.MODEL_NAME, **order_map_mean.to_dict()}
        
        return global_metrics, order_map_results
    

def recommender_gs(recommender, train_data, test_data, param_dict, scoring = 'map', scoring_n_range = [5,10], verbose = False):
    
    t_start = time.time()
    
    from sklearn.model_selection import ParameterGrid
    param_grid = ParameterGrid(param_dict)
    
    if type(scoring) is not list:
        scoring = [scoring]
    
    mev = ModelEvaluator(eval_data=test_data, n_range=scoring_n_range)
    
    gs_results = []
    
    for param_set in param_grid:
        if verbose:
            print('Performing evaluation for {}'.format(param_set))
        
        if verbose:
            print('\t{:30}'.format('Fitting...'), end='\r')
        rec = recommender
        rec.set_params(**param_set, verbose=False)
        rec.fit(train_data)
        
        scores = {}
        for scoring_method in scoring:
            if verbose:
                print('\t{:30}'.format("Evaluating '{}'...".format(scoring_method)), end='\r')
            scores.update(mev.evaluate_model(rec, method = scoring_method))
        
        summary = {'params': param_set, **{**scores}}
        try:
            del summary['model_name']
        except:
            return summary
        gs_results.append(summary)
        
        if verbose:
            print('\t{:30}'.format("Complete!".format(scoring_method)))
    
    gs_results_df = pd.DataFrame(gs_results)
    return gs_results_df