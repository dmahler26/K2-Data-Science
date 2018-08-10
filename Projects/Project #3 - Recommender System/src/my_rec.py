import numpy as np
import pandas as pd

import itertools
from collections import Counter
import random
import time

from dask.distributed import Client, progress
import dask.delayed as delayed
import dask.dataframe as dd

import warnings

class RecommenderSystem():
    '''
    Base class for recommender systems
    '''
    MODEL_NAME = 'Base'
    
    def __init__(self, n_rec=10, rank=False, verbose=False, decay=False, decay_method='linear', decay_constant=None):
        '''
        Parameters
        ----------
        n_rec: int, optional (default: 10)
            Number of recommendations to return. Value of -1 returns full list.
        rank: bool, optional (default: True)
            Whether to include rank in index of returned recommendations. Note that results will typically be
            sorted regardless of the inclusion of this rank index.
        verbose: bool, optional (default: False)
            Whether to print status/progress during processing. Implementation may vary depending on model complexity. 
        '''
        self.n_rec = n_rec
        self.verbose = verbose
        self.rank = rank
        
        self.fitted = False
    
    def fit(data=None):
        '''
        Dummy base class fit method.
        '''
        raise NotImplementedError()
    
    def get_model_name(self):
        '''
        Get model name.
        '''
        return self.MODEL_NAME
    
    def get_params(self):
        '''
        Get model parameters.
        Returns: dict
        '''
        d = {'n_rec': self.n_rec,
             'rank': self.rank,
             'verbose': self.verbose}
        
        return d
    
    def set_params(self, **kwargs):
        '''
        Set model parameters.
        
        Parameters
        ----------
        n_rec: int, optional
            Provides option to overwrite existing n_rec parameter for the number of recommendations to return per user.
            A value of -1 returns the full list. 
        rank: bool, optional
            Provides option to overwrite existing rank parameter to include ranking index values of recommendations.
            Note that recommendations will be returned sorted by rank regardless of the inclusion of this index.
        verbose: bool, optional
            Whether to print status/progress during processing. Implementation may vary depending on model complexity.            
        ----------
        '''
        self.n_rec = kwargs.pop('n_rec', self.n_rec)
        self.verbose = kwargs.pop('verbose', self.verbose)
        self.rank = kwargs.pop('rank', self.rank)
        
        if len(kwargs) > 0:
            warnings.warn('Parameters {} not found and have been ignored.'.format(list(kwargs.keys())))
            
    def recommend(self, df_rec=None):
        '''
        Base recommend method. Not implemented for use outside of child recommenders.
        '''
        # Check if called without provides recommendation results (i.e. from actual recommender)
        if df_rec is None:
            raise NotImplementedError()
        
        if self.rank:
            ranking = []
            for user, recs in df_rec.groupby(level=0):
                # Rank recs per user from 1:n_rec and append to overarching list
                ranking += [i + 1 for i, rec in enumerate(recs)]
            if type(df_rec) is pd.Series:
                df_rec = df_rec.to_frame()
            # Set generated list to new rank column (aligns with data)
            df_rec['rank'] = ranking
            # Set user id and rank as index
            df_rec = df_rec.reset_index().set_index(['user_id', 'rank'])
        
        return df_rec
    
class GlobalPopularityRecommender(RecommenderSystem):
    '''
    Global Popularity Recommender system: produces recommendations based off most popular (i.e. frequently ordered)
    items across all users.
    
    Parameters
    ----------
    n_rec: int (default: -1)
        Number of product recommendations to return. Setting to -1 will return all products ranked.
    ----------
    '''
    MODEL_NAME = 'Global Popularity'
    
    def fit(self, data):
        '''
        Fit recommender using prior order product data.
        
        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe containing history of order products. Assumes format of one row per product ordered.
            Must contain column of 'product_id' on which to sum frequencies.
        ----------
        '''
        # Produce sorted list of products IDs according to purchase frequency
        self.sorted_product_ids = data['product_id'].value_counts().sort_values(ascending=False).index.values
        self.fitted = True
    
    def recommend(self, user_id=0, **kwargs):
        '''
        Recommend products for a given user. Note that given the nature of this global popularity recommender,
        recommendations are the same for all users, but user_id can still be provided for consistency amongst
        recommender system parameters.
        
        Parameters
        ----------
        user_id: int or array-like, optional (default: 0)
            User ID to produce recommendations for. Note that for this particular recommender system, recommendations
            are the same for all users and this parameter is merely included for consistency amongst recommenders.
        ----------
        Returns: pandas.DataFrame, index = ['user_id', 'rank'], cols = ['product_id]
        '''
        # Check if fit was performed
        if not self.fitted:
            raise RuntimeError('cannot recommend without fitting!')
        # Overwrite params if provided
        self.set_params(**kwargs)
        # Check user_id
        if isinstance(user_id, (int, np.integer)):
            user_id = [user_id]
        if type(user_id) not in (list, np.ndarray):
            raise ValueError('user_id not of accepted type. Must be int or array-like.')
        
        if self.n_rec == -1:
            # get all products
            df_rec = pd.Series(data=np.tile(self.sorted_product_ids,len(user_id)),
                               index=np.repeat(user_id,self.sorted_product_ids.shape[0]), name='product_id')
        else:
            # get top n products
            df_rec = pd.Series(data=np.tile(self.sorted_product_ids[:self.n_rec],len(user_id)),\
                               index=np.repeat(user_id,self.n_rec), name='product_id')
        
        df_rec.index.name = 'user_id'
        
        return super(GlobalPopularityRecommender, self).recommend(df_rec=df_rec)
    
    
class UserPopularityRecommender(RecommenderSystem):
    '''
    Global Popularity Recommender system: produces recommendations based off most popular (i.e. frequently ordered)
    items across all users.
    
    Parameters
    ----------
    n_rec: int (default: 10)
        Number of product recommendations to return. Setting to -1 will return all products ranked.
    decay: bool (default: True)
        Whether to implement time decay to product recommendation ranking
    decay_method: str (default: 'linear')
        Method with which to decay product purchase values according to time of purchase.
        Choice between 'linear' or 'exponential' decay methods.
        Linear decay reduces products weight linearly toward 0 relative to the maximum order age present.
        Exponential decay redced products weight according to calculated/provided half-life of the product.
    decay_constant: float (default: None)
        Decay parameter determining the strength of decay. Is automatically calculated if not provided
        For linear decay, default value is 1. Values less than 1 will decrease decay rate, values above will increase decay rate.
        For exponential decay, default value is median order age. Defining this parameter manually sets the half-life interval (scale dependent)
        at which products weights decrease by half.
        
    ----------
    '''
    MODEL_NAME = 'User Popularity'
    
    def __init__(self, decay=True, decay_method='linear', decay_constant=None, **kwargs):
        super(UserPopularityRecommender, self).__init__(**kwargs)
        self.decay = decay
        self.decay_method = decay_method
        self.decay_constant = decay_constant
    
    def set_params(self, **kwargs):
        self.decay = kwargs.pop('decay', self.decay)
        self.decay_method = kwargs.pop('decay_method', self.decay_method)
        self.decay_constant = kwargs.pop('decay_constant', self.decay_constant)
        super(UserPopularityRecommender, self).set_params(**kwargs)
    
    def get_params(self):
        d = super(UserPopularityRecommender, self).get_params()
        d['decay'] = self.decay
        d['decay_method'] = self.decay_method
        d['decay_constant'] = self.decay_constant
        return d
    
    def fit(self, data):
        '''
        Fit recommender using prior order product data.
        
        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe containing history of order products. Assumes format of one row per product ordered.
            Must contain column of 'product_id' on which to sum frequencies.
        ----------
        '''
        t_start = time.time()
        if self.verbose: print('[{:.0f}s] Fit starting...'.format(time.time()-t_start))
        data = data.copy()
        
        if self.decay:
            if self.verbose:
                print('[{:.0f}s] Adjusting purchase values for time decay'.format(time.time()-t_start))
            # Fill NaN (i.e. the first orders) with 0
            data['days_since_prior_order'].fillna(0, inplace=True)
            
            # Cumulative sum of days passed with progressing order numbers
            order_times = data.groupby(['user_id', 'order_number'])[['days_since_prior_order']].mean().groupby(level=[0]).cumsum()
            
            # Rename columns
            order_times.rename(columns={'days_since_prior_order': 'days_since_first_order'}, inplace=True)
            
            # Invert time scale (days since first order -> days since last order)
            order_times['days_since_last_order'] = dd.from_pandas(order_times.reset_index(1), npartitions=8)\
                                                     .groupby('user_id')['days_since_first_order']\
                                                     .apply(lambda x: x.max()-x, meta=int)
            self.order_times = order_times
            # Merge new time values into original dataframe
            data = data.merge(order_times, left_on = ['user_id', 'order_number'], right_index=True)
            
            if self.decay_method == 'linear':
                if self.decay_constant is None:
                    self.decay_constant = 1
                t_max = data['days_since_last_order'].max()
                t_min = 0
                data['weighted_product_value'] = data['days_since_last_order'].apply(lambda x: max(0, 1 - self.decay_constant*(x/(t_max - t_min))))
            
            elif self.decay_method == 'exponential':
                if self.decay_constant is None:
                    # Default to half life being the median age of an order
                    self.decay_constant = data['days_since_last_order'].median()*np.log(2)
                data['weighted_product_value'] = np.exp(-1/self.decay_constant * data['days_since_last_order'])
            
            else:
                raise ValueError("decay method not recognized.")
                
        else:
            # All products get value of 1 without decay
            data['weighted_product_value'] = 1
            
        # convert data to dask dataframe
        dd_data = dd.from_pandas(data[['user_id', 'order_id', 'product_id', 'weighted_product_value']], npartitions=8)
        # get global product counts
        if self.verbose:
            print('[{:.0f}s] Getting global product values'.format(time.time()-t_start))
        self.glob_product_val = dd_data.groupby('product_id')['weighted_product_value'].sum()\
                                                                                       .to_frame(name='global_purchase_value')\
                                                                                       .compute()\
                                                                                       .sort_values(by='global_purchase_value')
        
        # get user product counts
        if self.verbose:
            print('[{:.0f}s] Getting user product values'.format(time.time()-t_start))
        self.user_product_val = dd_data.groupby(['user_id','product_id'])['weighted_product_value'].sum()\
                                                                            .to_frame(name = 'user_purchase_value')\
                                                                            .compute()\
                                                                            .reset_index(1) # extract product id from index
        
        if self.verbose: print('[{:.0f}s] Fit complete'.format(time.time()-t_start))
        self.fitted = True
    
    def recommend(self, user_id, **kwargs):
        '''
        Recommend products for a given user. Note that given the nature of this global popularity recommender,
        recommendations are the same for all users, but user_id can still be provided for consistency amongst
        recommender system parameters.
        
        Parameters
        ----------
        user_id: int or array-like, optional (default: 0)
            User ID to produce recommendations for. Note that for this particular recommender system, recommendations
            are the same for all users and this parameter is merely included for consistency amongst recommenders.
        ----------
        Returns: pandas.DataFrame, index = ['user_id'], cols = ['product_id]
        '''
        # Check if fit was performed
        if not self.fitted:
            raise RuntimeError('cannot recommend without fitting!')
        # Overwrite params if provided
        self.set_params(**kwargs)
        
        t_start = time.time()
        if self.verbose: print('[{:.0f}s] Rec starting...'.format(time.time()-t_start))
        
        # Limit to top n_rec results per user
        if self.verbose: print('[{:.0f}s] Getting top n user products'.format(time.time()-t_start))
        user_data = self.user_product_val.loc[user_id].reset_index()
        top_user_products =  dd.from_pandas(user_data, npartitions=8)\
                                     .groupby('user_id')[['product_id', 'user_purchase_value']]\
                                     .apply(lambda x: x.nlargest(self.n_rec, columns=['user_purchase_value']),\
                                            meta={'product_id': int, 'user_purchase_value': int})\
                                     .compute()\
                                     .reset_index(1, drop=True).reset_index() # clean up grouped indexing
        
        # create dataframe of user purchases with user & global purchase counts
        if self.verbose: print('[{:.0f}s] Combining user and global purchase counts'.format(time.time()-t_start))
        user_recs_df = top_user_products.join(self.glob_product_val, on='product_id')
        # get additional recommendations for users with insufficient purchase histories
        if self.verbose: print('[{:.0f}s] Getting additionl global recommendations per user'.format(time.time()-t_start))
        def get_extra_glob_recs(user_recs, n_rec, global_product_counts):
            '''
            Helper function for getting additional recommendations to pad results up to n_rec
            '''
            n = user_recs.shape[0]
            # Initialize dict of additional recs
            new_data = {'product_id': [],
                        'global_purchase_value': [],
                        'user_purchase_value': 0}
            # If missing recs
            if n < n_rec:
                # n missing recs
                n_delta = n_rec-n
                # non-recommended products from global list
                new_prods_mask = ~global_product_counts.index.isin(user_recs['product_id'].values)
                # get top n recs to add
                add_rec = global_product_counts[new_prods_mask][:n_delta]
                # populate dict lists
                new_data['product_id'] += add_rec.index.tolist()
                new_data['global_purchase_value'] += add_rec['global_purchase_value'].values.tolist()
            return pd.DataFrame(new_data)
        
        add_rec = dd.from_pandas(user_recs_df, npartitions=8)
        add_rec = add_rec.groupby('user_id').apply(lambda x: get_extra_glob_recs(x, self.n_rec, self.glob_product_val),
                                                   meta={'product_id': int, 'global_purchase_value': int, 'user_purchase_value': int})\
                                            .compute()
        
        # if any additional results
        if add_rec.shape[0] > 0:
            add_rec = add_rec.reset_index(1, drop=True).reset_index() # clean up grouped indexing
            add_rec = add_rec[['user_id', 'product_id', 'user_purchase_value', 'global_purchase_value']]
            # combine user recs with additional recs
            if self.verbose: print('[{:.0f}s] Combining original and additional recommendation results'.format(time.time()-t_start))
            user_recs_df = pd.concat([user_recs_df, add_rec], ignore_index=True)
        
        # sort and index
        self.user_recs_df = user_recs_df.sort_values(by=['user_id', 'user_purchase_value', 'global_purchase_value'],
                                                     ascending=[True, False, False]).set_index('user_id')
        
        if self.verbose: print('[{:.0f}s] Rec complete'.format(time.time()-t_start))
        df_rec = self.user_recs_df.loc[user_id]['product_id']
        return super(UserPopularityRecommender, self).recommend(df_rec=df_rec)