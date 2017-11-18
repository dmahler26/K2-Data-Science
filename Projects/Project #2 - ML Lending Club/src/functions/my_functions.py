import pandas as pd
import numpy as np

import time

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV, cross_val_predict, learning_curve
from sklearn.metrics.scorer import make_scorer

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, precision_recall_curve, fbeta_score

### PIPELINE FUNCTIONS ###

class DataFrame_Selector(BaseEstimator, TransformerMixin):
    '''
    Selects the provided list of attributte (column) names from a pandas dataframe and converts to numpy array.
    '''
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    def get_feature_names(self):
        return self.attribute_names   
    
class DataFrame_DummyEncoder(BaseEstimator, TransformerMixin):
    '''
    Performs OHE on pandas dataframe via pandas.get_dummies(),
    using the provided list of attribute (column) names.
    
    List of valid pandas.get_dummies() column names can be provided
    to ensure consistency between datasets with different sets of values (recommended)
    '''
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        # Get set of valid dummy columns when fitting
        self.valid_dummy_cols = pd.get_dummies(X[self.attribute_names]).columns
        return self
    
    def transform(self, X):
        zero_data = np.zeros(shape=(len(X),len(self.valid_dummy_cols)))
        self.dummies = pd.DataFrame(zero_data, columns=self.valid_dummy_cols)
        d = pd.get_dummies(X[self.attribute_names])

        for col in d.columns:
            if col in self.dummies.columns:
                self.dummies[col] = d[col].values
        
        return self.dummies.values
    
    def get_feature_names(self):
        return self.dummies.columns.tolist()
        
class CustomNumAttributes(BaseEstimator, TransformerMixin):
    '''
    Creates custom numerical attributes. Requires that the data be passed in as a pandas dataframe
    to allow for column name references versus hardcoding column indexes.
    
    A references dictionary is built when fitting, which is used in subsequent calculations dependent
    on reference dataset statistics (mean, std, etc.). See reference_stats function for details.
    '''
    def __init__(self):
        pass
    def fit(self, data, y=None):
        ### Get reference stats 
        self.ref_dict = reference_stats(data)
        return self
    def transform(self, data, y=None):
        X = data.copy()
        # Note: following assumes that X is still be a pandas DataFrame vs. numpy array. 
        self.custom_attr_names = []
        ###
        grade_map = {grade: i for i, grade in enumerate('ABCDEFG')}
        X['grade_value'] = X['grade'].map(grade_map)
        self.custom_attr_names.append('grade_value')
        ###
        subgrade_map = {sg: grade_map[sg[0]]*10 + int(sg[1]) for sg in [c + str(i) for c in 'ABCDEFG' for i in range(1,6)]}
        X['subgrade_value'] = X['sub_grade'].map(subgrade_map)
        self.custom_attr_names.append('subgrade_value')
        ###
        X['lti'] = X['funded_amnt'] / X['annual_inc']
        X['iti'] = X['installment'] / X['annual_inc']
        X['rbti'] = X['revol_bal'] / X['annual_inc']
        X['tbti'] = X['tot_cur_bal'] / X['annual_inc']
        self.custom_attr_names.append(['lti', 'iti', 'rbti', 'tbti'])
        ###
        X['revol_bal_log'] = X['revol_bal'].apply(lambda x: np.log10(x) if x >= 1 else 0)
        X['tot_coll_log'] = X['tot_coll_amt'].apply(lambda x: np.log10(x) if x >= 1 else 0)
        X['rev_lim_log'] = X['total_rev_hi_lim'].apply(lambda x: np.log10(x) if x >= 1 else 0)
        X['rev_lim_sqrt'] = np.sqrt(X['total_rev_hi_lim'])
        self.custom_attr_names.append(['revol_bal_log', 'tot_coll_log', 'rev_lim_log', 'rev_lim_sqrt'])
        ###
        X['earliest_cr_line_td'] = [(issue_d.date() - cr.date()).days for issue_d, cr in zip(X['issue_d'], X['earliest_cr_line'])]
        X['cr_line_td_log'] = X['earliest_cr_line_td'].apply(lambda x: np.log10(x) if x >= 1 else 0)
        self.custom_attr_names.append(['earliest_cr_line_td', 'cr_line_td_log'])
        ###
        ref_dict = self.ref_dict
        if ref_dict is not None:
            ###
            if 'grade_p_map' in ref_dict:
                X['grade_p_value'] = X['grade'].map(ref_dict['grade_p_map'])
                self.custom_attr_names.append('grade_p_value')
            ###
            if 'subgrade_p_map' in ref_dict:
                X['subgrade_p_value'] = X['sub_grade'].map(ref_dict['subgrade_p_map'])
                self.custom_attr_names.append('subgrade_p_value')
            ###
            if ('subgrade_int_rate_mean' in ref_dict) and ('subgrade_int_rate_std' in ref_dict):
                X['int_rate_delta'] = X[['int_rate','sub_grade']].apply(lambda x:
                                                                        (x['int_rate'] - 
                                                                         ref_dict['subgrade_int_rate_mean'][x['sub_grade']]) /
                                                                         ref_dict['subgrade_int_rate_std'][x['sub_grade']], axis=1)
                self.custom_attr_names.append('int_rate_delta')
            ###
            if 'annual_inc_q10' in ref_dict:
                X['annual_inc_q10'] = X['annual_inc'].apply(lambda x: len(ref_dict['annual_inc_q10'])
                                                            if x >max(ref_dict['annual_inc_q10'])
                                                            else np.argmax(x <= ref_dict['annual_inc_q10'])+1)
                self.custom_attr_names.append('annual_inq_q10')
            ##
            if 'funded_amnt_q10' in ref_dict:
                X['funded_amnt_q10'] = X['funded_amnt'].apply(lambda x: len(ref_dict['funded_amnt_q10'])
                                                              if x > max(ref_dict['funded_amnt_q10'])
                                                              else np.argmax(x <= ref_dict['funded_amnt_q10'])+1)
                self.custom_attr_names.append('funded_amnt_q10')
            ###
        return X
    def get_feature_names(self):
        return self.custom_attr_names
    

class CustomBinAttributes(BaseEstimator, TransformerMixin):
    '''
    Creates custom binary attributes. Requires that the data be passed in as a pandas dataframe
    to allow for column name references versus hardcoding column indexes.
    '''
    def __init__(self):
        pass
    def fit(self, data, y=None):
        return self
    def transform(self, data, y=None):
        X = data.copy()        
        # Note: following assumes that X is still be a pandas DataFrame vs. numpy array. 
        self.custom_attr_names = []
        ###
        X['verified'] = (X['verification_status'] != 'Not Verified').astype(int)
        self.custom_attr_names.append('verified')
        ###
        return X
    def get_feature_names(self):
        return self.feature_names
    

def reference_stats(reference_data):
    '''
    Creates a dictionary of statistics etc. for the provided reference dataset.
    Used in CustomNumAttributes() transformer for custom feature calculations which require original means, standard devs, etc.
    '''
    d = {}
    data = reference_data.copy()
    default = data['loan_status'].str.contains('Charged Off|Default').astype(int)
    d['grade_p_map'] = default.groupby(data['grade']).mean()
    d['subgrade_p_map'] = default.groupby(data['sub_grade']).mean()
    
    d['subgrade_int_rate_mean'] = data.groupby('sub_grade')['int_rate'].mean()
    d['subgrade_int_rate_std'] = data.groupby('sub_grade')['int_rate'].std()
    
    d['annual_inc_q10'] = data['annual_inc'].quantile(np.arange(0.1, 1.1, 0.1))
    d['funded_amnt_q10'] = data['funded_amnt'].quantile(np.arange(0.1, 1.1, 0.1))
    return d

###

### PEFORMANCE EVAL SUMMARIES & VISUALIZATIONS ###

def classifier_summary(y_actual, y_pred, print_results=True): 
    '''
    Calculate basic accuracy scores and confusion matrix for a set of classifier actual vs. predicted values.
    Prints results by default, and returns a confusion matrix and scores dataframe.
    '''
    
    f2 = fbeta_score(y_actual, y_pred, beta=2)
    recall = recall_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    
    conf_mat = confusion_matrix(y_actual, y_pred)
    
    specificity = conf_mat[0,0] / (conf_mat[0,:].sum())
    fallout = 1 - specificity
    precision_neg = conf_mat[0,0] / (conf_mat[:,0].sum())
    
    df_cmat = pd.DataFrame(conf_mat).rename(index={0:'Actual Negative', 1:'Actual Positive'},
                                  columns={0:'Predicted Negative', 1:'Predicted Positive'})
    
    df_scores = pd.DataFrame([{'Rate': 'F2', 'Score': f2},
                              {'Rate': 'Recall', 'Score': recall},
                              {'Rate': 'Precision (pos)', 'Score': precision},
                              {'Rate': 'Precision (neg)', 'Score': precision_neg},
                              {'Rate': 'Specificity', 'Score': specificity}]).set_index('Rate')
    
    if print_results:
        print('Confusion Matrix:')
        print(df_cmat)
        print(20*'-')
        print('Accuracy Scores:')
        print(df_scores)
   
    return df_cmat, df_scores


def gs_score_summary(gs):
    '''
    Prints the best scoring results for each scoring method supplied to a grid search.
    '''
    scores = gs.scoring
    print('-'*20)
    for score in scores:
        i = np.argmin(gs.cv_results_['rank_test_' + str(score)])
        print('Best {}:'.format(score.title()))
        print('Params: {}'.format(gs.cv_results_['params'][i]))

        for s in scores:
            print('{} = {}'.format(s.title(), gs.cv_results_['mean_test_'+str(s)][i]))
        print('-'*20)

        
def print_cvs(cvs, scoring='CV'):
    '''
    Prints mean score and standard deviation for a cross_val_score
    '''
    print('Mean {} score = {:.3f} (+\- {:.3f})'.format(scoring, cvs.mean(), cvs.std()))
    
    
def plot_learning_curve(estimator, X, y, scoring=None, train_sizes=[0.1, 0.325, 0.55 , 0.775, 1.], cv=None, random_state=None, n_jobs=1, figsize=(10,5), xlim=None, ylim=None, title=None, subplots=False):
    '''
    Plots the learing curve for a given estimator and datasets X, y. Can take a single scorer or list of scorers, and plot on the same axes
    or use individual subplots for each depending on the subplots=True/False param.
    '''
    scoring = list(scoring)
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if subplots:
            fig, axs = plt.subplots(len(scoring),1, figsize=figsize)
    else:
        axs = [plt.figure(figsize=figsize).gca()] * len(scoring)
          
    for i, (score, ax) in enumerate(zip(scoring, axs)):
        try:
            lc = learning_curve(estimator=estimator, X=X, y=y, train_sizes=train_sizes, scoring=score, cv=cv, 
                                random_state=random_state, n_jobs=n_jobs)
            ax.plot(lc[0], np.mean(lc[1], axis=1), label='Train ({})'.format(score.title()), ls='--', alpha=0.5, color=colors[i], lw=2)
            ax.plot(lc[0], np.mean(lc[2], axis=1), label='Validate ({})'.format(score.title()), color=colors[i], lw=2)
            
            if xlim is not None:
                ax.xlim(xlim)
            if ylim is not None:
                ax.ylim(ylim)
            
            ax.set_xlabel('Samples')
            ax.set_ylabel('Score')
            ax.legend(bbox_to_anchor=(1,1))
            ax.set_title('Learning Curve: {}'.format(score.title() if subplots else ', '.join(scoring).title()))
            
        except Exception as e:
            print(e)
    
    if title is not None:
        plt.suptitle(title)
    
    plt.tight_layout()
    plt.show()   

    
def plot_pr_curve(model, X, y, label=None, f_score=False, beta=1, cv=3, n_jobs=1):
    '''
    Plots both PR vs. decision thresholds and PR curve for a given estimator and dataset X, y.
    '''
    if hasattr(model, 'decision_function'):
        y_scores = cross_val_predict(model, X, y, cv=cv, method='decision_function', n_jobs=n_jobs)[:,1]
    else:
        y_scores = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=n_jobs)[:,1]
    
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    
    fig, axs = plt.subplots(1,2,figsize=(15,7))
    
    x_min = max(thresholds[(len(recalls) - np.argmax(recalls[::-1][:-1]) -1)]*1.1, min(thresholds))
    x_max = min(thresholds[np.argmax(precisions[:-1])]*1.1, max(thresholds))
    
    y_min = min(min(recalls), min(precisions))
    y_max = max(max(recalls), max(precisions))
    y_delta = y_max-y_min
    y_min = max(y_min-0.1*y_delta, 0)
    y_max += 0.1*y_delta

    axs[0].plot(thresholds, precisions[:-1], 'b-', label='Precision')
    axs[0].plot(thresholds, recalls[:-1], 'g-', label='Recall')
    if f_score:
        f_scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls)
        best_index = np.argmax(f_scores[:-1])
        axs[0].plot(thresholds, f_scores[:-1], 'k--', label='F'+str(beta))
        axs[0].plot([thresholds[best_index], ] * 2, [y_min, max(f_scores[best_index], recalls[best_index], precisions[best_index])],
                    linestyle='-.', color='k', lw=1)
        axs[0].plot(thresholds[best_index], y_min, marker='x', color='k', markeredgewidth=2, ms=6)
        axs[0].plot(thresholds[best_index], f_scores[best_index], marker='x', color='k', markeredgewidth=2, ms=6)
        axs[0].annotate("%0.2f" % f_scores[best_index], (thresholds[best_index], f_scores[best_index] + 0.01))
        axs[0].annotate("%0.2f" % thresholds[best_index], (thresholds[best_index], y_min + 0.015))
        
        axs[0].plot(thresholds[best_index], recalls[best_index], color='g', marker='x', markeredgewidth=2, ms=6)
        axs[0].annotate("%0.2f" % recalls[best_index], (thresholds[best_index], recalls[best_index] + 0.015))
        
        axs[0].plot(thresholds[best_index], precisions[best_index], color='b', marker='x', markeredgewidth=2, ms=6)
        axs[0].annotate("%0.2f" % precisions[best_index], (thresholds[best_index], precisions[best_index] + 0.015))
    x_min = max(thresholds[(len(recalls) - np.argmax(recalls[::-1][:-1]) -1)]*1.1, min(thresholds))
    x_max = min(thresholds[np.argmax(precisions[:-1])]*1.1, max(thresholds))
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    axs[0].set_xlabel('Decision Threshold')
    axs[0].set_ylabel('Precision / Recall')
    axs[0].legend()

    axs[1].plot(recalls[:-1], precisions[:-1], 'r-')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')

    plt.suptitle('Precision/Recall Curve{}'.format(': ' + str(label) if label is not None else ''))

    plt.show()
    
    return precisions, recalls, thresholds


def plot_gs_param(gs, set_xscale=False, ylim=(0,1), val_label=True, figsize=(12,12)):
    '''
    Plot scores with varying grid search param values (only compatiable with single param grid searches)
    '''
    if len(gs.param_grid) > 1:
        print('Error: grid search contains more than one parameter. Function only compatible with single parameter grid searches.')
        return None
    
    param = list(gs.param_grid.keys())[0]
    
    plt.figure(figsize=figsize)
    plt.title("GridSearchCV Scores for Parameter {}".format(param), fontsize=16)

    plt.xlabel(param)
    plt.ylabel("Score")
    plt.grid()

    ax = plt.axes()
    if set_xscale:
        ax.set_xscale(set_xscale)
    
    ax.set_ylim(ylim)
    
    results = gs.cv_results_

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_' + param].data, dtype=float)
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, scorer in enumerate(sorted(gs.scoring)):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=colors[i])
            ax.plot(X_axis, sample_score_mean, style, color=colors[i],
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, 'val' if (val_label and sample=='test') else sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [ylim[0], best_score],
                linestyle='-.', color=colors[i], marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))
    
    plt.legend(bbox_to_anchor=(1,1))
    plt.grid('off')
    plt.show()
###

### MISC HELPER FUNCTIONS ###

def run_time(reset=False, print_time=True, return_time=False, pretty=True):
    '''
    Basic timer function. Pass reset=True to reset the start time.
    Call function again (default reset=False) to display time passed since start.
    '''
    if reset:
        run_time.start_time = time.time()
    else:
        td = time.time()-run_time.start_time
        m = td//60
        s = td%60
        ms = 1000*(s%1)
        if print_time:
            if pretty:
                display = 'Time: '
                display += ('{:.0f}min '.format(m) if m > 0 else '')
                display += ('{:.0f}s '.format(s) if (s > 1 and m > 0) else ('{:.2f}s '.format(s) if (s > 1) else ''))
                display += ((str(round(ms))+'ms ') if (s < 1) else '')
            else:
                display = td
            print(display)
        if return_time:
            return td
