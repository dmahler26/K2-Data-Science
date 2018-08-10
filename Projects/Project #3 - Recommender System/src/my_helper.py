from sys import getsizeof
from os import path
import pandas as pd

def mem_size(obj):
    '''
    Return the memory size of an object in MB in string format.
    '''
    return "{0:.2f} MB".format(getsizeof(obj) / (1000 * 1000))

def chunkify(lst,n):
    '''
    Split list into n chunks
    '''
    return [lst[i::n] for i in range(n)]

def get_product_name(product_id, fp=path.join('..','data','raw','products.csv'), reload=False):
    '''
    Get the product name for a given product_id. Function caches product data for successive calls, but can be forced
    to reload data via 'reload' parameter.
    
    Parameters
    ----------
    product_id: int or array-like
        Product ID to retrieve name for.
    fp: string, (default: path.join('..','data','raw','products.csv'))
        Filepath from which to retrieve product csv data.
    reload: bool (default: False)
        Reload the product csv data.
    ----------
    Returns: product name (string) if ID found, else np.nan
    '''
    # shorthand for "self"
    f = get_product_name
    try:
        f.product_data
        if reload:
            raise          
    except:
        # Load product data
        with open(fp, 'rb') as file:
            f.product_data = pd.read_csv(file, encoding='utf8').set_index('product_id')
    
    product_name = f.product_data.loc[product_id,'product_name'].values

    return product_name