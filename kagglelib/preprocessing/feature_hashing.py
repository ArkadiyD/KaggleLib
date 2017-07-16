from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
import gc
import copy
from scipy import sparse

def hash_data(train, test, colnames_to_hash, type = 'tfidf', min_df = 0.001, max_df = 0.999, binary = False, sep = '_', verbose = True):
    ''' hashes text features
    
    It is assumed that in colnames_to_hash columns there are text values which may be separaterd by single space
    Hashing is performed with respect to column, the same words in different columns are treated as different words,
    Word tokens as obtained as column_name + sep + word

    Parameters
    ----------
    train : pandas.DataFrame
        train dataset
    test : pandas.DataFrame
        test dataset
    colnames_to_hash : list of strings
        colnames which will be hashed, all other columns will be preserved
    type : string, optional, 'tfidf', 'tfidf_modified' or 'counter'
        type of hashing operation, 'tfidf' for binary TfidfVectorizer, 
        'tfidf_modified' for TfidfVectorizer with l2 norm,
        'counter' for binary CountVectorizer,
        'tfidf' by default
    min_df : float, optional
        words with frequency (n_occurences / words_in_dataframe_count) less than min_df will be ignored
        0.001 by default
    max_df : float, optional
        words with frequency (n_occurences / words_in_dataframe_count) less than min_df will be ignored
        0.001 by default
    binary : boolean, optional
        whether to produce binary (0/1) hashing
        False by default
    sep : string, optional
        symbol to set between column name and words
        '_' by default

    Returns
    -------
    predictions array
    '''

    colnames_not_to_hash = list(set(train.columns.values) - set(colnames_to_hash))
    df_all = pd.concat((train.copy(), test.copy()), axis = 0, ignore_index = True)
    split_len = len(train)

    if type == 'tfidf':
        tfv = TfidfVectorizer(min_df = min_df, max_df = max_df, binary = binary)
    elif type == 'tfidf_modified':
        tfv = TfidfVectorizer(min_df = min_df, max_df = max_df, sublinear_tf = True, binary = binary, norm = 'l2')
    elif type == 'counter':
        tfv = CountVectorizer(min_df = min_df, max_df = max_df, binary = binary)
    else:
        raise ValueError("type should be 'tfidf' or 'tfidf_modified' or 'counter', got %s" % type)

    try:
        for col in colnames_to_hash:
            df_all[col] = df_all[col].astype(np.str).apply(lambda x: x.replace(' ', ',' + col + sep)).fillna('Missing')
            df_all[col] = df_all[col].astype(np.str).apply(lambda x: col + sep + x).fillna('Missing')
        
        df_all = df_all[colnames_to_hash].astype(np.str).apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')
    
    except Exception:
        for col in colnames_to_hash:
            df_all[col] = df_all[col].astype('U').apply(lambda x: col + sep + x).fillna('Missing')
            df_all[col] = df_all[col].astype('U').apply(lambda x: x.replace(' ', ',' + col + sep)).fillna('Missing')

        df_all = df_all[colnames_to_hash].astype('U').apply(lambda x: ','.join(s for s in x), axis=1).fillna('Missing')

    if verbose:
        print '-'*30, 'df for hashing\n', df_all
    
    df_all = tfv.fit_transform(df_all)
    
    if verbose:
        print 'vocab:'
        for v in tfv.vocabulary_:
            print unicode(v)
        print 'stop words:'
        for v in tfv.stop_words_:
            print unicode(v)

    gc.collect()
    
    if verbose:
        print '-'*30, 'df hashed\n', df_all.shape, df_all
    
    not_hashed_df_train = train[colnames_not_to_hash]
    not_hashed_df_test = test[colnames_not_to_hash]
    
    not_hashed_df_train = sparse.coo_matrix(not_hashed_df_train.values)
    not_hashed_df_test = sparse.coo_matrix(not_hashed_df_test.values)

    train = df_all[:split_len, :]
    test = df_all[split_len:, :]
    

    if len(colnames_not_to_hash) > 0:
        train = sparse.hstack([not_hashed_df_train, train])
        test = sparse.hstack([not_hashed_df_test, test])

    del df_all
    gc.collect()
    
    return train, test
