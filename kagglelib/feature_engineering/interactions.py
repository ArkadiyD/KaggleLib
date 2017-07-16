import numpy as np
import pandas as pd

def make_numerical_interactions(train, test, colnames, operation, order = 2, symmetric = False):
    '''create arithmetic combinations of order 2 or order 3 for numerical features

    Parameters
    ----------
    train : Pandas Dataframe
        train dataset
    test : Pandas Dataframe
        test dataset
    colnames : list of string
        column names to make interactions
    operation : 'plus', 'minus', 'mult', 'div'
        performed operation between columns
    order : boolean, optional.
        order of multiplications
        2 by default.
    symmetric : boolean, optional
        whether to include both a<operation>b and b<operation>a for division
        False by default

    Returns
    -------
    list of names of created columns
    '''

    new_cols = []
    for i in xrange(len(colnames)):
        col1 = colnames[i]
        for j in xrange(i, len(colnames), 1):
            
            col2 = colnames[j]
            
            if order == 2:

                if operation == 'plus':
                    train.loc[:, col1 + '_plus_' + col2] = train[col1].astype(np.float32) + train[col2].astype(np.float32)
                    test.loc[:, col1 + '_plus_' + col2] = test[col1].astype(np.float32) + test[col2].astype(np.float32)

                elif operation == 'minus':
                    train.loc[:, col1 + '_minus_' + col2] = train[col1].astype(np.float32) - train[col2].astype(np.float32)
                    test.loc[:, col1 + '_minus_' + col2] = test[col1].astype(np.float32) - test[col2].astype(np.float32)

                elif operation == 'mult':
                    train.loc[:, col1 + '_mult_' + col2] = train[col1].astype(np.float32) * train[col2].astype(np.float32)
                    test.loc[:, col1 + '_mult_' + col2] = test[col1].astype(np.float32) * test[col2].astype(np.float32)

                elif operation == 'div':
                    train.loc[:, col1 + '_div_' + col2] = train[col1].astype(np.float32) / train[col2].astype(np.float32)
                    test.loc[:, col1 + '_div_' + col2] = test[col1].astype(np.float32) / test[col2].astype(np.float32)
                    if symmetric:
                        train.loc[:, col2 + '_div_' + col1] = train[col2].astype(np.float32) / train[col1].astype(np.float32)
                        test.loc[:, col2 + '_div_' + col1] = test[col2].astype(np.float32) / test[col1].astype(np.float32)
                else:
                    raise ValueError("operation should be 'plus' or 'minus' or 'mult' or 'div', got %s" % operation)

                new_cols.append(col1 + '_%s_' % operation + col2) 
            
            elif order == 3:
                for q in xrange(j, len(colnames), 1):
                    
                    col3 = colnames[q]
                    
                    if operation == 'plus':
                        train.loc[:, col1 + '_plus_' + col2 + '_plus_' + col3] = train[col1].astype(np.float32) + train[col2].astype(np.float32) +  train[col3].astype(np.float32)
                        test.loc[:, col1 + '_plus_' + col2 + '_plus_' + col3] = test[col1].astype(np.float32) + test[col2].astype(np.float32) + test[col3].astype(np.float32)

                    elif operation == 'minus':
                        train.loc[:, col1 + '_minus_' + col2 + '_minus_' + col3] = train[col1].astype(np.float32) - train[col2].astype(np.float32) - train[col3].astype(np.float32)
                        test.loc[:, col1 + '_minus_' + col2 + '_minus_' + col3] = test[col1].astype(np.float32) - test[col2].astype(np.float32) - test[col3].astype(np.float32)
                        train.loc[:, col2 + '_minus_' + col1 + '_minus_' + col3] = train[col2].astype(np.float32) - train[col1].astype(np.float32) - train[col3].astype(np.float32)
                        test.loc[:, col2 + '_minus_' + col1 + '_minus_' + col3] = test[col2].astype(np.float32) - test[col1].astype(np.float32) - test[col3].astype(np.float32)
                        train.loc[:, col3 + '_minus_' + col1 + '_minus_' + col2] = train[col3].astype(np.float32) - train[col1].astype(np.float32) - train[col2].astype(np.float32)
                        test.loc[:, col3 + '_minus_' + col1 + '_minus_' + col2] = test[col3].astype(np.float32) - test[col1].astype(np.float32) - test[col2].astype(np.float32)

                    elif operation == 'mult':
                        train.loc[:, col1 + '_mult_' + col2 + '_mult_' + col3] = train[col1].astype(np.float32) * train[col2].astype(np.float32) * train[col3].astype(np.float32)
                        test.loc[:, col1 + '_mult_' + col2 + '_mult_' + col3] = test[col1].astype(np.float32) * test[col2].astype(np.float32) * test[col2].astype(np.float32)

                    elif operation == 'div':
                        train.loc[:, col1 + '_div_' + col2 + '_div_' + col3] = train[col1].astype(np.float32) / train[col2].astype(np.float32) / train[col3].astype(np.float32)
                        test.loc[:, col1 + '_div_' + col2 + '_div_' + col3] = test[col1].astype(np.float32) / test[col2].astype(np.float32) / test[col3].astype(np.float32)
                        train.loc[:, col2 + '_div_' + col1 + '_div_' + col3] = train[col2].astype(np.float32) / train[col1].astype(np.float32) / train[col3].astype(np.float32)
                        test.loc[:, col2 + '_div_' + col1 + '_div_' + col3] = test[col2].astype(np.float32) / test[col1].astype(np.float32) / test[col3].astype(np.float32)
                        train.loc[:, col3 + '_div_' + col1 + '_div_' + col2] = train[col3].astype(np.float32) / train[col1].astype(np.float32) / train[col2].astype(np.float32)
                        test.loc[:, col3 + '_div_' + col1 + '_div_' + col2] = test[col3].astype(np.float32) / test[col1].astype(np.float32) / test[col2].astype(np.float32)
                    else:
                        raise ValueError("operation should be 'plus' or 'minus' or 'mult' or 'div', got %s" % operation)

                new_cols.append(col1 + '_%s_' % operation + col2 + '_%s_' % operation + col3) 

    return new_cols

def make_categorical_interactions(train, test, colnames, order = 2, sep = '_'):
    '''create multiplications of order 2 or order 3 for numerical features

    Parameters
    ----------
    train : Pandas Dataframe
        train dataset
    test : Pandas Dataframe
        test dataset
    colnames : list of string
        column names to make interactions
    order : boolean, optional.
        order of multiplications
        2 by default.
    sep : string, optional
        separator to insert in generatedinteractions
        '_' by default

    Returns
    -------
    list of names of created columns
    '''
    
    new_cols = []
    for i in xrange(len(colnames)):
        col1 = colnames[i]
        for j in xrange(i + 1, len(colnames), 1):
            col2 = colnames[j]
            
            if order == 2:
                train.loc[:, col1 + sep + col2] = train[col1].astype(np.str) + train[col2].astype(np.str)
                test.loc[:, col1 + sep + col2] = test[col1].astype(np.str) + test[col2].astype(np.str)
                new_cols.append(col1 + sep + col2)
            
            elif order == 3:
                for q in xrange(j + 1, len(colnames), 1):
                    col3 = colnames[q]
        
                    train.loc[:, col1 + sep + col2 + sep + col3] = train[col1].astype(np.str) + train[col2].astype(np.str) + train[col3].astype(np.str)
                    test.loc[:, col1 + sep + col2 + sep + col3] = test[col1].astype(np.str) + test[col2].astype(np.str) + test[col3].astype(np.str)
                    new_cols.append(col1 + sep + col2 + sep + col3)
    
    return new_cols