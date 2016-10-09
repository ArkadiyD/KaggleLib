Description of functions.

Data hashing. Functions to hash dataframes into sparse matrices.

  get_hash_data(train, test, colnames_to_hash, colnames_not_to_hash = [], type = 'tfidf', verbose = True)
    hashes train and test datasets into a sparse matrices
    
    train: Pandas dataframe; input dataframe to hash
    test: Pandas dataframe; input dataframe to apply hashing based on train dataset
    colnames_to_hash: list of strings; list of column names to be hashed
    colnames_not_to_hash: list of strings; lsit of column names to be remained without hashing
    type: string; 'tfidf' - hashing with TfidfVectorizer(min_df=1),'tfidf_modified' - hashing with TfidfVectorizer(min_df=1, ngram_range=(1,1),sublinear_tf = True,binary = True, norm = 'l2'), 'counter' - hashing with CountVectorizer(min_df=1, binary=1)
    verbose: boolean, display messages or not
    
  get_hash_data_simple(train, test, colnames_to_hash, type = 'tfidf', verbose = True)
  get_hash_data_memory_efficient(train, test, colnames_to_hash, type = 'tfidf', verbose = True)
