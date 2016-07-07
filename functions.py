
# coding: utf-8

# In[1]:

#convert shot flags to colors
def flag_colors(flag):
    if flag == 1: return 'green'
    if flag == 0: return 'red'
    return 'black'


# In[3]:

# convert matchup string to Home=1, away=0
# example: LAL @ POR: home=0
# example: LAL vs. POR: home=1
def get_home_away(string):
    if '@' in string: return 0 #away
    elif 'vs' in string : return 1 #home
    else: return "ACORDA CRL"


# In[4]:

# convert season in format '1999-00' to 
# example: '1999-00': 3
def get_season_num(year):
    # year is in format '2009-10'
    year0 = 1996
    y = year[:4]
    y_int = int(y)
    
    return y_int - year0

#quick asserts
#assert get_season_num('2010-11') == 14
#assert get_home_away('LAL @ POR') == 0
#assert get_home_away('LAL vs UTA') == 1    


# In[5]:

def write_kaggle_submission(df, probs, output_file='kobe_submission.csv'):
    """
    write output to kaggle format (for kobe competition)
    
    Input:
        df (pandas dataframe): dataframe used for predicting probabilities
        probs (numpy array): array of probability values
        
    Output:
        None
    
    Side effects:
        creates csv file "output_file"
        
    Comments:
        number of rows in df should match size of probs array
    """

    # create new Series with df indexes as shot_id and probs values as shot_made_flag
    
    data_values = probs
    index_values = df.index.values +1
    data_dict = {'shot_id': index_values, 'shot_made_flag': data_values}
    
    df_towrite = pd.DataFrame(data = data_dict)
    
    df_towrite.to_csv(output_file, header=True, index=False)
    
    return


# In[6]:

def logloss(act, pred):
    """
    logloss function
    imported from kaggle evalutation
    """
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


# In[ ]:

def preprocess(original_data, shot_made_flag=False):
    """
    Pre-processing of dataframe into desired form
    
    Input:
        original_data (Pandas dataframe): original data from which the final dataframe will be formed
        shot_made_flag (boolean): whether to include labels (shot made) or not
    
    Output:
        df (pandas dataframe): the dataframe in final form

    Comments:
        highly overfit to kobe shot selection problem
        
    """
    
    # select relevant features
    feature_list = ['loc_x', 'loc_y', 'shot_distance', 'period', 'season', 'minutes_remaining', 'seconds_remaining', 
           'matchup']
    
    if shot_made_flag is True:
        feature_list.append('shot_made_flag')
    
    df = original_data[feature_list].copy()
    
    # modify selected features
    
    # convert shot_distance from feet to meters
    df.loc[:,'shot_distance'] = df['shot_distance'].apply(lambda x: x*0.3048)    
    
    # add angle feature and clean NaN by assuming angle=0 when distance=0
    df.loc[:,'angle'] = pd.Series(np.degrees(np.arctan(df['loc_x']/df['loc_y'])))
    df['angle'].fillna(0, inplace=True)

    # convert matchup to Home/Away (Home=0, Away=1)
    df.loc[:, 'Home'] = df['matchup'].apply(get_home_away)    

    # convert seasons to first, second etc
    # needs: convert to date 
    df.loc[:,'season'] = df['season'].apply(get_season_num)

    # convert minutes + seconds remaining to time remaining in quarter (in seconds)
    df.loc[:, 'time_remaining'] = df['minutes_remaining']*60 + df['seconds_remaining']

    # clean dataframe
    cols_to_delete = ['loc_x', 'loc_y', 'minutes_remaining', 'seconds_remaining', 'matchup']
    df.drop(cols_to_delete, axis=1, inplace=True)
    
    # clean NaN in shot_made_flag column
    df.dropna(axis=0, how='any', inplace=True)

    # make sure no NaNs in dm
    assert df.isnull().any().any()==False    
    
    return df
    

