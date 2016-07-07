
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



