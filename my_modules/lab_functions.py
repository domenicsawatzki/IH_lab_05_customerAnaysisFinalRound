import pandas as pd
import numpy as np

    
def customerDataCleanAndPrepareForModel(df):
    import numpy as np
    df.columns= [col.lower() for col in df.columns]
    df.columns= [col.replace(' ', '_') for col in df.columns]
    df.rename(columns={'st':'state'}, inplace=True)
    df.rename(columns={'monthly_premium_auto':'monthly_premium_costs'}, inplace=True)
    df.drop(columns=['customer', 'effective_to_date'], axis=1)

    
    y = df['total_claim_amount']
    X = df.drop(['total_claim_amount'], axis=1)

    X_num = X.select_dtypes(include = np.number)
    X_cat = X.select_dtypes(include = object)
    
    from sklearn.preprocessing import MinMaxScaler # do not use the function Normalise() - it does something entirely different
    MinMaxtransformer = MinMaxScaler().fit(X_num)
    
    X_normalized = MinMaxtransformer.transform(X_num)
    X_normalized = pd.DataFrame(X_normalized,columns=X_num.columns)

    
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(drop='first')
    encoder.fit(X_cat)
    encoded_X_cat = encoder.transform(X_cat)
    # Convert the encoded data to a pandas DataFrame
    encoded_X_cat = pd.DataFrame(encoded_X_cat.toarray(), columns=encoder.get_feature_names_out(X_cat.columns))
    
    final_X = pd.concat([X_normalized ,encoded_X_cat], axis=1)

    return final_X, y
