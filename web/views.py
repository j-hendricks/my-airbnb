from django.shortcuts import render


# our home page view
def home(request):    
    return render(request, 'index.html')

# custom method for generating predictions
def getPredictions(city, zipcode, latitude, longitude, accommodates, bathrooms, bedrooms, beds, review_scores_rating, month):
    import pandas as pd
    import pickle
    import numpy as np
    model = pickle.load(open("model.sav", "rb"))
    scaled = pickle.load(open("scaler.sav", "rb"))
    enc = pickle.load(open('enc.sav','rb'))
    df = pickle.load(open('df.sav','rb'))

    df.loc[len(df.index)] = [city, zipcode, latitude, longitude, accommodates, bathrooms, bedrooms, beds, review_scores_rating, month]

    encode_df = pd.DataFrame(enc.fit_transform(df[['city', 'zipcode', 'month']]))
    encode_df.columns = enc.get_feature_names(['city','zipcode','month'])
    
    df = df.merge(encode_df, left_index=True, right_index=True)
    df.drop(columns=['city','zipcode','month'], inplace=True)
    
    df['bedrooms']=df['bedrooms'].apply(np.sqrt)
    df['bathrooms'] = df['bathrooms'].apply(lambda x: pow(x,1/3))
    df['accommodates']=df['accommodates'].apply(np.log10)

    vals = df.iloc[-1].values
    vals = vals.reshape(1,-1)
    vals = scaled.transform(vals)

    print("THE COLUMNS:", df.columns)

    prediction = model.predict(vals)
    
    results = f"The price will be between ${round([prediction - 25][0][0][0])} and ${round([prediction + 25][0][0][0])}."

    return results
        

# our result page view
def result(request):
    city = str(request.GET['city'])
    zipcode = str(request.GET['zipcode'])
    latitude = float(request.GET['latitude'])
    longitude = float(request.GET['longitude'])
    accommodates = int(request.GET['accommodates'])
    bathrooms = int(request.GET['bathrooms'])
    bedrooms = int(request.GET['bedrooms'])
    beds = int(request.GET['beds'])
    review_scores_rating = int(request.GET['review_scores_rating'])
    month = int(request.GET['month'])

    result = getPredictions(city, zipcode, latitude, longitude, accommodates, bathrooms, bedrooms, beds, review_scores_rating, month)

    return render(request, 'result.html', {'result':result})
