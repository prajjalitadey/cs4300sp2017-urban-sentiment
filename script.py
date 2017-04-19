import pandas as pd

df_reviews = pd.DataFrame.from_csv('jsons/nycreviews.csv', index_col=None)
df_listings = pd.DataFrame.from_csv('jsons/nyclistings.csv', index_col=None)
df_reviews = df_reviews[['listing_id', 'comments']]
df_listings = df_listings[['id', 'neighbourhood']]
df_listings = df_listings.rename(columns={'id': 'listing_id'})
df_combination = pd.merge(df_listings, df_reviews, on='listing_id')
df_combination.to_csv('jsons/nyc_combination.csv')
