import pandas as pd
import numpy as np
import logging
import geopandas as gpd
from cleaning_preprocessing_func import *
logging.basicConfig(level=logging.WARNING)

pd.set_option('display.max_rows', None)


# Nick path
#PATH_REPO = r'C:/Users/nickb/OneDrive/Documenten/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team4'

# Cristian path
PATH_REPO = r'C:/Users/ASUS/OneDrive/Documents/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team4' 

PATHS_CSV = [
    (f'{PATH_REPO}/data/raw_data/cbr_grid_data_raw.csv', ','),
    (f'{PATH_REPO}/data/raw_data/breda_grid_keys.csv', ';'),
    (f'{PATH_REPO}/data/raw_data/neighborhood_data_most_recent.csv', ';'),
    (f'{PATH_REPO}/data/dashboard_data/revised/boom_per_buurt_breda.csv', ','),
    (f'{PATH_REPO}/data/raw_data/housing_data/page-9_table-1.csv', ','),
    (f'{PATH_REPO}/data/raw_data/housing_data/page-10_table-1.csv', ','),
    (f'{PATH_REPO}/data/raw_data/housing_data/page-11_table-1.csv', ','),
    (f'{PATH_REPO}/data/raw_data/housing_data/page-12_table-1.csv', ','),
    (f'{PATH_REPO}/data/raw_data/housing_data/page-13_table-1.csv', ','),
    (f'{PATH_REPO}/data/raw_data/Leefbaarometer 3.0 - meting 2020 - scores grid.csv', ',')
]

df_grid, df_grid_keys, df_neighborhood, df_boom, df_p9, df_p10, df_p11, df_p12, df_p13, df_lbm = load_csvs(PATHS_CSV)



# check_data_pd(df_boom)
neighborhood_drop = ['Opleidingsniveau 15-75-jarigen - Laag', 'Opleidingsniveau 15-75-jarigen - Midden',
             'Opleidingsniveau 15-75-jarigen - Hoog', 'Personen met een migratieachtergrond',
             'Personen met een westerse achtergrond', 'Personen met een niet westerse achtergrond'] 

grid_drop = ['Unnamed: 0', 'MAN', 'VROUW', 'P_NL_ACHTG', 'P_WE_MIG_A', 'P_NW_MIG_A',]

df_grid = clean_csv(df_grid, col_drop_threshold=1.0, to_drop=grid_drop, check_data=False)
#df_grid = clean_csv(df_grid, col_drop_threshold=1.0, value=-99997, fill_mean=False, to_drop=grid_drop, check_data=False)
df_grid_fillmean = clean_csv(df_grid, fill_mean=True, value=-99997, check_data=False)
df_grid_keys = clean_csv(df_grid_keys, col_drop_threshold=0.5, check_data=False, dropna=True,)
df_neighborhood = replace_value(df_neighborhood, '?', np.nan, replace_str_to_int=False)
df_neighborhood = replace_comma(df_neighborhood)
df_neighborhood = clean_csv(df_neighborhood, fill_mean=True, col_drop_threshold=0.9, check_data=False, to_drop=neighborhood_drop,)
df_lbm_18 = df_lbm[df_lbm['jaar'] == 2018]
df_lbm = clean_csv(df_lbm_18, fill_mean=True, check_data=False)

diff_boom_key = find_difference(df_boom['neighborhood_name'], df_grid_keys['neighborhood_name'])
df_boom = clean_csv(df_boom, fill_mean=False, to_drop=['cbs_grid_code', 'bu_naam', 'gm_code',], col_drop_threshold=0.5,)
df_boom = pd.concat([df_boom, df_grid_keys[df_grid_keys['neighborhood_name'].isin(diff_boom_key[1])]]).drop_duplicates('neighborhood_name')
df_boom = clean_csv(df_boom, fill_mean=True, to_drop=['cbs_grid_code'])
df_p9 = clean_csv(df_p9, col_dropna=True, specified_col='Buurtnaam', col_drop_threshold=0.5, fill_mean=True, to_drop=['Totaal'], check_data=False,)
df_p10 = clean_csv(df_p10, col_dropna=True, specified_col='Buurtnaam', col_drop_threshold=0.5, fill_mean=True, to_drop=['Totaal'], check_data=False,)
df_p11 = clean_csv(df_p11, col_dropna=True, specified_col='Buurtnaam', col_drop_threshold=0.5, fill_mean=True, to_drop=['Totaal'], check_data=False,)
df_p12 = clean_csv(df_p12, col_dropna=True, specified_col='Buurtnaam', col_drop_threshold=0.5, fill_mean=True, to_drop=['Totaal'], check_data=False,)
df_p13 = clean_csv(df_p13, col_dropna=True, specified_col='Buurtcode', col_drop_threshold=0.5, fill_mean=True, to_drop=['Totaal'], check_data=False,)

# print(df_grid_keys[df_grid_keys['neighborhood_name'].isin(diff_boom_key[1])])

# check_data_pd(df_grid, value_count=-99997)
# check_data_pd(df_grid_keys)
# check_data_pd(df_neighborhood, value_count='?', dtype=True)
# check_data_pd(df_boom)
# check_data_pd(df_p11)
# check_data_pd(df_p12)
# check_data_pd(df_p13)

df_grid.to_csv(f'{PATH_REPO}/data/cleaned_data/CBS_grid_data.csv', index=False)
df_grid_fillmean.to_csv(f'{PATH_REPO}/data/cleaned_data/CBS_grid_data_fillmean.csv', index=False)
df_grid_keys.to_csv(f'{PATH_REPO}/data/cleaned_data/breda_grid_keys.csv', index=False)
df_neighborhood.to_csv(f'{PATH_REPO}/data/cleaned_data/neighborhood_data.csv', index=False)
df_boom.to_csv(f'{PATH_REPO}/data/cleaned_data/boom_per_buurt.csv', index=False)
df_p9.to_csv(f'{PATH_REPO}/data/cleaned_data/housing_data/page-9_table-1.csv', index=False)
df_p10.to_csv(f'{PATH_REPO}/data/cleaned_data/housing_data/page-10_table-1.csv', index=False)
df_p11.to_csv(f'{PATH_REPO}/data/cleaned_data/housing_data/page-11_table-1.csv', index=False)
df_p12.to_csv(f'{PATH_REPO}/data/cleaned_data/housing_data/page-12_table-1.csv', index=False)
df_p13.to_csv(f'{PATH_REPO}/data/cleaned_data/housing_data/page-13_table-1.csv', index=False)
df_lbm_18.to_csv(f'{PATH_REPO}/data/cleaned_data/lbm_18.csv', index=False)
# python data_cleaning.py
