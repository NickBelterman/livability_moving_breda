import pandas as pd
from cleaning_preprocessing_func import * 

# Nick Path
#PATH_REPO = r'C:\Users\nickb\OneDrive\Documenten\GitHub\2022-23d-1fcmgt-reg-ai-01-group-team4'

#Cristian Path
PATH_REPO= r'C:/Users/ASUS/OneDrive/Documents/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team4'

PATHS_MODELLING = [
    (f'{PATH_REPO}/data/cleaned_data/breda_grid_keys.csv', ','),
    (f'{PATH_REPO}/old_data/moving_data_processed.csv', ','),
    (f'{PATH_REPO}/data/cleaned_data/CBS_grid_data.csv', ',')
]

cbs_keys, moving, cbs_grid_data = load_csvs(PATHS_MODELLING)

# Translation for columns from Dutch to English
column_mapper = column_names = {
    'INWONER': 'POPULATION',
    'c28992r100': 'cbs_grid_code',
    'MAN': 'MALE',
    'VROUW': 'FEMALE',
    'INW_014': 'POP_0_14',
    'INW_1524': 'POP_15_24',
    'INW_2544': 'POP_25_44',
    'INW_4564': 'POP_45_64',
    'INW_65PL': 'POP_65_PLUS',
    'GEBOORTE': 'BIRTHS',
    'P_NL_ACHTG': 'P_DUTCH_BACKGROUND',
    'P_WE_MIG_A': 'P_WESTERN_IMMIGRANTS',
    'P_NW_MIG_A': 'P_NON_WESTERN_IMMIGRANTS',
    'AANTAL_HH': 'NUM_HOUSEHOLDS',
    'TOTHH_EENP': 'TOT_SINGLE_PERSON_HH',
    'TOTHH_MPZK': 'TOT_MULTI_PERSON_HH',
    'HH_EENOUD': 'HH_ONE_PARENT',
    'HH_TWEEOUD': 'HH_TWO_PARENTS',
    'GEM_HH_GR': 'AVG_HH_SIZE',
    'WONING': 'HOUSES',
    'WONVOOR45': 'HOUSES_BEFORE_1945',
    'WON_4564': 'HOUSES_1945_1964',
    'WON_6574': 'HOUSES_1965_1974',
    'WON_7584': 'HOUSES_1975_1984',
    'WON_8594': 'HOUSES_1985_1994',
    'WON_9504': 'HOUSES_1995_2004',
    'WON_0514': 'HOUSES_2005_2014',
    'WON_1524': 'HOUSES_2015_LATER',
    'WON_MRGEZ': 'HOUSES_MULTYFAMILY',
    'P_KOOPWON': 'P_OWNED_HOUSING',
    'P_HUURWON': 'P_RENTED_HOUSING',
    'WON_HCORP': 'CORP_HOUSING',
    'WON_NBEW': 'NOT_INHABITED_HOUSING',
    'WOZWONING': 'AVG_HOUSING_VALUE',
    'G_GAS_WON': 'GAS_CONNECTION',
    'G_ELEK_WON': 'ELECTRICITY_CONNECTION',
    'M_INKHH': 'AVG_INCOME_HH',
    'UITKMINAOW': 'INCOME_SOCIAL_SECURITY',
    'AFS_SUPERM': 'DIST_SUPERMARKETS',
    'AV1_SUPERM': 'NUM_1_SUPERMARKET',
    'AV3_SUPERM': 'NUM_3_SUPERMARKETS',
    'AV5_SUPERM': 'NUM_5_SUPERMARKETS',
    'AFS_DAGLMD': 'DIST_DAYCARE_CENTERS',
    'AV1_DAGLMD': 'NUM_1_DAYCARE_CENTER',
    'AV3_DAGLMD': 'NUM_3_DAYCARE_CENTERS',
    'AV5_DAGLMD': 'NUM_5_DAYCARE_CENTERS',
    'AFS_WARENH': 'DIST_DEPARTMENT_STORES',
    'AV5_WARENH': 'NUM_5_DEPARTMENT_STORES',
    'AV10WARENH': 'NUM_10_DEPARTMENT_STORES',
    'AV20WARENH': 'NUM_20_DEPARTMENT_STORES',
    'AFS_CAFE': 'DIST_CAFES',
    'AV1_CAFE': 'NUM_1_CAFE',
    'AV3_CAFE': 'NUM_3_CAFES',
    'AV5_CAFE': 'NUM_5_CAFES',
    'AFS_CAFTAR': 'DIST_NIGHTCLUBS',
    'AV1_CAFTAR': 'NUM_1_NIGHTCLUB',
    'AV3_CAFTAR': 'NUM_3_NIGHTCLUBS',
    'AV5_CAFTAR': 'NUM_5_NIGHTCLUBS',
    'AFS_HOTEL': 'DIST_HOTELS',
    'AV5_HOTEL': 'NUM_5_HOTELS',
    'AV10_HOTEL': 'NUM_10_HOTELS',
    'AV20_HOTEL': 'NUM_20_HOTELS',
    'AFS_RESTAU': 'DIST_RESTAURANTS',
    'AV1_RESTAU': 'NUM_1_RESTAURANT',
    'AV3_RESTAU': 'NUM_3_RESTAURANTS',
    'AV5_RESTAU': 'NUM_5_RESTAURANTS',
    'AFS_BSO': 'DIST_AFTER_SCHOOL_CARE',
    'AV1_BSO': 'NUM_1_AFTER_SCHOOL_CARE',
    'AV3_BSO': 'NUM_3_AFTER_SCHOOL_CARE',
    'AV5_BSO': 'NUM_5_AFTER_SCHOOL_CARE',
    'AFS_KDV': 'DIST_CHILD_CARE',
    'AV1_KDV': 'NUM_1_CHILD_CARE',
    'AV3_KDV': 'NUM_3_CHILD_CARE',
    'AV5_KDV': 'NUM_5_CHILD_CARE',
    'AFS_BRANDW': 'DIST_FIRE_STATIONS',
    'AFS_OPRIT': 'DIST_HIGHWAY_ENTRIES',
    'AFS_TRNOVS': 'DIST_TRAIN_STATIONS',
    'AFS_TREINS': 'DIST_TRAINS',
    'AFS_ATTRAC': 'DIST_ATTRACTIONS',
    'AV10ATTRAC': 'NUM_10_ATTRACTIONS',
    'AV20ATTRAC': 'NUM_20_ATTRACTIONS',
    'AV50ATTRAC': 'NUM_50_ATTRACTIONS',
    'AFS_BIOS': 'DIST_CINEMAS',
    'AV5_BIOS': 'NUM_5_CINEMAS',
    'AV10_BIOS': 'NUM_10_CINEMAS',
    'AV20_BIOS': 'NUM_20_CINEMAS',
    'AFS_MUS': 'DIST_MUSEUMS',
    'AV5_MUS': 'NUM_5_MUSEUMS',
    'AV10_MUS': 'NUM_10_MUSEUMS',
    'AV20_MUS': 'NUM_20_MUSEUMS',
    'AFS_PODIUM': 'DIST_THEATERS',
    'AV5_PODIUM': 'NUM_5_THEATERS',
    'AV10PODIUM': 'NUM_10_THEATERS',
    'AV20PODIUM': 'NUM_20_THEATERS',
    'AFS_BIBLIO': 'DIST_LIBRARIES',
    'AF_IJSBAAN': 'DIST_ICE_RINKS',
    'AFS_POP': 'DIST_PARKS',
    'AFS_SAUNA': 'DIST_SAUNAS',
    'AFS_ZONBNK': 'DIST_SOLARIUMS',
    'AFS_ZWEMB': 'DIST_SWIMMING_POOLS',
    'AFS_ONDBAS': 'DIST_PRIMARY_SCHOOLS',
    'AV1_ONDBAS': 'NUM_1_PRIMARY_SCHOOL',
    'AV3_ONDBAS': 'NUM_3_PRIMARY_SCHOOLS',
    'AV5_ONDBAS': 'NUM_5_PRIMARY_SCHOOLS',
    'AFS_ONDHV': 'DIST_SECONDARY_SCHOOLS',
    'AV3_ONDHV': 'NUM_3_SECONDARY_SCHOOLS',
    'AV5_ONDHV': 'NUM_5_SECONDARY_SCHOOLS',
    'AV10_ONDHV': 'NUM_10_SECONDARY_SCHOOLS',
    'AFS_ONDVMB': 'DIST_VOCATIONAL_SCHOOLS',
    'AV3_ONDVMB': 'NUM_3_VOCATIONAL_SCHOOLS',
    'AV5_ONDVMB': 'NUM_5_VOCATIONAL_SCHOOLS',
    'AV10ONDVMB': 'NUM_10_VOCATIONAL_SCHOOLS',
    'AFS_ONDVRT': 'DIST_HIGH_SCHOOLS',
    'AV3_ONDVRT': 'NUM_3_HIGH_SCHOOLS',
    'AV5_ONDVRT': 'NUM_5_HIGH_SCHOOLS',
    'AV10ONDVRT': 'NUM_10_HIGH_SCHOOLS',
    'AFS_HAPRAK': 'DIST_SNACK_BARS',
    'AV1_HAPRAK': 'NUM_1_SNACK_BAR',
    'AV3_HAPRAK': 'NUM_3_SNACK_BARS',
    'AV5_HAPRAK': 'NUM_5_SNACK_BARS',
    'AFS_ZIEK_E': 'DIST_MEDICAL_CENTERS',
    'AV5_ZIEK_E': 'NUM_5_MEDICAL_CENTERS',
    'AV10ZIEK_E': 'NUM_10_MEDICAL_CENTERS',
    'AV20ZIEK_E': 'NUM_20_MEDICAL_CENTERS',
    'AFS_ZIEK_I': 'DIST_PHARMACIES',
    'AV5_ZIEK_I': 'NUM_5_PHARMACIES',
    'AV10ZIEK_I': 'NUM_10_PHARMACIES',
    'AV20ZIEK_I': 'NUM_20_PHARMACIES',
    'AFS_APOTH': 'DIST_DRUGSTORES',
    'AFS_HAPOST': 'DIST_POST_OFFICES'
}
cbs_grid_data = cbs_grid_data.rename(columns = column_mapper)

# Grid that has no neighborhood name attatched
cbs_keys = cbs_keys.drop(4760, axis=0)
cbs_keys = cbs_keys.reset_index()
cbs_keys = cbs_keys.drop(columns={'index'})

# Create the dataset and order it based on CBS grid codes
the_dataset = pd.merge(cbs_keys[['cbs_grid_code', 'neighborhood_code']], cbs_grid_data, on='cbs_grid_code', how='left')

# Drop columns that have all missing values
the_dataset = the_dataset.fillna('drop').drop_duplicates().reset_index(drop=True)
the_dataset = the_dataset.loc[the_dataset["POPULATION"] != 'drop']
the_dataset = the_dataset.replace('drop', -99997)

yr1718 = moving[moving['time_period'] == '2017-2018']
yr1819 = moving[moving['time_period'] == '2018-2019']
yr1920 = moving[moving['time_period'] == '2019-2020']

y_columns = yr1920.columns.to_list()[6::]

yr1718 = preprocess_moving_data(yr1718, cbs_keys, y_columns, the_dataset)
yr1819 = preprocess_moving_data(yr1819, cbs_keys, y_columns, the_dataset)
yr1920 = preprocess_moving_data(yr1920, cbs_keys, y_columns, the_dataset)

y = create_y(yr1718, yr1819, yr1920, y_columns)

# Keep the numeric values and the neighborhood code for filling missing values
X = the_dataset[[
'neighborhood_code',
'POPULATION',
'POP_0_14',
'POP_15_24', 
'POP_25_44',
'POP_45_64',
'POP_65_PLUS',
'NUM_HOUSEHOLDS',
'AVG_HH_SIZE',
'HOUSES',
'HOUSES_BEFORE_1945',
'HOUSES_1945_1964',
'HOUSES_1965_1974',
'HOUSES_1975_1984',
'HOUSES_1985_1994',
'HOUSES_1995_2004',
'HOUSES_2005_2014',
'HOUSES_2015_LATER',
'P_OWNED_HOUSING',
'P_RENTED_HOUSING',
'AVG_HOUSING_VALUE',
'GAS_CONNECTION',
'ELECTRICITY_CONNECTION',
'INCOME_SOCIAL_SECURITY',
'DIST_SUPERMARKETS',
'DIST_DAYCARE_CENTERS',
'DIST_DEPARTMENT_STORES',
'DIST_CAFES',
'DIST_NIGHTCLUBS',
'DIST_HOTELS',
'DIST_RESTAURANTS',
'DIST_AFTER_SCHOOL_CARE',
'DIST_CHILD_CARE',
'DIST_FIRE_STATIONS',
'DIST_HIGHWAY_ENTRIES',
'DIST_TRAIN_STATIONS',
'DIST_TRAINS',
'DIST_ATTRACTIONS',
'DIST_CINEMAS',
'DIST_MUSEUMS',
'DIST_THEATERS',
'DIST_LIBRARIES',
'DIST_ICE_RINKS',
'DIST_PARKS',
'DIST_SAUNAS',
'DIST_SOLARIUMS',
'DIST_SWIMMING_POOLS',
'DIST_PRIMARY_SCHOOLS',
'DIST_SECONDARY_SCHOOLS',
'DIST_VOCATIONAL_SCHOOLS',
'DIST_HIGH_SCHOOLS',
'DIST_SNACK_BARS',
'DIST_MEDICAL_CENTERS',
'DIST_PHARMACIES',
'DIST_DRUGSTORES',
'DIST_POST_OFFICES'
]]

# Replace strange values
X = fill_values_based_on_neighborhood(-99997, X)

# Save datasets
X.to_csv(f'{PATH_REPO}/data/modelling_X_y/X.csv', index=False)
y.to_csv(f'{PATH_REPO}/data/modelling_X_y/y.csv', index=False)