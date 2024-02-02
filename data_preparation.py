import pandas as pd
import numpy as np
import geopandas as gpd
import logging
from sklearn.preprocessing import MinMaxScaler
from cleaning_preprocessing_func import *

logging.basicConfig(level=logging.WARNING)

# Path Nick
#PATH_REPO = r'C:/Users/nickb/OneDrive/Documenten/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team4'  

# Path Kian
#PATH_REPO = r'C:\Users\Kianv\Desktop\Breda University\Data Science & Artificial Intelligence\Year 1\2022-23d-1fcmgt-reg-ai-01-group-team4'

# Cristian path
PATH_REPO = r'C:/Users/ASUS/OneDrive/Documents/GitHub/2022-23d-1fcmgt-reg-ai-01-group-team4' 

PATHS_CSV = [
    (f'{PATH_REPO}/data/cleaned_data/CBS_grid_data.csv', ','),
    (f'{PATH_REPO}/data/cleaned_data/CBS_grid_data_fillmean.csv', ','),
    (f'{PATH_REPO}/data/cleaned_data/breda_grid_keys.csv', ','),
    (f'{PATH_REPO}/data/cleaned_data/neighborhood_data.csv', ','),
    (f'{PATH_REPO}/data/cleaned_data/boom_per_buurt.csv', ','),
    (f'{PATH_REPO}/data/raw_data/housing_data/lookup1.csv', ','),
    (f'{PATH_REPO}/data/raw_data/housing_data/lookup2.csv', ','),
    (f'{PATH_REPO}/data/cleaned_data/housing_data/page-9_table-1.csv', ','),
    (f'{PATH_REPO}/data/cleaned_data/housing_data/page-10_table-1.csv', ','),
    (f'{PATH_REPO}/data/cleaned_data/housing_data/page-11_table-1.csv', ','),
    (f'{PATH_REPO}/data/cleaned_data/housing_data/page-12_table-1.csv', ','),
    (f'{PATH_REPO}/data/cleaned_data/housing_data/page-13_table-1.csv', ','),
    (f'{PATH_REPO}/data/cleaned_data/lbm_18.csv', ',')
]

PATHS_GPKG = [
    f'{PATH_REPO}/data/raw_data/moves_grid_breda.gpkg',
]

gdf_move_house = gpd.read_file(PATHS_GPKG[0])

column_mapper_move_house = {'Vestiging': 'Moving to the gridcel from outside Breda',
                       'Vertrek': 'Moving from the gridcel to outside Breda',
                       'Verhuizing binnen gridcel': 'Relocation inside the gridcel',
                       'Verhuizing': 'Relocation inside Breda'}

df_grid, df_grid_fillmean, df_grid_keys, df_neighborhood, df_boom, df_lp1, df_lp2, df_p9, df_p10, df_p11, df_p12, df_p13, df_lbm = load_csvs(PATHS_CSV)

column_mapper_neighborhood = {
    'Buurten': 'neighborhood_name',
    'Werkzame beroepsbevolking': 'employed_labor_force',
    'Beroeps- en niet-beroepsbevolking': 'labor_force_and_non_labor_force',
    'Werknemer met vaste arbeidsrelatie': 'employee_with_permanent_employment',
    'Werknemer met flexibele arbeidsrelatie': 'employee_with_flexible_employment',
    'Zelfstandige': 'self_employed',
    'Huishoudens': 'households',
    'Eenpersoonshuishoudens': 'single_person_households',
    'Meerpersoonshuishoudens zonder kinderen': 'multi_person_households_without_children',
    'Meerpersoonshuishoudens met kinderen': 'multi_person_households_with_children',
    'Huishoudensgrootte': 'household_size',
    'Personen met een laag inkomen': 'persons_with_low_income',
    'Personen met een hoog inkomen': 'persons_with_high_income',
    'Inkomen per inkomensontvanger': 'income_per_income_recipient',
    'Huishoudens onder of rond sociaal minimum': 'households_below_or_around_social_minimum',
    'Kindplaatsen kinderopvang': 'childcare_places',
    'Locaties kinderopvang': 'childcare_locations',
    'Totaalscore Leefbaarometer 3.0': 'total_score_Leefbaarometer_3.0',
    'Afwijking totaalscore': 'deviation_total_score',
    'Score dimensie fysieke omgeving (afwijkin andelijk_gemiddelde)': 'score_physical_environment',
    'Score dimensie overlast & onveiligheid (afwijking landelijk gemiddelde)': 'score_nuisance_&_insecurity',
    'Score dimensie sociale samenhang (afwijking landelijk gemiddelde)': 'score_social_cohesion',
    'Score dimensie voorzieningen (afwijking landelijk gemiddelde)': 'score_amenities',
    'Score dimensie woningvoorraad (afwijking landelijk gemiddelde)': 'score_housing_stock',
    'Personen met een Werkloosheidsuitkering vanaf 15 jaar': 'persons_receiving_unemployment_benefits_from_15_years',
    'Personen met een Bijstandsuitkering vanaf 15 jaar': 'persons_receiving_social_assistance_from_15_years',
    'Personen met een AOW uitkering vanaf 15 jaar': 'persons_receiving_old_age_pension_from_15_years',
    'Opleidingsniveau 15-75-jarigen - Laag': 'education_level_15_75_years_low',
    'Opleidingsniveau 15-75-jarigen - Midden': 'education_level_15_75_years_medium',
    'Opleidingsniveau 15-75-jarigen - Hoog': 'education_level_15_75_years_high',
    'Personen met een migratieachtergrond': 'persons_with_a_migration_background',
    'Personen met een westerse achtergrond': 'persons_with_a_Western_background',
    'Personen met een niet westerse achtergrond': 'persons_with_a_non_Western_background',
    'Misdrijven totaal': 'total_crimes',
    'Vernielingen en misdrijven tegen openbare orde en gezag - totaal': 'vandalism_and_crimes_against_public_order_and_authority_total',
    'Gewelds- en seksuele misdrijven - totaal': 'violent_and_sexual_crimes_total',
    'Vermogensmisdrijven - totaal': 'property_crimes_total',
    'Bevolkingsdichtheid': 'population_density'
}

column_mapper_grid = {
    'c28992r100': 'cbs_grid_code',
    'INWONER': 'residents',
    'MAN': 'male',
    'VROUW': 'female',
    'INW_014': 'population_0_14',
    'INW_1524': 'population_15_24',
    'INW_2544': 'population_25_44',
    'INW_4564': 'population_45_64',
    'INW_65PL': 'population_65_plus',
    'P_NL_ACHTG': 'dutch_origin_percentage',
    'AANTAL_HH': 'number_of_households',
    'TOTHH_EENP': 'total_single_person_households',
    'TOTHH_MPZK': 'total_multi_person_households',
    'HH_TWEEOUD': 'households_with_two_parents',
    'GEM_HH_GR': 'average_household_size',
    'WONING': 'dwellings',
    'P_KOOPWON': 'owner_occupied_percentage',
    'WOZWONING': 'dwellings_average_value',
    'G_GAS_WON': 'dwellings_gas',
    'G_ELEK_WON': 'dwellings_electricity',
    'M_INKHH': 'average_income_per_household',
    'AFS_SUPERM': 'average_distance_supermarket',
    'AV1_SUPERM': 'amount_within_1km_supermarket',
    'AV3_SUPERM': 'amount_within_3km_supermarket',
    'AV5_SUPERM': 'amount_within_5km_supermarket',
    'AFS_DAGLMD': 'average_distance_daily_needs',
    'AV1_DAGLMD': 'amount_within_1km_daily_needs',
    'AV3_DAGLMD': 'amount_within_3km_daily_needs',
    'AV5_DAGLMD': 'amount_within_5km_daily_needs',
    'AFS_WARENH': 'average_distance_department_store',
    'AV5_WARENH': 'amount_within_5km_department_store',
    'AV10WARENH': 'amount_within_10km_department_store',
    'AV20WARENH': 'amount_within_20km_department_store',
    'AFS_CAFE': 'average_distance_cafe',
    'AV1_CAFE': 'amount_within_1km_cafe',
    'AV3_CAFE': 'amount_within_3km_cafe',
    'AV5_CAFE': 'amount_within_5km_cafe',
    'AFS_CAFTAR': 'average_distance_cafetaria',
    'AV1_CAFTAR': 'amount_within_1km_cafetaria',
    'AV3_CAFTAR': 'amount_within_3km_cafetaria',
    'AV5_CAFTAR': 'amount_within_5km_cafetaria',
    'AFS_HOTEL': 'average_distance_hotel',
    'AV5_HOTEL': 'amount_within_5km_hotel',
    'AV10_HOTEL': 'amount_within_10km_hotel',
    'AV20_HOTEL': 'amount_within_20km_hotel',
    'AFS_RESTAU': 'average_distance_restaurant',
    'AV1_RESTAU': 'amount_within_1km_restaurant',
    'AV3_RESTAU': 'amount_within_3km_restaurant',
    'AV5_RESTAU': 'amount_within_5km_restaurant',
    'AFS_BSO': 'average_distance_BSO',
    'AV1_BSO': 'amount_within_1km_BSO',
    'AV3_BSO': 'amount_within_3km_BSO',
    'AV5_BSO': 'amount_within_5km_BSO',
    'AFS_KDV': 'average_distance_KDV',
    'AV1_KDV': 'amount_within_1km_KDV',
    'AV3_KDV': 'amount_within_3km_KDV',
    'AV5_KDV': 'amount_within_5km_KDV',
    'AFS_BRANDW': 'average_distance_fire_station',
    'AFS_OPRIT': 'average_distance_motorway_entry',
    'AFS_TRNOVS': 'average_distance_ov_tranfer',
    'AFS_TREINS': 'average_distance_train_stops',
    'AFS_ATTRAC': 'average_distance_attractions',
    'AV10ATTRAC': 'amount_within_10km_attractions',
    'AV20ATTRAC': 'amount_within_20km_attractions',
    'AV50ATTRAC': 'amount_within_50km_attractions',
    'AFS_BIOS': 'average_distance_cinema',
    'AV5_BIOS': 'amount_within_5km_cinema',
    'AV10_BIOS': 'amount_within_10km_cinema',
    'AV20_BIOS': 'amount_within_20km_cinema',
    'AFS_MUS': 'average_distance_museum',
    'AV5_MUS': 'amount_within_5km_museum',
    'AV10_MUS': 'amount_within_10km_museum',
    'AV20_MUS': 'amount_within_20km_museum',
    'AFS_PODIUM': 'average_distance_theater',
    'AV5_PODIUM': 'amount_within_5km_theater',
    'AV10PODIUM': 'amount_within_10km_theater',
    'AV20PODIUM': 'amount_within_20km_theater',
    'AFS_BIBLIO': 'average_distance_library',
    'AF_IJSBAAN': 'average_distance_ice_skating_rink',
    'AFS_POP': 'average_distance_pop_music_venue',
    'AFS_SAUNA': 'average_distance_sauna',
    'AFS_ZONBNK': 'average_distance_solarium',
    'AFS_ZWEMB': 'average_distance_swimming_pool',
    'AFS_ONDBAS': 'average_distance_primary_school',
    'AV1_ONDBAS': 'amount_within_1km_primary_school',
    'AV3_ONDBAS': 'amount_within_3km_primary_school',
    'AV5_ONDBAS': 'amount_within_5km_primary_school',
    'AFS_ONDHV': 'average_distance_high_school',
    'AV3_ONDHV': 'amount_within_3km_high_school',
    'AV5_ONDHV': 'amount_within_5km_high_school',
    'AV10_ONDHV': 'amount_within_10km_high_school',
    'AFS_ONDVMB': 'average_distance_vocational_school',
    'AV3_ONDVMB': 'amount_within_3km_vocational_school',
    'AV5_ONDVMB': 'amount_within_5km_vocational_school',
    'AV10ONDVMB': 'amount_within_10km_vocational_school',
    'AFS_ONDVRT': 'average_distance_HAVO_VWO',
    'AV3_ONDVRT': 'amount_within_3km_HAVO_VWO',
    'AV5_ONDVRT': 'amount_within_5km_HAVO_VWO',
    'AV10ONDVRT': 'amount_within_10km_HAVO_VWO',
    'AFS_HAPRAK': 'average_distance_general_practitioner',
    'AV1_HAPRAK': 'amount_within_1km_general_practitioner',
    'AV3_HAPRAK': 'amount_within_3km_general_practitioner',
    'AV5_HAPRAK': 'amount_within_5km_general_practitioner',
    'AFS_ZIEK_E': 'average_distance_hospital',
    'AV5_ZIEK_E': 'amount_within_5km_hospital',
    'AV10ZIEK_E': 'amount_within_10km_hospital',
    'AV20ZIEK_E': 'amount_within_20km_hospital',
    'AFS_ZIEK_I': 'average_distance_hospital_outpatient_clinic',
    'AV5_ZIEK_I': 'amount_within_5km_hospital_outpatient_clinic',
    'AV10ZIEK_I': 'amount_within_10km_hospital_outpatient_clinic',
    'AV20ZIEK_I': 'amount_within_20km_hospital_outpatient_clinic',
    'AFS_APOTH': 'average_distance_pharmacy',
    'AFS_HAPOST': 'average_distance_general_practioner'
}

column_mapper_boom = {
    'buurt_opp_zonderagr':'neighbourhood_surface_without_agr',
    'perc_bebouwing_totaal':'perc_buildings_total',
    'perc_openbaar_totaal':'perc_public_total',
    'perc_overig_totaal':'perc_other_total',
    'perc_privaat_totaal':'perc_private_total',
    'perc_transitie_totaal':'perc_transition_total',
    'perc_water_totaal':'perc_water_total',
    'perc_privaat_boom':'perc_private_tree',
    'perc_privaat_groen':'perc_private_green',
    'perc_privaat_niet_groen':'perc_private_not_green',
    'perc_bebouwing_boom':'perc_building_tree',
    'perc_bebouwing_groen':'perc_building_green',
    'perc_bebouwing_niet_groen':'perc_buildings_not_green',
    'perc_openbaar_boom':'perc_public_tree',
    'perc_openbaar_groen':'perc_public_green',
    'perc_openbaar_niet_groen':'perc_public_not_green',
    'perc_overig_boom':'perc_other_tree',
    'perc_overig_groen':'perc_other_green',
    'perc_overig_niet_groen':'perc_other_not_green',
    'perc_transitie_boom':'perc_transition_tree',
    'perc_transitie_groen':'perc_transition_green',
    'perc_transitie_niet_groen':'perc_transition_not_green',
    'perc_water':'perc_water',
    'perc_water_boom':'perc_water_tree',
    'buurt_opp_incl_agrarisch':'neighbourhood_opp_incl_agricultural',
    'openbaar_m2_groen':'public_m2_green',
    'openbaar_oppervlakte':'public_surface',
    'openbaar_m2_nietgroen':'public_m2_not_green',
    'privaat_m2_groen':'private_m2_green',
    'privaat_oppervlakte':'private_area',
    'privaat_m2_nietgroen':'private_m2_not_green',
    'bebouwing_m2_groen':'building_m2_green',
    'bebouwing_oppervlakte':'building_area',
    'bebouwing_m2_nietgroen':'building_m2_not_green',
    'water_oppervlakte':'water_surface',
    'agrarisch_m2_groen':'agricultural_m2_green',
    'agrarisch_oppervlakte':'agricultural_area',
    'agrarisch_m2_nietgroen':'agricultural_m2_not_green',
    'overig_m2_groen':'other_m2_green',
    'overig_oppervlakte':'other_area',
    'overig_m2_nietgroen':'other_m2_not_green',
    'transitie_m2_groen':'transition_m2_green',
    'transitie_oppervlakte':'transition_surface',
    'transitie_m2_nietgroen':'transition_m2_not_green',
    'boom_openbaar_oppervlakte':'tree_public_surface',
    'boom_privaat_oppervlakte':'tree_private_surface',
    'boom_bebouwing_oppervlakte':'tree_building_surface',
    'boom_water_oppervlakte':'tree_water_surface',
    'boom_agrarisch_oppervlakte':'tree_agricultural_surface',
    'boom_overig_oppervlakte':'tree_other_surface',
    'boom_transitie_oppervlakte':'tree_transition_surface',
    'percentagegroen':'percentage_green',
    'percentagelaaggroen':'percentage_low_green',
    'percentageboom':'percentage_tree',
    'percentagegrijs':'percentage_grey',
    'sted':'town',
    'bev_dichth':'bev_density',
    'aant_inw':'number_inhabit',
    'bu_naam':'neighborhood_name_drop',
    'groenbinnenopenbareruimte':'green_indoor_public_space',
    'groenbinnennietopenbareruimte':'greeninside_non-public_space'
}

column_mapper_lp = {
    'Buurt': 'neighborhood_id',
    'Unnamed: 1': 'neighborhood_name',
    'Wijk': 'district_id',
    }

column_mapper_p9 = {
    'Buurtcode': 'neighborhood_id',
    'Buurtnaam': 'neighborhood_name',
    "Eengezins": "single-family",
    "Meergezins": "multi-family",
    "Totaal": "total",
    "Eengezins.1": "single-family_percentage",
    "Meergezins.1": "multi-family_percentage",
    "Totaal.1": "total_percentage"
}

column_mapper_p10 = {
    'Buurtcode': 'neighborhood_id',
    'Buurtnaam': 'neighborhood_name',
    "Huur": "rent",
    "Koop": "buy",
    "Totaal": "total",
    "Huur.1": "rent_percentage",
    "Koop.1": "buy_percentage",
    "Totaal.1": "total_percentage"
}

column_mapper_p11 = {
    'Buurtcode': 'neighborhood_id',
    'Buurtnaam': 'neighborhood_name',
    'voor 1906': 'before_1906',
    '1906 t/m 1945': '1906_to_1945',
    '1946 t/m 1959': '1946_to_1959',
    '1960 t/m 1969': '1960_to_1969',
    '1970 t/m 1979': '1970_to_1979',
    '1980 t/m 1989': '1980_to_1989',
    '1990 t/m 1999': '1990_to_1999',
    '2000 t/m 2009': '2000_to_2009',
    '2010 t/m 2019': '2010_to_2019',
    'Totaal': 'total'
}

column_mapper_p12 = {
    'Buurtcode': 'neighborhood_id',
    'Buurtnaam': 'neighborhood_name',    
    'Vrijstaand': 'detached',
    'Twee onder één kap': 'semi_detached',
    'Rij-en hoekwoningen': 'row_and_corner_houses',
    'Boven beneden- portiekwoningen': 'upstairs_downstairs_apartments',
    'Flats en appartementen': 'flats_and_apartments',
    'Recreatie, studenten en overige woningen': 'recreational_student_other_dwellings',
    'Woonwagen- voorziening': 'caravan_accommodation',
    'Agrarische woningen': 'agricultural_dwellings',
    'Overige bedrijfs- woningen': 'other_business_dwellings',
    'Overige woningen': 'other_dwellings',
    'Nader te bepalen': 'to_be_determined',
    'Totaal': 'total'
}

column_mapper_p13 = {
    'Buurtcode': 'neighborhood_id',
    'Buurtnaam': 'neighborhood_name',    
    'Sociaal Minder dan €195.100': 'social_less_than_€195,100',
    'Middelduur laag €195.100 tot €225.000': 'medium_low_priced_€195,100_to_€225,000',
    'Middelduur hoog €225.000 tot €300.000': 'medium_high_priced_€225,000_to_€300,000',
    'Duur laag €300.000 tot €450.000': 'expensive_low_priced_€300,000_to_€450,000',
    'Duur hoog €450.000 of meer': 'expensive_high_priced_€450,000_or_more',
    'onbekend': 'unknown',
    'Totaal': 'total'
}

df_neighborhood = replace_value(lower_string(df_neighborhood.rename(columns=column_mapper_neighborhood), exclude=['cbs_grid_code']), 'chass?©', 'chasse')
df_grid = lower_string(df_grid.rename(columns=column_mapper_grid), exclude=['cbs_grid_code'])
df_grid_fillmean = lower_string(df_grid_fillmean.rename(columns=column_mapper_grid), exclude=['cbs_grid_code'])
df_grid_keys = lower_string(df_grid_keys, exclude=['cbs_grid_code', 'neighborhood_code'])
df_boom = lower_string(df_boom.rename(columns=column_mapper_boom), exclude=['cbs_grid_code'])

df_p9 = replace_value(lower_string(df_p9.rename(columns=column_mapper_p9)), 'chassé', 'chasse').drop(['total_percentage', 'neighborhood_id'], axis=1)
df_p9 = replace_value(df_p9, '%', '', replace_str_to_int=True)

df_p10 = replace_value(lower_string(df_p10.rename(columns=column_mapper_p10)), 'chassé', 'chasse').drop(['total_percentage', 'neighborhood_id'], axis=1)
df_p10 = replace_value(df_p10, '%', '', replace_str_to_int=True)

df_p11 = replace_value(lower_string(df_p11.rename(columns=column_mapper_p11)), 'chassé', 'chasse').drop(['neighborhood_id'], axis=1)
df_p12 = replace_value(lower_string(df_p12.rename(columns=column_mapper_p12)), 'chassé', 'chasse').drop(['neighborhood_id'], axis=1)
df_p13 = replace_value(lower_string(df_p13.rename(columns=column_mapper_p13)), 'chassé', 'chasse').drop(['neighborhood_id'], axis=1)


## KIAN'S PREPROCESSING FOR DASHBAORD
df_lbm = df_lbm[df_lbm['grid_id'].isin(df_grid_keys['cbs_grid_code'])]

scaler = MinMaxScaler()

vrz_norm = scaler.fit_transform(df_lbm['vrz'].values.reshape(-1, 1))
df_lbm['vrz_norm'] = vrz_norm

onv_norm = scaler.fit_transform(df_lbm['onv'].values.reshape(-1, 1))
df_lbm['onv_norm'] = onv_norm

df_lbm.to_csv(f'{PATH_REPO}/data/cleaned_data/lbm_18.csv', index=False)


# print(f'check_data_pd:df_neighborhood:{check_data_pd(df_neighborhood)}')
# print(f'check_data_pd:df_grid_keys:{check_data_pd(df_grid_keys)}')
# print(f'check_data_pd:df_grid:{check_data_pd(df_grid)}')
# print(f'check_data_pd:df_grid_fillmean:{check_data_pd(df_grid_fillmean)}')
# print(f'check_data_pd:df_boom:{check_data_pd(df_boom)}')
# print(f'check_data_pd:df_lp1:{check_data_pd(df_lp1)}')
# print(f'check_data_pd:df_lp2:{check_data_pd(df_lp2)}')
# print(f'check_data_pd:df_p9:{check_data_pd(df_p9)}')
# print(f'check_data_pd:df_p10:{check_data_pd(df_p10)}')
# print(f'check_data_pd:df_p11:{check_data_pd(df_p11)}')
# print(f'check_data_pd:df_p12:{check_data_pd(df_p12)}')
# print(f'check_data_pd:df_p13:{check_data_pd(df_p13)}')
# print(f'check_data_pd:gdf_move_house:{check_data_pd(gdf_move_house)}')

# print(find_difference(df_neighborhood['neighborhood_name'], df_grid_keys['neighborhood_name']))
# print(find_difference(df_p11['neighborhood_name'], df_grid_keys['neighborhood_name']))

df_neighborhood = pd.merge(df_neighborhood, df_grid_keys, on='neighborhood_name', 
                           how='left', suffixes=['', '_drop']).drop_duplicates('neighborhood_name')

df_grid = pd.merge(df_grid, df_grid_keys, on='cbs_grid_code', 
                   how='left', suffixes=['', '_drop'])

df_grid_fillmean = pd.merge(df_grid_fillmean, df_grid_keys, on='cbs_grid_code', 
                            how='left', suffixes=['', '_drop'])

df_lp = pd.concat([df_lp1, df_lp2], axis=0)
df_lp = lower_string(
    df_lp.rename(columns=column_mapper_lp),
    ).drop(['neighborhood_id', 'district_id'], axis=1)

df_boom = df_boom[df_boom.columns[~df_boom.columns.str.endswith('_drop')]]

df_grid_keys = pd.merge(df_grid_keys, df_lp, on='neighborhood_name', 
                        how='left')

df_housing = pd.merge(df_p9, df_p10, on='neighborhood_name', 
                      how='left', suffixes=['', '_drop'])
df_housing = pd.merge(df_housing, df_p11, on='neighborhood_name', 
                      how='left', suffixes=['', '_drop'])
df_housing = pd.merge(df_housing, df_p12, on='neighborhood_name', 
                      how='left', suffixes=['', '_drop'])
df_housing = pd.merge(df_housing, df_p13, on='neighborhood_name', 
                      how='left', suffixes=['', '_drop'])
df_housing = lower_string(df_housing)

# Change the data type
gdf_move_house['time_period'] = gdf_move_house['time_period'].astype(str)

# Drop unnecesary values
gdf_move_house.drop(gdf_move_house[gdf_move_house['time_period'] == '20132014'].index, inplace = True)
gdf_move_house.drop(gdf_move_house[gdf_move_house['time_period'] == '20142015'].index, inplace = True)
gdf_move_house.drop(gdf_move_house[gdf_move_house['time_period'] == '20152016'].index, inplace = True)
gdf_move_house.drop(gdf_move_house[gdf_move_house['time_period'] == '20152016'].index, inplace = True)
gdf_move_house.drop(gdf_move_house[gdf_move_house['time_period'] == '20162017'].index, inplace = True)
gdf_move_house.drop(gdf_move_house[gdf_move_house['moving_type'] == 'Relocation inside the gridcel'].index, inplace = True)

# Restructure the time_period values
gdf_move_house['time_period'] = gdf_move_house['time_period'].str[:4] + '-' + gdf_move_house['time_period'].str[4:]

# Create column about people moving inside a gridcell
gdf_move_house['moving_inside'] = gdf_move_house[gdf_move_house['moving_type']=='Moving to the gridcel from outside Breda'].iloc[:,3]
gdf_move_house['moving_inside'] = gdf_move_house['moving_inside'].fillna("0") 
gdf_move_house['moving_inside'] = gdf_move_house['moving_inside'].astype(int)

# Create column about people moving from a gridcell to another in Breda
gdf_move_house['moving_grid_in_breda'] = gdf_move_house[gdf_move_house['moving_type'] == 'Relocation inside Breda'].iloc[:,3]
gdf_move_house['moving_grid_in_breda'] = gdf_move_house['moving_grid_in_breda'].fillna("0")
gdf_move_house['moving_grid_in_breda'] = gdf_move_house['moving_grid_in_breda'].astype(int)

# Create column about people moving from a gridcell to outside of Breda
gdf_move_house['moving_grid_outside_breda'] = gdf_move_house[gdf_move_house['moving_type'] == 'Moving from the gridcel to outside Breda'].iloc[:,3]
gdf_move_house['moving_grid_outside_breda'] = gdf_move_house['moving_grid_outside_breda'].fillna("0")
gdf_move_house['moving_grid_outside_breda'] = gdf_move_house['moving_grid_outside_breda'].astype(int)

# Crate a column with the total number of people leaving a gridcell
gdf_move_house['moving_outside'] = gdf_move_house['moving_grid_in_breda'] + gdf_move_house['moving_grid_outside_breda']

# Drop unnecesary columns
gdf_move_house.drop(columns=['moving_grid_in_breda', 'moving_grid_outside_breda'], inplace=True)

# Merge to have the moving data synced with the CBS and neighborhood codes
moving = pd.merge(gdf_move_house, df_grid_keys[['cbs_grid_code', 'neighborhood_code', 'neighborhood_name']], on='cbs_grid_code')

# Prepare data to order it based on years for the lookup table
yr1718 = moving[moving['time_period'] == '2017-2018']
yr1819 = moving[moving['time_period'] == '2018-2019']
yr1920 = moving[moving['time_period'] == '2019-2020']

df1718 = yr1718.groupby('cbs_grid_code')['moving_outside'].sum().reset_index()
df1718['time_period']='2017-2018'
df1819 = yr1819.groupby('cbs_grid_code')['moving_outside'].sum().reset_index()
df1819['time_period']='2018-2019'
df1920 = yr1920.groupby('cbs_grid_code')['moving_outside'].sum().reset_index()
df1920['time_period']='2019-2020'

# Crate a lookup table
lookup = pd.concat([df1718, df1819, df1920], ignore_index=True, axis=0)

# Create temporary index column
moving['time_and_grid_code'] = moving['time_period'] + " " + moving['cbs_grid_code']
lookup['time_and_grid_code'] = lookup['time_period'] + " " + lookup['cbs_grid_code']

# Merge the data with the lookup table
moving_in_out = pd.merge(moving, lookup[['moving_outside', 'time_and_grid_code']], on='time_and_grid_code')
moving_in_out = moving_in_out.replace('Relocation inside Breda', 'Moving from the gridcel to outside Breda')
moving_in_out.drop_duplicates(inplace=True)
moving_in_out.drop(columns=['moving_outside_x', 'moving_type', 'frequency'], inplace=True)
moving_in_out.rename(columns = {'moving_outside_y':'moving_outside_gridcell', 'moving_inside':'moving_inside_gridcell'}, inplace=True)

# Pivot the data 
pivot = pd.pivot_table(moving, values='frequency', index = 'time_and_grid_code', columns = 'moving_type', fill_value=0)
# Translate columns
pivot = pivot.rename(columns=column_mapper_move_house)

# Merge data on the temporary index
moving_processed = pd.merge(moving_in_out, pivot, on = 'time_and_grid_code')
moving_processed.drop_duplicates(inplace=True)

# Modify columns and drop unnecesary ones
moving_in_out.rename(columns = {'moving_outside_y':'moving_outside_gridcell', 'moving_inside':'moving_inside_gridcell'}, inplace=True)
moving_processed.drop(columns=['moving_inside_gridcell', 'moving_outside_gridcell'], inplace=True)
moving_processed.drop(columns=['time_and_grid_code'], inplace=True)
moving_processed.rename(columns = {'Moving from the gridcel to outside Breda':'leaving_gridcell_outside_breda', 
                                   'Relocation inside Breda':'leaving_gridcell_in_breda',
                                   'Moving to the gridcel from outside Breda':'moving_inside_gridcell'}, inplace=True)
moving_processed['leaving_gridcell'] = moving_processed['leaving_gridcell_outside_breda'] + moving_processed['leaving_gridcell_in_breda']

# Reorder columns
moving_processed = moving_processed[[
 'time_period',
 'cbs_grid_code',
 'neighborhood_code',
 'neighborhood_name',
 'geometry',
 'moving_inside_gridcell',
 'leaving_gridcell',
 'leaving_gridcell_outside_breda',
 'leaving_gridcell_in_breda',
 ]]

# print(f'check_data_pd:df_housing:{check_data_pd(df_housing)}')
# print(f'check_data_pd:df_neighborhood:{check_data_pd(df_neighborhood)}')an
# print(f'check_data_pd:df_grid_keys:{check_data_pd(df_grid_keys)}')
# print(f'check_data_pd:df_grid:{check_data_pd(df_grid)}')
# print(f'check_data_pd:df_grid_fillmean:{check_data_pd(df_grid_fillmean)}')
# print(f'check_data_pd:df_boom:{check_data_pd(df_boom)}')
# print(f'check_data_pd:df_lp:{check_data_pd(df_lp)}')
print(f'check_data_pd:gdf_move_house:{check_data_pd(gdf_move_house)}')
print(df_boom['neighborhood_code'])
print(df_neighborhood['neighborhood_code'])
print(df_grid['cbs_grid_code'])

df_grid.to_csv(f'{PATH_REPO}/data/preprocessed_data/CBS_grid_data.csv', index=False)
df_grid_keys.to_csv(f'{PATH_REPO}/data/preprocessed_data/breda_grid_keys.csv', index=False)
df_neighborhood.to_csv(f'{PATH_REPO}/data/preprocessed_data/neighborhood_data.csv', index=False)
df_boom.to_csv(f'{PATH_REPO}/data/preprocessed_data/boom_per_buurt.csv', index=False)
df_housing.to_csv(f'{PATH_REPO}/data/preprocessed_data/housing.csv', index=False)
moving_processed.to_csv(f'{PATH_REPO}/data/preprocessed_data/moving_data_processed.csv', index=False)

# python data_preparation.py