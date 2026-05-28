#Full: car_forest_tree_tunnel car_urban_day_penno_small_loop
#Half: falcon_indoor_flight_1 falcon_indoor_flight_2 falcon_indoor_flight_3 spot_indoor_obstacles spot_indoor_building_loop

M3ED_HEIGHT = 720
M3ED_WIDTH = 1280

# Indoor sequence falcon_indoor_flight_1 (TH: 2.144660472869873) masked size: 235
# Indoor sequence falcon_indoor_flight_2 (TH: 1.583741307258606) masked size: 233
# Indoor sequence falcon_indoor_flight_3 (TH: 2.224697470664978) masked size: 231
# Indoor sequence spot_indoor_building_loop (TH: 4.733609199523926) masked size: 456
# Indoor sequence spot_indoor_obstacles (TH: 2.3564841747283936) masked size: 340
# Indoor sequence spot_indoor_stairs (TH: 5.231571435928345) masked size: 452
# Total new size for indoor sequences: 1947

# Outdoor night sequence car_urban_night_penno_small_loop (TH: 1.8472122550010681) masked size: 193
# Outdoor night sequence car_urban_night_penno_small_loop_darker (TH: 2.0531070232391357) masked size: 200
# Outdoor night sequence falcon_outdoor_night_high_beams (TH: 4.245989084243774) masked size: 274
# Outdoor night sequence falcon_outdoor_night_penno_parking_1 (TH: 1.8305336236953735) masked size: 465
# Outdoor night sequence spot_outdoor_night_penno_plaza_lights (TH: 2.7551608085632324) masked size: 328
# Outdoor night sequence spot_outdoor_night_penno_short_loop (TH: 2.84389591217041) masked size: 611
# Total new size for outdoor night sequences: 2071

# Outdoor day sequence car_urban_day_city_hall (TH: 1.002073049545288) masked size: 1441
# Outdoor day sequence falcon_outdoor_day_penno_plaza (TH: 2.904557704925537) masked size: 289
# Outdoor day sequence spot_outdoor_day_art_plaza_loop (TH: 3.4601845741271973) masked size: 651
# Total new size for outdoor day sequences: 2381



# ~150 GB Raw Data	
_INDOOR_LIST = [
    'falcon_indoor_flight_1',
    'falcon_indoor_flight_2',
    'falcon_indoor_flight_3',
    'spot_indoor_building_loop',
    'spot_indoor_obstacles',
    'spot_indoor_stairs',
    #'spot_indoor_stairwell' # Error too high
]

# _INDOOR_LIST = [
#     # 'falcon_indoor_flight_1',
#     # 'falcon_indoor_flight_2',
#     # 'falcon_indoor_flight_3',
#     'spot_indoor_building_loop',
#     #'spot_indoor_obstacles',
#     # 'spot_indoor_stairs',
#     #'spot_indoor_stairwell' # Error too high
# ]

# ~160 GB Raw Data (Without car_urban_night_city_hall, car_urban_night_penno_big_loop, car_urban_night_rittenhouse)
_OUTDOOR_NIGHT_LIST = [
    #'car_urban_night_city_hall',
    #'car_urban_night_penno_big_loop',
    'car_urban_night_penno_small_loop',
    'car_urban_night_penno_small_loop_darker',
    #'car_urban_night_rittenhouse',
    #'car_urban_night_ucity_small_loop', # Error too high
    'falcon_outdoor_night_high_beams',
    'falcon_outdoor_night_penno_parking_1',
    #'falcon_outdoor_night_penno_parking_2', # Error too high
    'spot_outdoor_night_penno_plaza_lights',
    'spot_outdoor_night_penno_short_loop',
]

# Around 160 GB Raw Data (subset)
_OUTDOOR_DAY_LIST = [
    'car_urban_day_city_hall',
    # 'car_urban_day_horse',
    # 'car_urban_day_penno_big_loop',
    # 'car_urban_day_penno_small_loop',
    # 'car_urban_day_rittenhouse',
    # 'car_urban_day_ucity_small_loop',
    #'falcon_outdoor_day_fast_flight_1', # Error too high
    # 'falcon_outdoor_day_fast_flight_2',
    # 'falcon_outdoor_day_penno_parking_1',
    # 'falcon_outdoor_day_penno_parking_2',
    'falcon_outdoor_day_penno_plaza',
    # 'falcon_outdoor_day_penno_trees',
    'spot_outdoor_day_art_plaza_loop',
    # 'spot_outdoor_day_penno_short_loop',
    #'spot_outdoor_day_rocky_steps', # Error too high
    # 'spot_outdoor_day_skatepark_1',
    # 'spot_outdoor_day_skatepark_2',
    # 'spot_outdoor_day_srt_green_loop',
    # 'spot_outdoor_day_srt_under_bridge_1',
    # 'spot_outdoor_day_srt_under_bridge_2',
]


DATA_SPLIT = {
    'val_indoor': [f'indoor/{seq}' for seq in _INDOOR_LIST],
    'val_outdoor_day': [f'outdoor_day/{seq}' for seq in _OUTDOOR_DAY_LIST],
    'val_outdoor_night': [f'outdoor_night/{seq}' for seq in _OUTDOOR_NIGHT_LIST],
    'none': [],
}