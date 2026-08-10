CONF = {
    "cookies_colors": {
        "Verde": {"min": [43, 49, 180], "max": [80, 255, 255]},
        "Rojo": {"min": [129, 92, 137], "max": [179, 255, 255]},
        "Amarillo": {"min": [6, 193, 226], "max": [33, 255, 255]},
    },
    # Color de la mira/cursor (cyan brillante H=106-108, S=217-245, V=255)
    "mira_color": {"min": [100, 200, 250], "max": [115, 255, 255]},
    "game_area": {"x_min": 350, "x_max": 940, "y_min": 310, "y_max": 892},
    "min_area_cookie": 350,
    "images_path" : "./imgs/",
    "detection": {
        "merge_distance": 30,
        "neighbor_distance": 108,
        "axis_tolerance": 28,
    },
    "game_area_border":{
        "color": (9, 255, 0),
        "thickness": 5
    }
}
