from math import radians, sin, cos, sqrt, atan2, tan

def lateral_deviation(from_lat, from_lon, to_lat, to_lon, true_psi):
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * cos(radians(to_lat))
    d_lat = from_lat - to_lat
    d_lon = from_lon - to_lon
    north = d_lat * meters_per_deg_lat
    east = d_lon * meters_per_deg_lon
    truepsi_rad = radians(true_psi)
    truepsi_east = sin(truepsi_rad)
    truepsi_north = cos(truepsi_rad)
    latdev_truepsi = east * truepsi_north - north * truepsi_east
    return latdev_truepsi

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)

    a = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def vertical_deviation(from_lat, from_lon, to_lat, to_lon, aircraft_elev, runway_elev, glide_slope_deg=3.0):
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * cos(radians(to_lat))
    d_lat = from_lat - to_lat
    d_lon = from_lon - to_lon
    north = d_lat * meters_per_deg_lat
    east = d_lon * meters_per_deg_lon
    ground_dist = (north ** 2 + east ** 2) ** 0.5
    glide_slope_rad = radians(glide_slope_deg)
    ideal_alt = tan(glide_slope_rad) * ground_dist + runway_elev
    vert_dev = aircraft_elev - ideal_alt
    return vert_dev

def draw_window_callback(inWindowID, inRefcon, debug_messages, episode_count, xp):
    try:
        left, top, right, bottom = xp.getWindowGeometry(inWindowID)
        xp.drawTranslucentDarkBox(left, top, right, bottom)

        color = (1.0, 1.0, 1.0)
        y_pos = top - 15

        xp.drawString(color, left + 5, y_pos, f"Episodes: {episode_count}", 0, xp.Font_Basic)
        y_pos -= 20

        # Debug messages
        for message in debug_messages[-16:]:
            xp.drawString(color, left + 5, y_pos, message, 0, xp.Font_Basic)
            y_pos -= 12
    except Exception as e:
        pass
