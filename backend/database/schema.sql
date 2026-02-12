CREATE TABLE soil_points (
    id INT AUTO_INCREMENT PRIMARY KEY,
    lat DOUBLE,
    lon DOUBLE,
    taw FLOAT,
    organic_carbon FLOAT,
    cec FLOAT,
    ph FLOAT,
    sand_pct FLOAT,
    bulk_density FLOAT,
    awc FLOAT,
    texture_class VARCHAR(20),
    cluster INT
);

CREATE TABLE cluster_explanations (
    cluster INT PRIMARY KEY,
    zone_name VARCHAR(100),
    zone_description TEXT
);