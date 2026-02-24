CREATE DATABASE IF NOT EXISTS soil_xai;
USE soil_xai;

CREATE TABLE soil_points (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model VARCHAR(30),
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
    cluster INT,
    UNIQUE KEY unique_soil_model (model, lat, lon)
);

CREATE TABLE cluster_explanations (
    model VARCHAR(30),
    cluster INT,
    zone_name VARCHAR(100),
    water_behavior TEXT,
    nutrient_strength TEXT,
    acidity TEXT,
    PRIMARY KEY (model, cluster)
);