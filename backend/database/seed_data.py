from pathlib import Path
import pandas as pd
from .db import get_connection


CSV_PATH = Path(r"C:\Users\chama\Weather-Prediction-and-Crop-Recommendation-System-\soil_suitability\outputs\soil_features_for_db_final.csv")


def seed_soil_points(cursor):
    df = pd.read_csv(CSV_PATH)

    insert_query = """
    INSERT INTO soil_points
    (lat, lon, taw, organic_carbon, cec, ph, sand_pct, bulk_density, awc, texture_class, cluster)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    for _, row in df.iterrows():
        cursor.execute(insert_query, (
            row["lat"],
            row["lon"],
            row["taw"],
            row["organic_carbon"],
            row["cec"],
            row["ph"],
            row["sand_pct"],
            row["bulk_density"],
            row["awc"],
            row["texture_class"],
            row["cluster"]
        ))

    print("Soil points inserted successfully.")


def seed_cluster_explanations(cursor):

    cluster_data = [
        (
            0,
            "Moisture Stable Sandy Loam",
            """Water Behavior: This soil acts like a giant sponge. Even though it feels a bit sandy, it has the best ability in the region to hold onto moisture. It won't dry out as fast as other soils during a sunny week.

Nutrient Strength: It has a decent battery for holding nutrients, but it is currently low on natural compost (organic matter). It's a clean slate that needs regular feeding.

Acidity: This soil is only mildly sour (acidic). It's in a comfortable middle ground where it isn't too harsh on plants.
"""
        ),
        (
            1,
            "Highly Leached Acidic Sand",
            """Water Behavior: This soil is very leaky. Because it is so sandy and weak, water runs right through it, often taking nutrients down with it before plants can reach them.

Nutrient Strength: This is the thinnest soil in the area. It has a very low capacity to store plant food and is low in natural richness.

Acidity: This soil is very sour (highly acidic). This acidity can actually be toxic to some plants and prevents them from growing deep roots.
"""
        ),
        (
            2,
            "Fast Draining Sandy Soil",
            """Water Behavior: This is the sandiest soil in this province. It dries out very quickly because it has almost no way to trap water. It will never stay soggy, but it will get thirsty within a few hours of hot sun.

Nutrient Strength: Surprisingly, it has a decent capacity to hold nutrients, but it lacks the organic bulk to keep them stable.

Acidity: This is the least sour soil in the group. While still slightly acidic, it's much sweeter than the other areas, making it easier to manage.
"""
        ),
        (
            3,
            "Nutrient Dense Acidic Soil",
            """Water Behavior: This is a heavy and fine textured soil. It doesn't have much sand, so it holds water well and stays damp for a long time.

Nutrient Strength: This is the most fertile soil in this province. It has the highest amount of natural compost and the strongest battery for storing fertilizers and minerals.

Acidity: Even though it is rich, it is very sour (acidic). This high acidity can lock up the nutrients, making it hard for plants to actually eat the food that is sitting in the soil.
"""
        )
    ]

    upsert_query = """
    INSERT INTO cluster_explanations (cluster, zone_name, zone_description)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE
        zone_name = VALUES(zone_name),
        zone_description = VALUES(zone_description)
    """

    for row in cluster_data:
        cursor.execute(upsert_query, row)

    print("Cluster explanations inserted successfully.")


def main():
    conn = get_connection()
    cursor = conn.cursor()

    seed_soil_points(cursor)
    seed_cluster_explanations(cursor)

    conn.commit()
    cursor.close()
    conn.close()

    print("Database seeding completed successfully.")


if __name__ == "__main__":
    main()

