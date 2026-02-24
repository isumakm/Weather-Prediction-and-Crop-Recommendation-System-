from pathlib import Path
import pandas as pd
from .db import get_connection

CSV_PATH = Path(
    r"C:\Users\chama\Weather-Prediction-and-Crop-Recommendation-System-\soil_suitability\outputs\soil_features_for_db_final_v2.csv"
)

# Map of model name
MODEL_CLUSTER_MAP = {
    "kmeans": "cluster_kmeans",
    "agglomerative": "cluster_agg",
    "gmm": "cluster_gmm"
}


def seed_soil_points(cursor):
    df = pd.read_csv(CSV_PATH)

    insert_query = """
    INSERT INTO soil_points
    (model, lat, lon, taw, organic_carbon, cec, ph, sand_pct, bulk_density, awc, texture_class, cluster)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        cluster = VALUES(cluster)
    """

    for _, row in df.iterrows():
        for model, cluster_col in MODEL_CLUSTER_MAP.items():
            cursor.execute(insert_query, (
                model,
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
                int(row[cluster_col])
            ))

    print("Soil points inserted successfully.")


def seed_cluster_explanations(cursor):
    cluster_data = [
        (
            "kmeans", 0,
            "Moisture Stable Low Organic Soil",
            "This soil has the highest water holding ability among all four groups. It stays moist the longest after rain and is the least likely to suffer drought stress during dry periods.",
            "Despite its good water retention, this soil has the lowest organic matter content in the region. Its nutrient storage capacity is moderate, sitting third among the four groups. Regular addition of organic matter and fertilizer is needed to maintain productivity.",
            "This soil sits in a moderate acidity range, the second least acidic among the four groups. It is generally manageable for most crops without major intervention."
        ),
        (
            "kmeans", 1,
            "Nutrient Poor Moderately Acidic Soil",
            "This soil has a below average ability to retain water. It dries out relatively quickly and may need attention during dry spells.",
            "This is the weakest soil in terms of nutrient storage. It has the lowest nutrient holding capacity in the region. This is confirmed as its single most defining characteristic, far more strongly than any other feature. Applied fertilizers are easily lost before plants can absorb them, making this its most critical limitation.",
            "This soil is moderately acidic, the second most acidic among the four groups. This acidity further reduces the already limited nutrient availability and should be monitored alongside fertilization efforts."
        ),
        (
            "kmeans", 2,
            "Sandy Droughty Low Acidity Soil",
            "This is the driest soil in the region. It has both the lowest water holding ability and the highest sand content, meaning water drains through it very quickly. Irrigation management is critical during dry periods.",
            "Despite its poor water retention, this soil has a relatively high nutrient storage capacity, the second highest among the four groups. However, the high sand content makes it difficult to accumulate organic matter naturally, so regular fertilization is still needed.",
            "This is the least acidic soil among all four groups. Its relatively neutral pH makes it the easiest to manage chemically and accessible to the widest range of crops without significant liming requirements."
        ),
        (
            "kmeans", 3,
            "Organic Rich Highly Acidic Soil",
            "This soil has a fairly good water holding ability, the second highest among the four groups. It retains moisture reasonably well and is unlikely to suffer from severe drought stress.",
            "This is the most fertile soil in the region. It has both the highest organic matter content and the highest nutrient storage capacity among all four groups, making it naturally rich in plant food. Fertilizers applied here are held effectively and available to plants over time.",
            "This is the most acidic soil in the region. Despite its exceptional richness, this high acidity actively locks up the nutrients present, preventing plants from fully accessing the available fertility. Liming to reduce acidity is the single most impactful management action that can take on this soil."
        ),
        (
            "agglomerative", 0,
            "Organic Rich Highly Acidic Soil",
            "This soil has an above average water holding ability, the second highest among the four groups. It retains moisture reasonably well and is less likely to suffer from drought stress between rain events.",
            "This is the most fertile soil in this grouping. It has both the highest organic matter content and the highest nutrient storage capacity and both confirmed as its dominant defining features. It is naturally rich and holds applied fertilizers effectively over time.",
            "This is the most acidic soil among the four groups, the lowest pH in the region. This acidity can lock up the abundant nutrients present, limiting what plants can access. Liming alongside careful fertilization is the most impactful management action to unlock the full fertility of this soil."
        ),
        (
            "agglomerative", 1,
            "Moisture Stable Moderately Fertile Soil",
            "This soil has the highest water holding ability among all four groups. It is the most moisture stable soil in the region and is the least likely to dry out between rain events.",
            "Despite its strong water retention, this soil has the lowest organic matter content in the region. However, its nutrient storage capacity is relatively high, the second highest among the four groups, meaning it can hold on to applied fertilizers effectively. Regular organic matter additions will help improve its natural fertility over time.",
            "This soil is the second least acidic among the four groups. It sits in a moderate and manageable acidity range that is unlikely to cause major problems for most crops without intervention."
        ),
        (
            "agglomerative", 2,
            "Nutrient Storage Weakest Soil",
            "This soil has a below average water holding ability, the second lowest among the four groups. It can dry out relatively quickly and may need attention during extended dry periods.",
            "This soil is most powerfully defined by its very low nutrient storage capacity, the weakest among all four groups and confirmed as overwhelmingly its most dominant characteristic. It struggles to hold on to fertilizers and minerals once applied, meaning nutrients are easily lost before plants can use them. Frequent fertilization is essential.",
            "This soil is the second most acidic among the four groups. This notable acidity further limits the availability of the already scarce nutrients, making liming an important management step alongside regular fertilization."
        ),
        (
            "agglomerative", 3,
            "Sandy Droughty Low Acidity Soil",
            "This soil has the lowest water holding ability in the region. Its very high sand content causes water to drain through rapidly, making it the most drought-prone soil among the four groups. Careful irrigation management during dry periods is essential.",
            "Nutrient storage capacity is moderate, but the high sand content makes it difficult for organic matter to accumulate naturally over time. Regular addition of organic matter and fertilizer is needed to sustain crop productivity.",
            "This is the least acidic soil among all four groups. Its relatively neutral pH is its most manageable chemical property and makes it accessible to a wide range of crops without significant liming requirements."
        ),
        (
            "gmm", 0,
            "Moisture Stable Low Organic Soil",
            "This soil has the highest water holding ability among all four groups. It stays moist the longest after rain and is the least likely to suffer drought stress during dry periods.",
            "Despite its good water retention, this soil has the lowest organic matter content in the region. Its nutrient storage capacity is moderate, sitting third among the four groups. Regular addition of organic matter and fertilizer is needed to maintain productivity.",
            "This soil sits in a moderate acidity range, the second least acidic among the four groups. It is generally manageable for most crops without major intervention."
        ),
        (
            "gmm", 1,
            "Nutrient Poor Moderately Acidic Soil",
            "This soil has a below average ability to retain water. It dries out relatively quickly and may need attention during dry spells.",
            "This is the weakest soil in terms of nutrient storage. It has the lowest nutrient holding capacity in the region. This is confirmed as its single most defining characteristic, far more strongly than any other feature. Applied fertilizers are easily lost before plants can absorb them, making this its most critical limitation.",
            "This soil is moderately acidic, the second most acidic among the four groups. This acidity further reduces the already limited nutrient availability and should be monitored alongside fertilization efforts."
        ),
        (
            "gmm", 2,
            "Sandy Droughty Low Acidity Soil",
            "This is the driest soil in the region. It has both the lowest water holding ability and the highest sand content, meaning water drains through it very quickly. Irrigation management is critical during dry periods.",
            "Despite its poor water retention, this soil has a relatively high nutrient storage capacity, the second highest among the four groups. However, the high sand content makes it difficult to accumulate organic matter naturally, so regular fertilization is still needed.",
            "This is the least acidic soil among all four groups. Its relatively neutral pH makes it the easiest to manage chemically and accessible to the widest range of crops without significant liming requirements."
        ),
        (
            "gmm", 3,
            "Organic Rich Highly Acidic Soil",
            "This soil has a fairly good water holding ability, the second highest among the four groups. It retains moisture reasonably well and is unlikely to suffer from severe drought stress.",
            "This is the most fertile soil in the region. It has both the highest organic matter content and the highest nutrient storage capacity among all four groups, making it naturally rich in plant food. Fertilizers applied here are held effectively and available to plants over time.",
            "This is the most acidic soil in the region. Despite its exceptional richness, this high acidity actively locks up the nutrients present, preventing plants from fully accessing the available fertility. Liming to reduce acidity is the single most impactful management action that can take on this soil."
        )
    ]

    upsert_query = """
    INSERT INTO cluster_explanations
    (model, cluster, zone_name, water_behavior, nutrient_strength, acidity)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        zone_name = VALUES(zone_name),
        water_behavior = VALUES(water_behavior),
        nutrient_strength = VALUES(nutrient_strength),
        acidity = VALUES(acidity)
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