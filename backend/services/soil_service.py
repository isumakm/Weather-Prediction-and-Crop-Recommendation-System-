from backend.database.db import get_connection


def get_soil_by_location(lat, lon, model):
    if not (6.4 <= lat <= 7.3 and 79.7 <= lon <= 80.3):
        return None

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    threshold = 0.0001

    query = """
        SELECT *,
        (POW(lat - %s, 2) + POW(lon - %s, 2)) AS distance
        FROM soil_points
        WHERE model = %s
        HAVING distance <= %s
        ORDER BY distance
        LIMIT 1
    """

    cursor.execute(query, (lat, lon, model, threshold))
    soil = cursor.fetchone()

    cursor.close()
    conn.close()

    return soil


def get_cluster_means(cluster_id, model):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT
            AVG(taw) AS taw,
            AVG(organic_carbon) AS organic_carbon,
            AVG(cec) AS cec,
            AVG(ph) AS ph,
            AVG(bulk_density) AS bulk_density,
            AVG(sand_pct) AS sand_pct
        FROM soil_points
        WHERE model = %s AND cluster = %s
    """

    cursor.execute(query, (model, cluster_id))
    means = cursor.fetchone()

    cursor.close()
    conn.close()

    return means


def get_cluster_explanation(cluster_id, model):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT *
        FROM cluster_explanations
        WHERE model = %s AND cluster = %s
    """

    cursor.execute(query, (model, cluster_id))
    explanation = cursor.fetchone()

    cursor.close()
    conn.close()

    return explanation