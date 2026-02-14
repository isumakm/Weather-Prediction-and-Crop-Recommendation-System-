from backend.database.db import get_connection

def get_soil_by_location(lat, lon, tolerance=0.001):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT *
    FROM soil_points
    WHERE ABS(lat - %s) < %s
      AND ABS(lon - %s) < %s
    LIMIT 1
    """

    cursor.execute(query, (lat, tolerance, lon, tolerance))
    soil = cursor.fetchone()

    cursor.close()
    conn.close()

    return soil


def get_cluster_explanation(cluster_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT *
    FROM cluster_explanations
    WHERE cluster = %s
    """

    cursor.execute(query, (cluster_id,))
    explanation = cursor.fetchone()

    cursor.close()
    conn.close()

    return explanation
