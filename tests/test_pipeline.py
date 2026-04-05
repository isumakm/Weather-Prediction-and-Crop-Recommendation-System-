import pytest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.model_selection import cross_val_score


# ==========================================================================
# SHARED FIXTURES
# ==========================================================================

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "lon":           np.random.uniform(79.8, 80.2, n),
        "lat":           np.random.uniform(6.5,  7.2,  n),
        "vmc33_0-5cm":   np.random.uniform(0.25, 0.40, n),
        "vmc33_5-15cm":  np.random.uniform(0.22, 0.38, n),
        "vmc15_0-5cm":   np.random.uniform(0.10, 0.18, n),
        "vmc15_5-15cm":  np.random.uniform(0.09, 0.16, n),
        "ph_0-5cm":      np.random.uniform(4.5,  6.0,  n),
        "ph_5-15cm":     np.random.uniform(4.3,  5.8,  n),
        "bd_0-5cm":      np.random.uniform(1.1,  1.5,  n),
        "bd_5-15cm":     np.random.uniform(1.2,  1.6,  n),
        "oc_0-5cm":      np.random.uniform(1.5,  5.0,  n),
        "oc_5-15cm":     np.random.uniform(1.0,  4.0,  n),
        "cec_0-5cm":     np.random.uniform(8.0,  22.0, n),
        "cec_5-15cm":    np.random.uniform(7.0,  20.0, n),
        "clay_0-5cm":    np.random.uniform(10.0, 30.0, n),
        "clay_5-15cm":   np.random.uniform(10.0, 30.0, n),
        "sand_0-5cm":    np.random.uniform(50.0, 80.0, n),
        "sand_5-15cm":   np.random.uniform(50.0, 80.0, n),
    })


@pytest.fixture
def cluster_features_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "taw":            np.random.uniform(5,   12,  n),
        "organic_carbon": np.random.uniform(2,   5,   n),
        "cec":            np.random.uniform(8,   20,  n),
        "ph":             np.random.uniform(4.5, 6.0, n),
        "sand_pct":       np.random.uniform(50,  90,  n),
    })


@pytest.fixture
def well_separated_X():
    np.random.seed(42)
    return np.vstack([
        np.random.randn(50, 5) + [0,  0,  0,  0,  0 ],
        np.random.randn(50, 5) + [10, 10, 10, 10, 10],
        np.random.randn(50, 5) + [20, 20, 20, 20, 20],
        np.random.randn(50, 5) + [30, 30, 30, 30, 30],
    ])


@pytest.fixture
def realistic_cluster_labels():
    return pd.Series([0]*353 + [1]*1305 + [2]*524 + [3]*1240)


@pytest.fixture
def realistic_cluster_means():
    return pd.DataFrame({
        "taw":            [8.83, 6.90, 6.63, 7.79],
        "organic_carbon": [2.91, 3.18, 3.31, 4.11],
        "cec":            [16.64, 9.77, 17.20, 18.65],
        "ph":             [5.19, 4.86, 5.34, 4.81],
        "sand_pct":       [70.33, 66.27, 78.76, 62.41],
    }, index=[0, 1, 2, 3])


@pytest.fixture
def mock_clusters_df():
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "lat":            np.random.uniform(6.5,  7.2,  n),
        "lon":            np.random.uniform(79.8, 80.2, n),
        "taw":            np.random.uniform(5,    12,   n),
        "organic_carbon": np.random.uniform(2,    5,    n),
        "cec":            np.random.uniform(8,    20,   n),
        "ph":             np.random.uniform(4.5,  6.0,  n),
        "sand_pct":       np.random.uniform(50,   90,   n),
        "cluster_kmeans": np.random.randint(0, 4, n),
        "cluster_agg":    np.random.randint(0, 4, n),
        "cluster_gmm":    np.random.randint(0, 4, n),
    })


# ==========================================================================
# NB01 - DATA LOADING
# ==========================================================================

def apply_nodata_mask(arr, nodata):
    result = arr.astype(float)
    result[result == nodata] = np.nan
    return result


def test_nodata_pixels_become_nan():
    data = np.array([1.0, 2.0, -9999.0, 4.0, -9999.0])
    result = apply_nodata_mask(data, -9999.0)
    assert np.isnan(result[2])
    assert np.isnan(result[4])


def test_valid_pixels_unchanged_after_nodata_mask():
    data = np.array([1.0, 2.0, -9999.0, 4.0])
    result = apply_nodata_mask(data, -9999.0)
    assert result[0] == 1.0
    assert result[1] == 2.0
    assert result[3] == 4.0


def test_nodata_mask_count():
    data = np.array([1.0, -9999.0, 3.0, -9999.0, 5.0])
    result = apply_nodata_mask(data, -9999.0)
    assert np.sum(np.isnan(result)) == 2


def test_no_nodata_array_unchanged():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    result = apply_nodata_mask(data, -9999.0)
    assert not np.any(np.isnan(result))


def test_dropna_removes_nan_rows():
    df = pd.DataFrame({
        "lon": [79.8, 80.0, np.nan],
        "lat": [6.9,  7.0,  7.1],
        "ph":  [5.0,  4.8,  5.2]
    })
    df_clean = df.dropna().reset_index(drop=True)
    assert df_clean.isnull().sum().sum() == 0
    assert len(df_clean) == 2


def test_dropna_preserves_valid_rows():
    df = pd.DataFrame({
        "lon": [79.8, np.nan, 80.0],
        "lat": [6.9,  7.0,   7.1],
    })
    df_clean = df.dropna().reset_index(drop=True)
    assert len(df_clean) == 2
    assert df_clean["lon"].iloc[0] == 79.8


def test_output_has_lon_lat_columns():
    df = pd.DataFrame({"lon": [79.8], "lat": [6.9]})
    assert "lon" in df.columns
    assert "lat" in df.columns


# ==========================================================================
# NB02 - EDA
# ==========================================================================

def is_within_western_province(lat, lon):
    return 6.4 <= lat <= 7.3 and 79.7 <= lon <= 80.3


def test_western_province_point_inside():
    assert is_within_western_province(6.9, 79.9) is True


def test_western_province_point_outside_lat():
    assert is_within_western_province(8.0, 79.9) is False


def test_western_province_point_outside_lon():
    assert is_within_western_province(6.9, 81.5) is False


def test_western_province_boundary_lat_low():
    assert is_within_western_province(6.4, 80.0) is True


def test_western_province_boundary_lat_high():
    assert is_within_western_province(7.3, 80.0) is True


def test_spatial_clip_filters_dataframe():
    df = pd.DataFrame({
        "lat": [6.9, 8.0, 7.0, 5.0],
        "lon": [79.9, 79.9, 80.0, 80.0]
    })
    mask = df.apply(lambda r: is_within_western_province(r.lat, r.lon), axis=1)
    clipped = df[mask]
    assert len(clipped) == 2


def test_western_province_uses_three_districts():
    western_names = ["Colombo", "Gampaha", "Kalutara"]
    assert len(western_names) == 3
    assert "Colombo"  in western_names
    assert "Gampaha"  in western_names
    assert "Kalutara" in western_names


# ==========================================================================
# NB03 - FEATURE ENGINEERING
# ==========================================================================

def calculate_taw_nb03(df):
    depths = ["0-5cm", "5-15cm"]
    thickness_mm = [50, 100]
    fc_cols = [f"vmc33_{d}" for d in depths]
    wp_cols = [f"vmc15_{d}" for d in depths]
    taw_layers = []
    for i in range(len(depths)):
        taw = (df[fc_cols[i]] - df[wp_cols[i]]) * thickness_mm[i]
        taw_layers.append(taw)
    return sum(taw_layers)


def test_taw_positive_for_valid_inputs(sample_df):
    taw = calculate_taw_nb03(sample_df)
    assert (taw > 0).all()


def test_taw_zero_when_fc_equals_wp():
    df = pd.DataFrame({
        "vmc33_0-5cm": [0.25], "vmc15_0-5cm": [0.25],
        "vmc33_5-15cm": [0.30], "vmc15_5-15cm": [0.30],
    })
    taw = calculate_taw_nb03(df)
    assert taw.iloc[0] == pytest.approx(0.0)


def test_taw_scales_with_moisture_difference():
    df_low  = pd.DataFrame({"vmc33_0-5cm": [0.20], "vmc15_0-5cm": [0.15],
                             "vmc33_5-15cm": [0.20], "vmc15_5-15cm": [0.15]})
    df_high = pd.DataFrame({"vmc33_0-5cm": [0.40], "vmc15_0-5cm": [0.15],
                             "vmc33_5-15cm": [0.40], "vmc15_5-15cm": [0.15]})
    assert calculate_taw_nb03(df_high).iloc[0] > calculate_taw_nb03(df_low).iloc[0]


def test_taw_no_nan_for_clean_inputs(sample_df):
    taw = calculate_taw_nb03(sample_df)
    assert not taw.isna().any()


def test_taw_within_agronomic_range(sample_df):
    taw = calculate_taw_nb03(sample_df)
    assert (taw >= 3).all()
    assert (taw <= 50).all()


def test_awc_equals_taw_over_depth():
    taw = pd.Series([10.0, 15.0, 8.0])
    total_depth_mm = 150
    awc = taw / total_depth_mm
    expected = pd.Series([10/150, 15/150, 8/150])
    pd.testing.assert_series_equal(awc, expected)


def test_awc_always_positive():
    taw = pd.Series([5.0, 8.0, 12.0])
    awc = taw / 150
    assert (awc > 0).all()


def thickness_weighted_avg(df, prop, depths, thickness_cm):
    prop_cols   = [f"{prop}_{d}" for d in depths]
    prop_values = np.vstack([df[c] for c in prop_cols])
    return np.average(prop_values, axis=0, weights=thickness_cm)


def test_thickness_weighted_avg_single_layer(sample_df):
    result = thickness_weighted_avg(sample_df, "ph", ["0-5cm"], [5])
    np.testing.assert_array_almost_equal(result, sample_df["ph_0-5cm"].values)


def test_thickness_weighted_avg_equal_weights(sample_df):
    result   = thickness_weighted_avg(sample_df, "ph", ["0-5cm", "5-15cm"], [10, 10])
    expected = (sample_df["ph_0-5cm"].values + sample_df["ph_5-15cm"].values) / 2
    np.testing.assert_array_almost_equal(result, expected)


def test_thickness_weighted_avg_bounded_by_inputs(sample_df):
    result  = thickness_weighted_avg(sample_df, "ph", ["0-5cm", "5-15cm"], [5, 10])
    min_val = np.minimum(sample_df["ph_0-5cm"].values, sample_df["ph_5-15cm"].values)
    max_val = np.maximum(sample_df["ph_0-5cm"].values, sample_df["ph_5-15cm"].values)
    assert (result >= min_val).all()
    assert (result <= max_val).all()


def test_thickness_weighted_avg_heavier_layer_dominates(sample_df):
    result          = thickness_weighted_avg(sample_df, "ph", ["0-5cm", "5-15cm"], [5, 10])
    diff_to_deep    = np.abs(result - sample_df["ph_5-15cm"].values).mean()
    diff_to_shallow = np.abs(result - sample_df["ph_0-5cm"].values).mean()
    assert diff_to_deep < diff_to_shallow


def mass_weighted_avg(df, prop, bd, depths, thickness_cm):
    prop_cols   = [f"{prop}_{d}" for d in depths]
    bd_cols     = [f"{bd}_{d}"   for d in depths]
    prop_values = np.vstack([df[c] for c in prop_cols])
    bd_values   = np.vstack([df[c] for c in bd_cols])
    weights     = bd_values * np.array(thickness_cm)[:, None]
    return np.sum(prop_values * weights, axis=0) / np.sum(weights, axis=0)


def test_mass_weighted_avg_no_nan(sample_df):
    result = mass_weighted_avg(sample_df, "oc", "bd", ["0-5cm", "5-15cm"], [5, 10])
    assert not np.any(np.isnan(result))


def test_mass_weighted_avg_bounded_by_inputs(sample_df):
    result  = mass_weighted_avg(sample_df, "oc", "bd", ["0-5cm", "5-15cm"], [5, 10])
    min_val = np.minimum(sample_df["oc_0-5cm"].values, sample_df["oc_5-15cm"].values)
    max_val = np.maximum(sample_df["oc_0-5cm"].values, sample_df["oc_5-15cm"].values)
    assert (result >= min_val).all()
    assert (result <= max_val).all()


def test_mass_weighted_avg_same_bd_equals_thickness_weighted(sample_df):
    df_eq = sample_df.copy()
    df_eq["bd_0-5cm"]  = 1.3
    df_eq["bd_5-15cm"] = 1.3
    mass_result      = mass_weighted_avg(df_eq, "oc", "bd", ["0-5cm", "5-15cm"], [5, 10])
    thickness_result = thickness_weighted_avg(df_eq, "oc", ["0-5cm", "5-15cm"], [5, 10])
    np.testing.assert_array_almost_equal(mass_result, thickness_result, decimal=10)


def classify_usda_texture(row):
    sand = row["sand_pct"]
    clay = row["clay_pct"]
    silt = 100 - sand - clay
    if sand >= 86 and clay <= 10 and silt <= 14:
        return "Sand"
    if 70 <= sand < 86 and clay <= 15 and silt <= 30:
        return "Loamy sand"
    if 50 <= sand < 70 and clay <= 20 and silt <= 50:
        return "Sandy loam"
    if sand <= 20 and clay <= 12 and silt >= 80:
        return "Silt"
    if clay <= 27 and 50 <= silt < 80:
        return "Silt loam"
    if 23 <= sand <= 52 and 7 <= clay <= 27 and 28 <= silt <= 50:
        return "Loam"
    if 45 <= sand <= 80 and 20 <= clay < 35 and silt <= 28:
        return "Sandy clay loam"
    if 20 <= sand < 45 and 27 <= clay < 40 and 15 <= silt <= 52:
        return "Clay loam"
    if sand <= 20 and 27 <= clay < 40 and silt >= 40:
        return "Silty clay loam"
    if sand >= 45 and clay >= 35 and silt <= 20:
        return "Sandy clay"
    if sand <= 20 and clay >= 40 and silt >= 40:
        return "Silty clay"
    if clay >= 40:
        return "Clay"
    return "Loam"


def test_texture_high_sand_returns_sand():
    assert classify_usda_texture({"sand_pct": 90, "clay_pct": 5}) == "Sand"


def test_texture_high_clay_returns_clay():
    assert classify_usda_texture({"sand_pct": 20, "clay_pct": 45}) == "Clay"


def test_texture_loamy_sand():
    assert classify_usda_texture({"sand_pct": 75, "clay_pct": 10}) == "Loamy sand"


def test_texture_sandy_loam():
    assert classify_usda_texture({"sand_pct": 60, "clay_pct": 15}) == "Sandy loam"


def test_texture_returns_string():
    assert isinstance(classify_usda_texture({"sand_pct": 60, "clay_pct": 20}), str)


def test_texture_returns_non_empty():
    assert len(classify_usda_texture({"sand_pct": 60, "clay_pct": 20})) > 0


def test_silt_always_non_negative():
    for sand in range(0, 80, 10):
        for clay in range(0, 100 - sand, 10):
            silt = 100 - sand - clay
            assert silt >= 0


def test_cluster_features_list_correct():
    CLUSTER_FEATURES = ["taw", "organic_carbon", "cec", "ph", "sand_pct"]
    assert len(CLUSTER_FEATURES) == 5
    assert "texture_class" not in CLUSTER_FEATURES


# ==========================================================================
# NB04 - PREPROCESSING
# ==========================================================================

def apply_winsorization(series, lower_pct=0.01, upper_pct=0.99):
    lower = series.quantile(lower_pct)
    upper = series.quantile(upper_pct)
    return series.clip(lower, upper), lower, upper


def test_winsorization_clips_upper_outliers():
    data = pd.Series(list(range(100)) + [9999])
    clipped, _, upper = apply_winsorization(data)
    assert clipped.max() <= upper


def test_winsorization_clips_lower_outliers():
    data = pd.Series([-9999] + list(range(100)))
    clipped, lower, _ = apply_winsorization(data)
    assert clipped.min() >= lower


def test_winsorization_preserves_length():
    data = pd.Series(range(200))
    clipped, _, _ = apply_winsorization(data)
    assert len(clipped) == len(data)


def test_winsorization_bounds_ordered():
    data = pd.Series(range(200))
    _, lower, upper = apply_winsorization(data)
    assert lower < upper


def test_winsorization_no_effect_on_clean_data():
    data = pd.Series(range(0, 1000))
    clipped, lower, upper = apply_winsorization(data)
    core = data[(data > lower) & (data < upper)]
    clipped_core = clipped[(data > lower) & (data < upper)]
    pd.testing.assert_series_equal(core, clipped_core, check_dtype=False)


def test_winsorization_only_on_sand_pct(cluster_features_df):
    df = cluster_features_df.copy()
    original_taw = df["taw"].copy()
    original_ph  = df["ph"].copy()
    df["sand_pct"], _, _ = apply_winsorization(df["sand_pct"])
    pd.testing.assert_series_equal(df["taw"], original_taw)
    pd.testing.assert_series_equal(df["ph"],  original_ph)


def test_scaler_mean_near_zero(cluster_features_df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_features_df)
    assert np.abs(scaled.mean(axis=0)).max() < 1e-10


def test_scaler_std_near_one(cluster_features_df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_features_df)
    assert np.abs(scaled.std(axis=0) - 1.0).max() < 1e-10


def test_no_nan_in_scaled_output(cluster_features_df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_features_df)
    assert not np.any(np.isnan(scaled))


def test_scaler_preserves_shape(cluster_features_df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_features_df)
    assert scaled.shape == cluster_features_df.shape


def test_transform_matches_fit_transform(cluster_features_df):
    scaler = StandardScaler()
    fit_result = scaler.fit_transform(cluster_features_df)
    transform_result = scaler.transform(cluster_features_df)
    np.testing.assert_array_almost_equal(fit_result, transform_result)


def test_scaled_df_retains_lat_lon():
    df = pd.DataFrame({
        "taw": [8.0], "organic_carbon": [3.0], "cec": [15.0],
        "ph": [5.0], "sand_pct": [65.0],
        "lat": [6.9], "lon": [79.9], "texture_class": ["Sandy loam"]
    })
    CLUSTER_FEATURES = ["taw", "organic_carbon", "cec", "ph", "sand_pct"]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[CLUSTER_FEATURES])
    df_scaled = pd.DataFrame(scaled_data, columns=CLUSTER_FEATURES)
    df_scaled[["lat", "lon", "texture_class"]] = df[["lat", "lon", "texture_class"]]
    assert "lat"          in df_scaled.columns
    assert "lon"          in df_scaled.columns
    assert "texture_class" in df_scaled.columns


# ==========================================================================
# NB05 - CLUSTERING
# ==========================================================================

def evaluate_clustering(X, labels):
    return {
        "silhouette":        silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin":    davies_bouldin_score(X, labels)
    }


def test_evaluate_clustering_returns_all_keys(well_separated_X):
    labels = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
    result = evaluate_clustering(well_separated_X, labels)
    assert "silhouette"        in result
    assert "calinski_harabasz" in result
    assert "davies_bouldin"    in result


def test_silhouette_in_valid_range(well_separated_X):
    labels = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
    assert -1 <= evaluate_clustering(well_separated_X, labels)["silhouette"] <= 1


def test_silhouette_high_for_well_separated(well_separated_X):
    labels = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
    assert evaluate_clustering(well_separated_X, labels)["silhouette"] > 0.8


def test_calinski_harabasz_positive(well_separated_X):
    labels = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
    assert evaluate_clustering(well_separated_X, labels)["calinski_harabasz"] > 0


def test_davies_bouldin_positive(well_separated_X):
    labels = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
    assert evaluate_clustering(well_separated_X, labels)["davies_bouldin"] > 0


def test_davies_bouldin_low_for_well_separated(well_separated_X):
    labels = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
    assert evaluate_clustering(well_separated_X, labels)["davies_bouldin"] < 0.5


def test_kmeans_reproducible(well_separated_X):
    m1 = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(well_separated_X)
    m2 = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(well_separated_X)
    np.testing.assert_array_equal(m1, m2)


def test_kmeans_labels_within_range(well_separated_X):
    labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(well_separated_X)
    assert labels.min() >= 0
    assert labels.max() <= 3


def test_kmeans_all_clusters_present(well_separated_X):
    labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(well_separated_X)
    assert set(labels) == {0, 1, 2, 3}


def test_kmeans_no_unassigned_pixels(well_separated_X):
    labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(well_separated_X)
    assert len(labels) == len(well_separated_X)


def test_kmeans_k4_better_silhouette_than_k2(well_separated_X):
    sil_k2 = silhouette_score(well_separated_X,
        KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(well_separated_X))
    sil_k4 = silhouette_score(well_separated_X,
        KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(well_separated_X))
    assert sil_k4 > sil_k2


def test_gmm_reproducible(well_separated_X):
    m1 = GaussianMixture(n_components=4, random_state=42).fit_predict(well_separated_X)
    m2 = GaussianMixture(n_components=4, random_state=42).fit_predict(well_separated_X)
    np.testing.assert_array_equal(m1, m2)


def test_gmm_labels_within_range(well_separated_X):
    labels = GaussianMixture(n_components=4, random_state=42).fit_predict(well_separated_X)
    assert labels.min() >= 0
    assert labels.max() <= 3


def test_agglomerative_labels_within_range(well_separated_X):
    labels = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(well_separated_X)
    assert labels.min() >= 0
    assert labels.max() <= 3


def test_cluster_merge_uses_lon_lat():
    df_original = pd.DataFrame({
        "lon": [79.8, 79.9, 80.0],
        "lat": [6.9,  7.0,  7.1],
        "taw": [8.0,  7.0,  6.0]
    })
    labels_df = pd.DataFrame({
        "lon":            [79.8, 79.9, 80.0],
        "lat":            [6.9,  7.0,  7.1],
        "cluster_kmeans": [0, 1, 2],
        "cluster_agg":    [1, 0, 2],
        "cluster_gmm":    [0, 1, 3],
    })
    merged = df_original.merge(
        labels_df[["lon", "lat", "cluster_kmeans", "cluster_agg", "cluster_gmm"]],
        on=["lon", "lat"], how="left"
    )
    assert "cluster_kmeans" in merged.columns
    assert "cluster_agg"    in merged.columns
    assert "cluster_gmm"    in merged.columns
    assert len(merged) == len(df_original)
    assert merged["cluster_kmeans"].iloc[0] == 0


def test_inertia_decreases_with_k(well_separated_X):
    inertias = []
    for k in range(2, 6):
        m = KMeans(n_clusters=k, random_state=42, n_init=10)
        m.fit(well_separated_X)
        inertias.append(m.inertia_)
    for i in range(len(inertias) - 1):
        assert inertias[i] > inertias[i + 1]


def test_aic_bic_columns_present():
    record = {"k": 4, "AIC": 12547.216, "BIC": 13056.668}
    df = pd.DataFrame([record])
    assert "k"   in df.columns
    assert "AIC" in df.columns
    assert "BIC" in df.columns


# ==========================================================================
# NB06 - CLUSTER INTERPRETATION
# ==========================================================================

def test_all_four_clusters_present(realistic_cluster_labels):
    assert set(realistic_cluster_labels.unique()) == {0, 1, 2, 3}


def test_cluster_counts_sum_to_total(realistic_cluster_labels):
    assert realistic_cluster_labels.value_counts().sum() == len(realistic_cluster_labels)


def test_no_cluster_is_empty(realistic_cluster_labels):
    for c in range(4):
        assert (realistic_cluster_labels == c).sum() > 0


def test_dominant_clusters_cover_majority(realistic_cluster_labels):
    counts = realistic_cluster_labels.value_counts()
    top_two = (counts[1] + counts[3]) / len(realistic_cluster_labels)
    assert top_two > 0.70


def test_taw_within_agronomic_range(realistic_cluster_means):
    for val in realistic_cluster_means["taw"]:
        assert 3 <= val <= 20


def test_ph_within_valid_range(realistic_cluster_means):
    for val in realistic_cluster_means["ph"]:
        assert 3.5 <= val <= 8.5


def test_organic_carbon_within_valid_range(realistic_cluster_means):
    for val in realistic_cluster_means["organic_carbon"]:
        assert 0 < val <= 20


def test_cec_within_valid_range(realistic_cluster_means):
    for val in realistic_cluster_means["cec"]:
        assert 0 < val <= 50


def test_sand_pct_within_valid_range(realistic_cluster_means):
    for val in realistic_cluster_means["sand_pct"]:
        assert 0 <= val <= 100


def test_radar_scaler_fitted_once_on_all_models(realistic_cluster_means):
    summaries = {"KMeans": realistic_cluster_means, "GMM": realistic_cluster_means.copy()}
    combined  = pd.concat(summaries.values())
    scaler    = MinMaxScaler()
    scaler.fit(combined)
    norm_a = scaler.transform(summaries["KMeans"])
    norm_b = scaler.transform(summaries["GMM"])
    np.testing.assert_array_almost_equal(norm_a, norm_b)


def test_ari_identical_labels_is_one():
    labels = np.array([0]*353 + [1]*1305 + [2]*524 + [3]*1240)
    assert adjusted_rand_score(labels, labels) == pytest.approx(1.0)


def test_nmi_identical_labels_is_one():
    labels = np.array([0]*353 + [1]*1305 + [2]*524 + [3]*1240)
    assert normalized_mutual_info_score(labels, labels) == pytest.approx(1.0)


def test_ari_range_valid():
    np.random.seed(42)
    a = np.random.randint(0, 4, 500)
    b = np.random.randint(0, 4, 500)
    assert -1 <= adjusted_rand_score(a, b) <= 1


def test_nmi_range_valid():
    np.random.seed(42)
    a = np.random.randint(0, 4, 500)
    b = np.random.randint(0, 4, 500)
    assert 0 <= normalized_mutual_info_score(a, b) <= 1


def test_ari_high_for_near_identical_labels():
    labels_a = np.array([0]*353 + [1]*1305 + [2]*524 + [3]*1240)
    labels_b = labels_a.copy()
    labels_b[:20] = (labels_b[:20] + 1) % 4
    assert adjusted_rand_score(labels_a, labels_b) > 0.9


# ==========================================================================
# NB07 - XAI CLUSTER INTERPRETATION
# ==========================================================================

@pytest.fixture
def surrogate_data(well_separated_X):
    labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(well_separated_X)
    return well_separated_X, labels


def test_centroid_deviation_shape(well_separated_X):
    labels = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
    df = pd.DataFrame(well_separated_X, columns=["f1","f2","f3","f4","f5"])
    df["cluster"] = labels
    global_mean   = df[["f1","f2","f3","f4","f5"]].mean()
    cluster_means = df.groupby("cluster")[["f1","f2","f3","f4","f5"]].mean()
    deviations    = cluster_means - global_mean
    assert deviations.shape == (4, 5)


def test_centroid_deviation_has_positive_and_negative(well_separated_X):
    labels = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
    df = pd.DataFrame(well_separated_X, columns=["f1","f2","f3","f4","f5"])
    df["cluster"] = labels
    global_mean   = df[["f1","f2","f3","f4","f5"]].mean()
    cluster_means = df.groupby("cluster")[["f1","f2","f3","f4","f5"]].mean()
    deviations    = cluster_means - global_mean
    assert (deviations.values > 0).any()
    assert (deviations.values < 0).any()


def test_dt_surrogate_trains_without_error(surrogate_data):
    X, labels = surrogate_data
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)
    dt.fit(X, labels)
    assert len(dt.predict(X)) == len(labels)


def test_dt_train_accuracy_above_threshold(surrogate_data):
    X, labels = surrogate_data
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    dt.fit(X, labels)
    assert accuracy_score(labels, dt.predict(X)) > 0.80


def test_dt_cv_accuracy_above_threshold(surrogate_data):
    X, labels = surrogate_data
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    cv_acc = cross_val_score(dt, X, labels, cv=5, scoring="accuracy").mean()
    assert cv_acc > 0.75


def test_dt_cv_accuracy_not_exceeds_train(surrogate_data):
    X, labels = surrogate_data
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    dt.fit(X, labels)
    train_acc = accuracy_score(labels, dt.predict(X))
    cv_acc    = cross_val_score(dt, X, labels, cv=5, scoring="accuracy").mean()
    assert cv_acc <= train_acc


def test_rf_trains_without_error(surrogate_data):
    X, labels = surrogate_data
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                min_samples_leaf=20, random_state=42)
    rf.fit(X, labels)
    assert len(rf.predict(X)) == len(labels)


def test_rf_train_accuracy_at_least_as_good_as_dt(surrogate_data):
    X, labels = surrogate_data
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                min_samples_leaf=10, random_state=42)
    dt.fit(X, labels)
    rf.fit(X, labels)
    assert accuracy_score(labels, rf.predict(X)) >= accuracy_score(labels, dt.predict(X))


def test_rf_cv_accuracy_above_threshold(surrogate_data):
    X, labels = surrogate_data
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                min_samples_leaf=10, random_state=42)
    assert cross_val_score(rf, X, labels, cv=5, scoring="accuracy").mean() > 0.80


def test_shap_data_structure_has_global_and_clusters():
    FEATURES = ["taw", "organic_carbon", "cec", "ph", "sand_pct"]
    shap_entry = {
        "global":   {f: 0.1 for f in FEATURES},
        "clusters": {str(i): {f: 0.1 for f in FEATURES} for i in range(4)}
    }
    assert "global"   in shap_entry
    assert "clusters" in shap_entry
    assert len(shap_entry["clusters"]) == 4
    for cluster_vals in shap_entry["clusters"].values():
        assert set(cluster_vals.keys()) == set(FEATURES)


def test_shap_global_values_non_negative():
    global_importance = {"taw": 0.0756, "organic_carbon": 0.1126,
                         "cec": 0.1594, "ph": 0.0625, "sand_pct": 0.0245}
    for val in global_importance.values():
        assert val >= 0


def test_shap_covers_all_features():
    FEATURES = ["taw", "organic_carbon", "cec", "ph", "sand_pct"]
    global_importance = {f: 0.1 for f in FEATURES}
    assert set(global_importance.keys()) == set(FEATURES)

# ==========================================================================
# NB08 - EXPORT FOR WEB
# ==========================================================================

def test_web_cluster_csv_has_correct_columns(mock_clusters_df):
    exported = mock_clusters_df[["lat", "lon", "cluster_kmeans"]].rename(
        columns={"cluster_kmeans": "cluster"})
    assert list(exported.columns) == ["lat", "lon", "cluster"]


def test_web_cluster_csv_row_count_matches_source(mock_clusters_df):
    exported = mock_clusters_df[["lat", "lon", "cluster_kmeans"]].rename(
        columns={"cluster_kmeans": "cluster"})
    assert len(exported) == len(mock_clusters_df)


def test_web_cluster_csv_cluster_values_valid(mock_clusters_df):
    exported = mock_clusters_df[["lat", "lon", "cluster_kmeans"]].rename(
        columns={"cluster_kmeans": "cluster"})
    assert exported["cluster"].between(0, 3).all()


def test_three_csv_files_produced(mock_clusters_df):
    CLUSTER_MODELS = {"kmeans": "cluster_kmeans",
                      "agglomerative": "cluster_agg", "gmm": "cluster_gmm"}
    exports = {
        model: mock_clusters_df[["lat", "lon", col]].rename(columns={col: "cluster"})
        for model, col in CLUSTER_MODELS.items()
    }
    assert len(exports) == 3
    assert set(exports.keys()) == {"kmeans", "agglomerative", "gmm"}


def test_cluster_means_json_has_all_models(mock_clusters_df):
    FEATURES = ["taw", "organic_carbon", "cec", "ph", "sand_pct"]
    CLUSTER_MODELS = {"kmeans": "cluster_kmeans",
                      "agglomerative": "cluster_agg", "gmm": "cluster_gmm"}
    cluster_means = {
        model: mock_clusters_df.groupby(col)[FEATURES].mean().round(3).to_dict()
        for model, col in CLUSTER_MODELS.items()
    }
    assert set(cluster_means.keys()) == {"kmeans", "agglomerative", "gmm"}


def test_cluster_means_json_has_all_features(mock_clusters_df):
    FEATURES = ["taw", "organic_carbon", "cec", "ph", "sand_pct"]
    means = mock_clusters_df.groupby("cluster_kmeans")[FEATURES].mean().round(3).to_dict()
    assert set(means.keys()) == set(FEATURES)


def test_cluster_means_values_are_numeric(mock_clusters_df):
    FEATURES = ["taw", "organic_carbon", "cec", "ph", "sand_pct"]
    means = mock_clusters_df.groupby("cluster_kmeans")[FEATURES].mean().round(3).to_dict()
    for feature, cluster_vals in means.items():
        for val in cluster_vals.values():
            assert isinstance(val, float)