import os
import rasterio
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, box
import contextily as ctx
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.features import geometry_mask
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Define the output directory
OUTPUT_DIR = '//sungis15/Hons_scratch/25022318/Data'


def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def clip_raster_to_study_area(raster_path, study_area_gdf):
    """
    Clips a raster file to the extent of a study area.

    @param raster_path: File path to raster.
    @param study_area_gdf: GeoDataFrame representing the study area polygon.
    @return: Dictionary with raster data, profile, transform, bounds, and CRS.
    """
    with rasterio.open(raster_path) as src:
        # Ensure CRS match between raster and vector
        if src.crs != study_area_gdf.crs:
            study_area_reproj = study_area_gdf.to_crs(src.crs)
        else:
            study_area_reproj = study_area_gdf

        geometries = study_area_reproj.geometry.values

        clipped_data, clipped_transform = mask(src, geometries, crop=True)
        clipped_profile = src.profile.copy()
        clipped_profile.update({
            'height': clipped_data.shape[1],
            'width': clipped_data.shape[2],
            'transform': clipped_transform
        })

        return {
            'data': clipped_data[0],
            'profile': clipped_profile,
            'transform': clipped_transform,
            'bounds': rasterio.transform.array_bounds(clipped_data.shape[1], clipped_data.shape[2], clipped_transform),
            'crs': src.crs
        }


def create_clipped_prediction_map(model, scaler, raster_data, reference_profile,
                                  study_area_gdf, output_path, selected_features,
                                  current_date=None):
    """
    Predicts fire probability across a raster area using a trained model.

    @param model: Trained classifier (e.g., RandomForest).
    @param scaler: StandardScaler used to normalize features.
    @param raster_data: Dictionary of raster features clipped to study area.
    @param reference_profile: Raster profile to use as output reference.
    @param study_area_gdf: GeoDataFrame of the study area.
    @param output_path: Output filepath for the prediction raster.
    @param selected_features: List of feature names used in model training.
    @param current_date: Optional datetime to use for temporal variables.
    @return: Tuple (probability_map, output_profile, transform, crs)
    """
    print(f"Creating clipped prediction map: {output_path}")

    height, width = reference_profile['height'], reference_profile['width']
    transform = reference_profile['transform']
    crs = reference_profile.get('crs', 'EPSG:4326')

    # Create lat/lon arrays
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    lons = np.array(xs)
    lats = np.array(ys)

    if current_date is None:
        current_date = datetime(2023, 7, 15)

    # Build feature matrix
    feature_stack = []
    for feature in selected_features:
        if feature == 'latitude':
            feature_stack.append(lats.flatten())
        elif feature == 'longitude':
            feature_stack.append(lons.flatten())
        elif feature == 'year':
            feature_stack.append(np.full(lats.size, current_date.year))
        elif feature == 'month':
            feature_stack.append(np.full(lats.size, current_date.month))
        elif feature == 'day':
            feature_stack.append(np.full(lats.size, current_date.day))
        else:
            raster_array = raster_data[feature]['data']
            feature_stack.append(raster_array.flatten())

    features_array = np.array(feature_stack).T

    # Fill NaN with training mean or median
    for i, feature in enumerate(selected_features):
        mask_nan = np.isnan(features_array[:, i]) | np.isinf(features_array[:, i])
        if np.any(mask_nan):
            features_array[mask_nan, i] = np.nanmedian(features_array[:, i])

    features_scaled = scaler.transform(features_array)

    # Predict probabilities
    print("Making predictions...")
    prediction_probs = model.predict_proba(features_scaled)[:, 1]
    probability_map = prediction_probs.reshape(height, width)

    # Apply study area mask
    study_area_reproj = study_area_gdf.to_crs(crs) if study_area_gdf.crs != crs else study_area_gdf
    study_mask = geometry_mask(study_area_reproj.geometry.values, (height, width), transform, invert=True)
    probability_map = np.where(study_mask, probability_map, np.nan)

    # Save raster to the specified directory
    ensure_output_dir()
    prob_output_path = os.path.join(OUTPUT_DIR, os.path.basename(output_path).replace('.tif', '_probability.tif'))
    output_profile = reference_profile.copy()
    output_profile.update({
        'dtype': 'float32',
        'count': 1,
        'compress': 'lzw',
        'nodata': np.nan
    })

    with rasterio.open(prob_output_path, 'w', **output_profile) as dst:
        dst.write(probability_map.astype(np.float32), 1)

    print(f"Clipped probability map saved: {prob_output_path}")
    return probability_map, output_profile, transform, crs


def create_styled_clipped_map(probability_map, profile, transform, crs, study_area_gdf,
                              output_image_path, model_name="Model"):
    """
    Generates a styled fire risk map using a custom colormap.

    @param probability_map: 2D array of fire probabilities.
    @param profile: Raster profile.
    @param transform: Raster transform.
    @param crs: Coordinate reference system.
    @param study_area_gdf: GeoDataFrame of the study area.
    @param output_image_path: Path to save the styled map image.
    @param model_name: Name of the model for the title.
    """
    print(f"Creating styled clipped fire risk map for {model_name}...")

    colors = ['#ffff99', '#ffcc00', '#ff9900', '#ff6600', '#ff3300', '#cc0000', '#990000']
    fire_cmap = LinearSegmentedColormap.from_list('fire_risk', colors, N=100)

    bounds = rasterio.transform.array_bounds(probability_map.shape[0], probability_map.shape[1], transform)
    west, south, east, north = bounds

    fig, ax = plt.subplots(figsize=(15, 12))
    im = ax.imshow(probability_map, cmap=fire_cmap, vmin=0, vmax=1,
                   extent=[west, east, south, north], interpolation='bilinear')

    # Plot boundaries and basemap
    study_area_plot = study_area_gdf.to_crs(crs) if study_area_gdf.crs != crs else study_area_gdf
    study_area_plot.boundary.plot(ax=ax, color='black', linewidth=2)

    try:
        ctx.add_basemap(ax, crs=crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.3)
    except:
        print("Basemap not available.")
        ax.set_facecolor('lightgray')

    ax.set_title(f'Fire Risk Assessment Map - {model_name}', fontsize=18, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Fire Risk Probability', fontsize=14)

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=12)

    plt.tight_layout()

    # Save to the specified directory
    ensure_output_dir()
    full_output_path = os.path.join(OUTPUT_DIR, os.path.basename(output_image_path))
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Styled clipped map saved: {full_output_path}")


def load_and_clip_rasters(raster_paths, study_area_gdf):
    raster_data = {}
    reference_profile = None

    for var, path in raster_paths.items():
        with rasterio.open(path) as src:
            out_image, out_transform = mask(src, study_area_gdf.geometry, crop=True)
            out_image = out_image[0]
            out_image = np.where(out_image == src.nodata, np.nan, out_image)

            raster_data[var] = out_image

            if reference_profile is None:
                reference_profile = src.profile.copy()
                reference_profile.update({
                    "height": out_image.shape[0],
                    "width": out_image.shape[1],
                    "transform": out_transform,
                    "count": 1
                })

    return raster_data, reference_profile


def create_fdi_prediction_map(model, scaler, raster_data, reference_profile, output_path, selected_features,
                              current_date):
    height, width = reference_profile['height'], reference_profile['width']
    prediction_map = np.full((height, width), np.nan, dtype=np.float32)

    features = []
    for var in selected_features:
        if var in raster_data:
            features.append(raster_data[var]['data'].flatten())
        elif var == 'latitude':
            lat = np.linspace(
                reference_profile['transform'][5],
                reference_profile['transform'][5] + reference_profile['transform'][4] * height,
                height
            )
            lat = np.repeat(lat[:, np.newaxis], width, axis=1).flatten()
            features.append(lat)
        elif var == 'longitude':
            lon = np.linspace(
                reference_profile['transform'][2],
                reference_profile['transform'][2] + reference_profile['transform'][0] * width,
                width
            )
            lon = np.tile(lon, (height, 1)).flatten()
            features.append(lon)
        elif var == 'year':
            features.append(np.full(height * width, current_date.year))
        elif var == 'month':
            features.append(np.full(height * width, current_date.month))
        elif var == 'day':
            features.append(np.full(height * width, current_date.day))
        else:
            raise ValueError(f"Missing raster for variable: {var}")

    feature_matrix = np.column_stack(features)
    valid_mask = ~np.any(np.isnan(feature_matrix), axis=1)
    feature_matrix_scaled = scaler.transform(feature_matrix[valid_mask])

    probas = np.zeros(height * width)
    probas[valid_mask] = model.predict_proba(feature_matrix_scaled)[:, 1]
    prediction_map = probas.reshape((height, width))

    # Save to the specified directory
    ensure_output_dir()
    full_output_path = os.path.join(OUTPUT_DIR, os.path.basename(output_path))

    with rasterio.open(full_output_path, 'w', **reference_profile) as dst:
        dst.write(prediction_map, 1)

    return prediction_map


def visualize_fdi(prediction_map, reference_profile, study_area_gdf, output_image_path, model_name="Model"):
    """
    Visualize Fire Danger Index map with model name in title.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get bounds in raster CRS: (west, south, east, north)
    west, south, east, north = rasterio.transform.array_bounds(
        reference_profile['height'],
        reference_profile['width'],
        reference_profile['transform']
    )

    # extent expects [left, right, bottom, top] = [west, east, south, north]
    extent = [west, east, south, north]

    img = ax.imshow(prediction_map, cmap='RdYlGn_r', vmin=0, vmax=1, extent=extent, origin='upper')

    # Reproject study area to raster CRS if necessary
    raster_crs = reference_profile.get('crs', None)
    if raster_crs and study_area_gdf.crs != raster_crs:
        study_area_plot = study_area_gdf.to_crs(raster_crs)
    else:
        study_area_plot = study_area_gdf

    plt.colorbar(img, ax=ax, label="Fire Danger Index (ML Probability)")
    ax.set_title(f"Fire Danger Index Map - {model_name}", fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    plt.tight_layout()

    # Save to the specified directory
    ensure_output_dir()
    full_output_path = os.path.join(OUTPUT_DIR, os.path.basename(output_image_path))
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance for tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance - {model_name}', fontsize=16, fontweight='bold')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.ylabel('Importance Score')
        plt.tight_layout()

        # Save to the specified directory
        ensure_output_dir()
        filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        full_output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Print importance values
        print(f"\n{model_name} - Feature Importances:")
        for i in indices:
            print(f"{feature_names[i]}: {importances[i]:.4f}")
    else:
        print(f"{model_name} does not have feature_importances_ attribute")


def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot confusion matrix for a model.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Fire', 'Fire'],
                yticklabels=['No Fire', 'Fire'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    # Save to the specified directory
    ensure_output_dir()
    filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    full_output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Comprehensive model evaluation including accuracy, cross-validation, and confusion matrix.
    """
    print(f"\n{'=' * 50}")
    print(f"EVALUATION RESULTS FOR {model_name.upper()}")
    print(f"{'=' * 50}")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Accuracy scores
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Classification report
    print(f"\nClassification Report - {model_name}:")
    print(classification_report(y_test, y_pred_test, target_names=['No Fire', 'Fire']))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_test, model_name)

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }


def compare_models(results_dict):
    """
    Create a comparison chart of model performances.
    """
    models = list(results_dict.keys())
    test_accuracies = [results_dict[model]['test_accuracy'] for model in models]
    cv_means = [results_dict[model]['cv_mean'] for model in models]
    cv_stds = [results_dict[model]['cv_std'] for model in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width / 2, test_accuracies, width, label='Test Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width / 2, cv_means, width, yerr=cv_stds, label='CV Accuracy (±std)', alpha=0.8)

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()

    # Save to the specified directory
    ensure_output_dir()
    full_output_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main entry point for fire risk prediction and map generation with multiple models.
    """
    print("Loading input data...")
    print(f"Output directory set to: {OUTPUT_DIR}")
    ensure_output_dir()

    xls = pd.read_csv('//sungis15/Hons_scratch/25022318/Data/Data.csv')
    with rasterio.open('//sungis15/Hons_scratch/25022318/Reserve.tif') as src:
        bounds = src.bounds
        crs = src.crs
        nodata_value = src.nodata

    # Create study area polygon from raster bounds
    study_area = gpd.GeoDataFrame({'geometry': [box(*bounds)]}, crs=crs)

    # Load and display reserve base image
    with rasterio.open('//sungis15/Hons_scratch/25022318/Reserve.tif') as src:
        clipped_raster, _ = mask(src, study_area.geometry, crop=True)
        clipped_raster = clipped_raster.astype('float32')
        if nodata_value is not None:
            clipped_raster[clipped_raster == nodata_value] = np.nan
        plt.figure(figsize=(10, 8))
        plt.imshow(np.ma.masked_invalid(clipped_raster[0]), cmap='YlOrRd')
        plt.title('Study Area - Reserve Boundary')
        plt.colorbar(label='Reserve Data')

        # Save study area plot
        study_area_path = os.path.join(OUTPUT_DIR, 'study_area_reserve_boundary.png')
        plt.savefig(study_area_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Study area plot saved: {study_area_path}")

    # Raster input files
    raster_paths = {
        'mean_elevation': '//sungis15/Hons_scratch/25022318/713_Project/mean_elevation.tif',
        'mean_slope': '//sungis15/Hons_scratch/25022318/713_Project/mean_slope.tif',
        'mean_aspect': '//sungis15/Hons_scratch/25022318/713_Project/mean_aspect.tif',
        'avg_windspeed': '//sungis15/Hons_scratch/25022318/713_Project/avg_windspeed.tif',
        'total_rainfall': '//sungis15/Hons_scratch/25022318/713_Project/total_rainfall.tif',
        'mean_temp': '//sungis15/Hons_scratch/25022318/713_Project/mean_temp.tif',
        'avg_hum': '//sungis15/Hons_scratch/25022318/713_Project/avg_humidity.tif'
    }

    # Load raster features
    raster_data = {}
    reference_profile = None
    for name, path in raster_paths.items():
        raster_data[name] = clip_raster_to_study_area(path, study_area)
        if reference_profile is None:
            reference_profile = raster_data[name]['profile']

    # Preprocess tabular data
    xls['Acq_date'] = pd.to_datetime(xls['Acq_date'])
    xls['year'] = xls['Acq_date'].dt.year
    xls['month'] = xls['Acq_date'].dt.month
    xls['day'] = xls['Acq_date'].dt.day

    selected_columns = [
        'mean_elevation', 'mean_slope', 'mean_aspect',
        'avg_windspeed', 'total_rainfall', 'mean_temp', 'avg_hum'
    ]

    # Upsample fire cases
    majority = xls[xls['burnt'] == 0]
    minority = xls[xls['burnt'] == 1]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    balanced_data = pd.concat([majority, minority_upsampled]).sample(frac=1).reset_index(drop=True)

    X = balanced_data[selected_columns]
    y = balanced_data['burnt']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
    }

    # Train models and evaluate
    results = {}
    trained_models = {}
    current_date = datetime(2023, 12, 15)

    for model_name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'=' * 60}")

        # Train model
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model

        # Evaluate model
        results[model_name] = evaluate_model(model, X_train_scaled, X_test_scaled,
                                             y_train, y_test, model_name)

        # Plot feature importance (for tree-based models only)
        plot_feature_importance(model, selected_columns, model_name)

        # Create prediction map for ALL models (including SVM)
        print(f"\nGenerating prediction map for {model_name}...")
        output_path = f'fdi_prediction_{model_name.lower().replace(" ", "_")}.tif'
        prediction_map = create_fdi_prediction_map(
            model=model,
            scaler=scaler,
            raster_data=raster_data,
            reference_profile=reference_profile,
            output_path=output_path,
            selected_features=selected_columns,
            current_date=current_date
        )

        # Visualize prediction map for ALL models (including SVM)
        output_image_path = f'fdi_map_{model_name.lower().replace(" ", "_")}.png'
        visualize_fdi(prediction_map, reference_profile, study_area,
                      output_image_path, model_name)

        print(f"Prediction map and visualization completed for {model_name}")

        # Compare all models
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    compare_models(results)

    # Print summary table
    print(f"\n{'Model':<20} {'Test Acc':<10} {'CV Mean':<10} {'CV Std':<10}")
    print("-" * 50)
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['test_accuracy']:<10.4f} "
              f"{result['cv_mean']:<10.4f} {result['cv_std']:<10.4f}")


if __name__ == "__main__":
    main()