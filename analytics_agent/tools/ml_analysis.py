"""Tools for machine learning analysis and modeling."""

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import StructuredTool
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from analytics_agent.tools.base import BaseTool


class MLAnalysisTools(BaseTool):
    """Machine Learning analysis tools for the analytics agent."""

    def __init__(self, output_dir: str = "exports") -> None:
        """Initialize ML analysis tools.

        Args:
            output_dir: Directory to save ML outputs and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def format_tool_call(tool_name: str, tool_args: dict) -> str:
        """Format a tool call for display in CLI.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool

        Returns:
            Formatted string representation of the tool call
        """
        output = f"🤖 Calling ML Tool: {tool_name}"

        if tool_name == "profile_dataset":
            output += "\n   Analyzing dataset statistics..."
        elif tool_name == "feature_correlation":
            output += "\n   Computing feature correlations..."
        elif tool_name in ["train_linear_regression", "train_logistic_regression"]:
            if "target_column" in tool_args:
                output += f"\n   Target: {tool_args['target_column']}"
        elif tool_name == "kmeans_cluster":
            if "n_clusters" in tool_args:
                output += f"\n   Clusters: {tool_args['n_clusters']}"
        elif tool_name == "moving_average_forecast":
            if "window_size" in tool_args:
                output += f"\n   Window Size: {tool_args['window_size']}"

        return output

    @staticmethod
    def format_tool_result(tool_name: str, result_content: str) -> str:
        """Format a tool result for display in CLI.

        Args:
            tool_name: Name of the tool that was called
            result_content: The result content returned by the tool

        Returns:
            Formatted string representation of the tool result
        """
        output = "✅ ML Tool result:"

        if "Error:" in result_content:
            output += f"\n   ❌ {result_content}"
        elif "Saved to:" in result_content:
            file_path = result_content.split("Saved to: ")[-1].split("\n")[0]
            output += f"\n   📊 Results saved: {file_path}"
        else:
            output += f"\n   {result_content[:200]}..."

        return output

    def _parse_data(self, data: str) -> pd.DataFrame:
        """Parse CSV data string into a pandas DataFrame.

        Args:
            data: Data as CSV string with headers in the first row

        Returns:
            DataFrame containing the parsed data

        Raises:
            ValueError: If data cannot be parsed as CSV
        """
        try:
            df = pd.read_csv(io.StringIO(data))
            if df.empty:
                raise ValueError("CSV data is empty")
            return df
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
            raise ValueError(
                f"Data must be in CSV format with headers in the first row. Error: {str(e)}"
            ) from e

    def profile_dataset(
        self,
        data: Annotated[
            str,
            "Data to profile as CSV string with headers in the first row",
        ],
    ) -> str:
        """Generate comprehensive statistical profile of a dataset.

        Args:
            data: Data as CSV string with headers in the first row

        Returns:
            Statistical profile including shape, data types, missing values, and descriptive stats
        """
        try:
            df = self._parse_data(data)

            profile = {
                "shape": {"rows": len(df), "columns": len(df.columns)},
                "columns": list(df.columns),
                "data_types": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            }

            # Get descriptive statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                desc_stats = df[numeric_cols].describe().to_dict()
                profile["numeric_summary"] = desc_stats

            # Get value counts for categorical columns
            categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if categorical_cols:
                cat_summary = {}
                for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                    cat_summary[col] = {
                        "unique_values": int(df[col].nunique()),
                        "top_values": df[col].value_counts().head(5).to_dict(),
                    }
                profile["categorical_summary"] = cat_summary

            # Format output
            output = "Dataset Profile:\n"
            output += f"Shape: {profile['shape']['rows']} rows × {profile['shape']['columns']} columns\n\n"
            output += f"Columns: {', '.join(profile['columns'])}\n\n"
            output += "Missing Values:\n"
            for col, count in profile["missing_values"].items():
                pct = profile["missing_percentage"][col]
                output += f"  {col}: {count} ({pct}%)\n"

            if "numeric_summary" in profile:
                output += "\nNumeric Columns Summary:\n"
                output += json.dumps(profile["numeric_summary"], indent=2)

            if "categorical_summary" in profile:
                output += "\n\nCategorical Columns Summary:\n"
                output += json.dumps(profile["categorical_summary"], indent=2)

            return output

        except Exception as e:
            return f"Error profiling dataset: {str(e)}"

    def feature_correlation(
        self,
        data: Annotated[
            str,
            "Data to analyze as CSV string with headers in the first row",
        ],
        method: Annotated[str, "Correlation method: 'pearson', 'spearman', or 'kendall'"] = "pearson",
        output_path: Annotated[
            Optional[str], "Custom output path for the correlation heatmap"
        ] = None,
    ) -> str:
        """Calculate and visualize correlation matrix between numeric features.

        Args:
            data: Data as CSV string with headers in the first row
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            output_path: Custom output path for visualization

        Returns:
            Correlation analysis results and path to visualization
        """
        try:
            df = self._parse_data(data)

            # Select only numeric columns
            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.empty:
                return "Error: No numeric columns found in the dataset"

            if len(numeric_df.columns) < 2:
                return "Error: Need at least 2 numeric columns to calculate correlation"

            # Calculate correlation matrix
            corr_matrix = numeric_df.corr(method=method)

            # Find highly correlated pairs (absolute correlation > 0.7)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        high_corr_pairs.append(
                            {
                                "feature1": corr_matrix.columns[i],
                                "feature2": corr_matrix.columns[j],
                                "correlation": round(corr_value, 3),
                            }
                        )

            # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                fmt=".2f",
                square=True,
                linewidths=0.5,
            )
            plt.title(f"Feature Correlation Matrix ({method.capitalize()})")
            plt.tight_layout()

            # Save visualization
            if output_path:
                filepath = Path(output_path)
                if not filepath.is_absolute():
                    filepath = self.output_dir / filepath
                filepath.parent.mkdir(parents=True, exist_ok=True)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"correlation_heatmap_{timestamp}.png"
                filepath = self.output_dir / filename

            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            # Format output
            output = f"Feature Correlation Analysis ({method}):\n\n"
            output += f"Analyzed {len(numeric_df.columns)} numeric features\n"
            output += f"Columns: {', '.join(numeric_df.columns)}\n\n"

            if high_corr_pairs:
                output += "Highly Correlated Pairs (|r| > 0.7):\n"
                for pair in high_corr_pairs:
                    output += f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']}\n"
            else:
                output += "No highly correlated pairs found (|r| > 0.7)\n"

            output += f"\nVisualization saved to: {filepath}"

            return output

        except Exception as e:
            return f"Error analyzing correlations: {str(e)}"

    def train_linear_regression(
        self,
        data: Annotated[
            str,
            "Training data as CSV string with headers in the first row",
        ],
        target_column: Annotated[str, "Name of the target column to predict"],
        feature_columns: Annotated[
            Optional[str],
            "Comma-separated list of feature columns (if not provided, uses all numeric columns except target)",
        ] = None,
        test_size: Annotated[float, "Proportion of data to use for testing (0.0-1.0)"] = 0.2,
    ) -> str:
        """Train a linear regression model and evaluate its performance.

        Args:
            data: Training data as CSV string
            target_column: Name of the target column
            feature_columns: Optional comma-separated feature column names
            test_size: Proportion of data for testing

        Returns:
            Model performance metrics and evaluation results
        """
        try:
            df = self._parse_data(data)

            if target_column not in df.columns:
                return f"Error: Target column '{target_column}' not found. Available: {', '.join(df.columns)}"

            # Select features
            if feature_columns:
                features = [f.strip() for f in feature_columns.split(",")]
                for feat in features:
                    if feat not in df.columns:
                        return f"Error: Feature column '{feat}' not found"
            else:
                # Use all numeric columns except target
                features = [
                    col
                    for col in df.select_dtypes(include=[np.number]).columns
                    if col != target_column
                ]

            if not features:
                return "Error: No valid feature columns found"

            # Prepare data
            X = df[features].fillna(df[features].mean())
            y = df[target_column].fillna(df[target_column].mean())

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            # Feature importance (coefficients)
            feature_importance = dict(zip(features, model.coef_))
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )

            # Format output
            output = "Linear Regression Model Results:\n\n"
            output += f"Target: {target_column}\n"
            output += f"Features: {', '.join(features)}\n"
            output += f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n\n"
            output += "Performance Metrics:\n"
            output += f"  Training R²: {train_r2:.4f}\n"
            output += f"  Test R²: {test_r2:.4f}\n"
            output += f"  Training RMSE: {train_rmse:.4f}\n"
            output += f"  Test RMSE: {test_rmse:.4f}\n"
            output += f"  Training MAE: {train_mae:.4f}\n"
            output += f"  Test MAE: {test_mae:.4f}\n\n"
            output += "Feature Coefficients (sorted by absolute value):\n"
            for feat, coef in sorted_features[:10]:  # Top 10 features
                output += f"  {feat}: {coef:.4f}\n"
            output += f"\nIntercept: {model.intercept_:.4f}\n"

            # Create visualization
            plt.figure(figsize=(12, 5))

            # Actual vs Predicted
            plt.subplot(1, 2, 1)
            plt.scatter(y_test, y_test_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"Actual vs Predicted (R² = {test_r2:.4f})")

            # Residuals
            plt.subplot(1, 2, 2)
            residuals = y_test - y_test_pred
            plt.scatter(y_test_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color="r", linestyle="--", lw=2)
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.title("Residual Plot")

            plt.tight_layout()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"linear_regression_{timestamp}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            output += f"\nVisualization saved to: {filepath}"

            return output

        except Exception as e:
            return f"Error training linear regression: {str(e)}"

    def train_logistic_regression(
        self,
        data: Annotated[
            str,
            "Training data as CSV string with headers in the first row",
        ],
        target_column: Annotated[str, "Name of the target column (binary classification)"],
        feature_columns: Annotated[
            Optional[str],
            "Comma-separated list of feature columns (if not provided, uses all numeric columns except target)",
        ] = None,
        test_size: Annotated[float, "Proportion of data to use for testing (0.0-1.0)"] = 0.2,
    ) -> str:
        """Train a logistic regression classifier and evaluate its performance.

        Args:
            data: Training data as CSV string
            target_column: Name of the target column (binary classification)
            feature_columns: Optional comma-separated feature column names
            test_size: Proportion of data for testing

        Returns:
            Model performance metrics and classification report
        """
        try:
            df = self._parse_data(data)

            if target_column not in df.columns:
                return f"Error: Target column '{target_column}' not found. Available: {', '.join(df.columns)}"

            # Select features
            if feature_columns:
                features = [f.strip() for f in feature_columns.split(",")]
                for feat in features:
                    if feat not in df.columns:
                        return f"Error: Feature column '{feat}' not found"
            else:
                # Use all numeric columns except target
                features = [
                    col
                    for col in df.select_dtypes(include=[np.number]).columns
                    if col != target_column
                ]

            if not features:
                return "Error: No valid feature columns found"

            # Prepare data
            X = df[features].fillna(df[features].mean())
            y = df[target_column]

            # Check if binary classification
            unique_classes = y.nunique()
            if unique_classes > 10:
                return f"Error: Target has {unique_classes} unique values. Logistic regression is best for binary/multi-class classification (<=10 classes)"

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Get classification report
            report = classification_report(y_test, y_test_pred)

            # Format output
            output = "Logistic Regression Model Results:\n\n"
            output += f"Target: {target_column}\n"
            output += f"Features: {', '.join(features)}\n"
            output += f"Classes: {sorted(y.unique())}\n"
            output += f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n\n"
            output += "Performance Metrics:\n"
            output += f"  Training Accuracy: {train_accuracy:.4f}\n"
            output += f"  Test Accuracy: {test_accuracy:.4f}\n\n"
            output += "Classification Report:\n"
            output += report

            # Feature importance (coefficients)
            if len(model.classes_) == 2:
                feature_importance = dict(zip(features, model.coef_[0]))
            else:
                # For multi-class, use average absolute coefficient
                feature_importance = dict(zip(features, np.abs(model.coef_).mean(axis=0)))

            sorted_features = sorted(
                feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )

            output += "\nTop 10 Feature Coefficients:\n"
            for feat, coef in sorted_features[:10]:
                output += f"  {feat}: {coef:.4f}\n"

            return output

        except Exception as e:
            return f"Error training logistic regression: {str(e)}"

    def kmeans_cluster(
        self,
        data: Annotated[
            str,
            "Data to cluster as CSV string with headers in the first row",
        ],
        n_clusters: Annotated[int, "Number of clusters to create"] = 3,
        feature_columns: Annotated[
            Optional[str],
            "Comma-separated list of feature columns (if not provided, uses all numeric columns)",
        ] = None,
        output_path: Annotated[Optional[str], "Custom output path for cluster visualization"] = None,
    ) -> str:
        """Perform K-means clustering analysis on the data.

        Args:
            data: Data as CSV string
            n_clusters: Number of clusters to create
            feature_columns: Optional comma-separated feature column names
            output_path: Custom output path for visualization

        Returns:
            Clustering results including cluster statistics and visualization
        """
        try:
            df = self._parse_data(data)

            # Select features
            if feature_columns:
                features = [f.strip() for f in feature_columns.split(",")]
                for feat in features:
                    if feat not in df.columns:
                        return f"Error: Feature column '{feat}' not found"
            else:
                features = df.select_dtypes(include=[np.number]).columns.tolist()

            if not features:
                return "Error: No valid numeric feature columns found"

            if len(features) < 2:
                return "Error: Need at least 2 features for clustering"

            # Prepare data
            X = df[features].fillna(df[features].mean())
            X_array = X.values  # Convert to numpy array for indexing

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)

            # Calculate metrics
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            inertia = kmeans.inertia_

            # Add cluster labels to original dataframe
            df["cluster"] = cluster_labels

            # Calculate cluster statistics
            cluster_stats = df.groupby("cluster")[features].mean()

            # Format output
            output = "K-Means Clustering Results:\n\n"
            output += f"Number of clusters: {n_clusters}\n"
            output += f"Features used: {', '.join(features)}\n"
            output += f"Data points: {len(df)}\n\n"
            output += "Cluster Quality Metrics:\n"
            output += f"  Silhouette Score: {silhouette_avg:.4f} (range: -1 to 1, higher is better)\n"
            output += f"  Inertia: {inertia:.2f}\n\n"
            output += "Cluster Sizes:\n"
            for i in range(n_clusters):
                count = (cluster_labels == i).sum()
                pct = count / len(cluster_labels) * 100
                output += f"  Cluster {i}: {count} points ({pct:.1f}%)\n"

            output += "\nCluster Centers (mean values):\n"
            output += cluster_stats.to_string()

            # Create visualization (2D projection using first 2 features or PCA)
            plt.figure(figsize=(12, 5))

            if len(features) >= 2:
                # Plot using first two features
                plt.subplot(1, 2, 1)
                scatter = plt.scatter(
                    X_array[:, 0], X_array[:, 1], c=cluster_labels, cmap="viridis", alpha=0.6
                )
                plt.scatter(
                    kmeans.cluster_centers_[:, 0],
                    kmeans.cluster_centers_[:, 1],
                    c="red",
                    marker="X",
                    s=200,
                    edgecolors="black",
                    label="Centroids",
                )
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                plt.title("K-Means Clusters")
                plt.colorbar(scatter, label="Cluster")
                plt.legend()

            # Cluster size bar chart
            plt.subplot(1, 2, 2)
            cluster_sizes = [
                (cluster_labels == i).sum() for i in range(n_clusters)
            ]
            plt.bar(range(n_clusters), cluster_sizes, color="skyblue", edgecolor="black")
            plt.xlabel("Cluster")
            plt.ylabel("Number of Points")
            plt.title("Cluster Distribution")
            plt.xticks(range(n_clusters))

            plt.tight_layout()

            # Save visualization
            if output_path:
                filepath = Path(output_path)
                if not filepath.is_absolute():
                    filepath = self.output_dir / filepath
                filepath.parent.mkdir(parents=True, exist_ok=True)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"kmeans_clusters_{timestamp}.png"
                filepath = self.output_dir / filename

            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            output += f"\n\nVisualization saved to: {filepath}"

            return output

        except Exception as e:
            return f"Error performing clustering: {str(e)}"

    def moving_average_forecast(
        self,
        data: Annotated[
            str,
            "Time series data as CSV string with headers in the first row",
        ],
        value_column: Annotated[str, "Name of the column containing values to forecast"],
        time_column: Annotated[
            Optional[str], "Name of the time/date column (optional, will use row index if not provided)"
        ] = None,
        window_size: Annotated[int, "Window size for moving average"] = 7,
        forecast_periods: Annotated[int, "Number of periods to forecast"] = 5,
        output_path: Annotated[Optional[str], "Custom output path for forecast visualization"] = None,
    ) -> str:
        """Perform moving average forecasting on time series data.

        Args:
            data: Time series data as CSV string
            value_column: Column containing values to forecast
            time_column: Optional time/date column
            window_size: Window size for moving average
            forecast_periods: Number of periods to forecast
            output_path: Custom output path for visualization

        Returns:
            Forecast results and visualization
        """
        try:
            df = self._parse_data(data)

            if value_column not in df.columns:
                return f"Error: Value column '{value_column}' not found. Available: {', '.join(df.columns)}"

            # Prepare data
            if time_column and time_column in df.columns:
                try:
                    df[time_column] = pd.to_datetime(df[time_column])
                    df = df.sort_values(time_column).reset_index(drop=True)
                    time_values = df[time_column]
                except Exception:
                    time_values = pd.Series(range(len(df)))
            else:
                time_values = pd.Series(range(len(df)))

            values = df[value_column].ffill().bfill()

            if len(values) < window_size:
                return f"Error: Need at least {window_size} data points for window size {window_size}"

            # Calculate moving average
            moving_avg = values.rolling(window=window_size).mean()

            # Simple forecast: use last moving average value
            last_ma = moving_avg.iloc[-1]
            forecast_values = [last_ma] * forecast_periods

            # Calculate error metrics on historical data where MA is available
            valid_idx = moving_avg.notna()
            if valid_idx.sum() > 0:
                mae = mean_absolute_error(values[valid_idx], moving_avg[valid_idx])
                rmse = np.sqrt(mean_squared_error(values[valid_idx], moving_avg[valid_idx]))
            else:
                mae = rmse = 0

            # Format output
            output = "Moving Average Forecast Results:\n\n"
            output += f"Value column: {value_column}\n"
            output += f"Window size: {window_size}\n"
            output += f"Historical data points: {len(values)}\n"
            output += f"Forecast periods: {forecast_periods}\n\n"
            output += "Historical Performance:\n"
            output += f"  MAE: {mae:.4f}\n"
            output += f"  RMSE: {rmse:.4f}\n\n"
            output += "Forecast values:\n"
            for i, val in enumerate(forecast_values, 1):
                output += f"  Period +{i}: {val:.4f}\n"

            # Create visualization
            plt.figure(figsize=(12, 6))

            # Plot historical data
            plt.plot(range(len(values)), values, label="Actual", marker="o", alpha=0.7)
            plt.plot(range(len(moving_avg)), moving_avg, label=f"MA({window_size})", linewidth=2)

            # Plot forecast
            forecast_x = range(len(values), len(values) + forecast_periods)
            plt.plot(
                forecast_x,
                forecast_values,
                label="Forecast",
                marker="o",
                linestyle="--",
                color="red",
            )

            # Shade forecast region
            plt.axvspan(len(values) - 0.5, len(values) + forecast_periods - 0.5, alpha=0.2, color="gray")

            plt.xlabel("Time Period")
            plt.ylabel(value_column)
            plt.title(f"Moving Average Forecast (Window={window_size})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save visualization
            if output_path:
                filepath = Path(output_path)
                if not filepath.is_absolute():
                    filepath = self.output_dir / filepath
                filepath.parent.mkdir(parents=True, exist_ok=True)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"moving_average_forecast_{timestamp}.png"
                filepath = self.output_dir / filename

            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            output += f"\nVisualization saved to: {filepath}"

            return output

        except Exception as e:
            return f"Error performing forecast: {str(e)}"

    def get_tools(self) -> List[StructuredTool]:
        """Get list of LangChain tools.

        Returns:
            List of StructuredTool instances
        """
        return [
            StructuredTool.from_function(
                func=self.profile_dataset,
                name="profile_dataset",
                description=(
                    "Generate a comprehensive statistical profile of a dataset including shape, "
                    "data types, missing values, descriptive statistics for numeric columns, "
                    "and value counts for categorical columns. Input must be CSV format with headers."
                ),
            ),
            StructuredTool.from_function(
                func=self.feature_correlation,
                name="feature_correlation",
                description=(
                    "Calculate and visualize correlation matrix between numeric features in the dataset. "
                    "Supports Pearson, Spearman, and Kendall correlation methods. Creates a heatmap "
                    "visualization and identifies highly correlated feature pairs. Input must be CSV format."
                ),
            ),
            StructuredTool.from_function(
                func=self.train_linear_regression,
                name="train_linear_regression",
                description=(
                    "Train a linear regression model to predict a continuous target variable. "
                    "Automatically splits data into train/test sets, evaluates performance using R², "
                    "RMSE, and MAE metrics, and shows feature coefficients. Creates visualization "
                    "of actual vs predicted values and residuals. Input must be CSV format."
                ),
            ),
            StructuredTool.from_function(
                func=self.train_logistic_regression,
                name="train_logistic_regression",
                description=(
                    "Train a logistic regression classifier for binary or multi-class classification. "
                    "Automatically scales features, splits data, and evaluates using accuracy and "
                    "classification report (precision, recall, F1-score). Shows feature importance. "
                    "Best for binary or multi-class (up to 10 classes). Input must be CSV format."
                ),
            ),
            StructuredTool.from_function(
                func=self.kmeans_cluster,
                name="kmeans_cluster",
                description=(
                    "Perform K-means clustering analysis to group similar data points. "
                    "Automatically scales features, calculates silhouette score for cluster quality, "
                    "and provides cluster statistics. Creates visualization showing cluster distribution "
                    "and centers. Input must be CSV format with numeric features."
                ),
            ),
            StructuredTool.from_function(
                func=self.moving_average_forecast,
                name="moving_average_forecast",
                description=(
                    "Perform simple moving average forecasting on time series data. "
                    "Calculates moving average with specified window size and forecasts future periods. "
                    "Shows MAE and RMSE for historical fit. Creates visualization of historical data, "
                    "moving average, and forecast. Input must be CSV format with a value column."
                ),
            ),
        ]

