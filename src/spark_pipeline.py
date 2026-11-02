"""
PySpark Big Data Pipeline for E-Commerce Return Intelligence.

Fulfills Academic Tasks:
Task 121: Load dataset into a Spark DataFrame, inspect schema, remove duplicates, handle missing values.
Task 122: Encode categorical features using StringIndexer, OneHotEncoder, and VectorAssembler.
Task 123: Train Logistic Regression and Random Forest using Spark MLlib.
Task 124: Evaluate models using precision, recall, F1-score, and ROC-AUC.
Task 125: Generate return predictions using the selected Spark MLlib model.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

# Ensure Java environment is configured if available
if not os.environ.get("JAVA_HOME"):
    candidate_paths = [
        r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot",
        r"C:\Java\jdk-1.8",
        r"C:\Java\latest"
    ]
    for p in candidate_paths:
        if os.path.isdir(p):
            os.environ["JAVA_HOME"] = p
            break


def run_spark_pipeline(
    data_path: str = "dataset/Ecommerce_2000.csv",
    output_dir: str = "models",
    predictions_dir: str = "output"
) -> Dict[str, Any]:
    """
    Execute end-to-end PySpark Data Processing, Modeling, and Evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    print("=" * 70)
    print("INITIALIZING PYSPARK BIG DATA PIPELINE")
    print("=" * 70)

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, when, lit
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
        from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
        from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    except ImportError as e:
        print(f"PySpark import error: {e}")
        return {"status": "error", "message": str(e)}

    # Initialize Spark Session optimized for single-node cluster / local execution
    spark = SparkSession.builder \
        .appName("EcommerceReturnIntelligence") \
        .master("local[*]") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.ui.enabled", "false") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print(f"Spark Session established. Spark Version: {spark.version}")

    # =========================================================================
    # TASK 121: Load Dataset, Inspect Schema, Remove Duplicates, Handle Nulls
    # =========================================================================
    print("\n" + "=" * 70)
    print("TASK 121: Spark DataFrame Ingestion, Schema Inspection & Cleansing")
    print("=" * 70)

    # 1. Load CSV into Spark DataFrame
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    raw_count = df.count()
    print(f"Raw Spark DataFrame loaded: {raw_count} rows, {len(df.columns)} columns")
    print("\nDataFrame Schema:")
    df.printSchema()

    # 2. Remove Duplicates
    df_dedup = df.dropDuplicates()
    dedup_count = df_dedup.count()
    duplicates_removed = raw_count - dedup_count
    print(f"Deduplicated count: {dedup_count} rows ({duplicates_removed} duplicates removed)")

    # 3. Handle Target Binarization (Yes -> 1.0, No -> 0.0)
    df_clean = df_dedup.withColumn(
        "label",
        when(col("Returned") == "Yes", 1.0).otherwise(0.0)
    )

    # 4. Defensive Imputation for Nulls
    # Impute numeric columns with 0.0 or median fallback
    numeric_cols = [
        "UnitPrice", "Quantity", "TotalAmount",
        "DiscountPercent", "DiscountAmount", "FinalAmount", "ShippingDays"
    ]
    for c in numeric_cols:
        df_clean = df_clean.withColumn(c, when(col(c).isNull(), lit(0.0)).otherwise(col(c).cast("double")))

    # Impute categorical columns
    cat_cols = [
        "ProductCategory", "ProductSubCategory", "PaymentMethod",
        "DeviceType", "MembershipTier", "OrderSeason", "OrderDayOfWeek"
    ]
    for c in cat_cols:
        df_clean = df_clean.withColumn(c, when(col(c).isNull(), lit("Unknown")).otherwise(col(c)))

    # 5. Drop Identifiers & Leakage (ReviewScore, OrderID, CustomerID, FirstName, LastName)
    drop_cols = ["OrderID", "CustomerID", "FirstName", "LastName", "ReviewScore", "Returned"]
    df_modeling = df_clean.drop(*[c for c in drop_cols if c in df_clean.columns])

    # =========================================================================
    # TASK 122: Feature Engineering & Categorical Encoding
    # =========================================================================
    print("\n" + "=" * 70)
    print("TASK 122: Feature Engineering & Categorical Encoding")
    print("=" * 70)

    # Derived Big Data engineered columns
    df_modeling = df_modeling.withColumn("DiscountRatio", col("DiscountAmount") / (col("TotalAmount") + lit(1e-5))) \
                             .withColumn("EffectiveUnitPrice", col("FinalAmount") / (col("Quantity") + lit(1e-5))) \
                             .withColumn("ShippingPerDollar", col("ShippingDays") / (col("FinalAmount") + lit(1e-5))) \
                             .withColumn("IsHighDiscount", when(col("DiscountPercent") >= 20, 1.0).otherwise(0.0)) \
                             .withColumn("IsBulkOrder", when(col("Quantity") >= 3, 1.0).otherwise(0.0))

    engineered_numeric_cols = numeric_cols + [
        "DiscountRatio", "EffectiveUnitPrice", "ShippingPerDollar", "IsHighDiscount", "IsBulkOrder"
    ]

    # StringIndexer and OneHotEncoder for categorical features
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
        for c in cat_cols
    ]

    assembler_inputs = engineered_numeric_cols + [f"{c}_vec" for c in cat_cols]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)

    feature_pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
    pipeline_model = feature_pipeline.fit(df_modeling)
    df_transformed = pipeline_model.transform(df_modeling).select("features", "label")

    print(f"Features assembled and scaled. Vector size: {len(assembler_inputs)} base feature blocks.")

    # Split into Train (80%) and Test (20%)
    train_data, test_data = df_transformed.randomSplit([0.80, 0.20], seed=42)
    train_count = train_data.count()
    test_count = test_data.count()
    print(f"Dataset split: Train={train_count} rows, Test={test_count} rows")

    # =========================================================================
    # TASK 123: Train Logistic Regression and Random Forest with Spark MLlib
    # =========================================================================
    print("\n" + "=" * 70)
    print("TASK 123: Training Spark MLlib Logistic Regression & Random Forest")
    print("=" * 70)

    # Logistic Regression
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=50,
        regParam=0.1,
        elasticNetParam=0.0
    )
    print("Fitting Spark MLlib Logistic Regression...")
    spark_lr_model = lr.fit(train_data)

    # Random Forest Classifier
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=8,
        seed=42
    )
    print("Fitting Spark MLlib Random Forest Classifier...")
    spark_rf_model = rf.fit(train_data)

    # =========================================================================
    # TASK 124: Evaluate Models (Precision, Recall, F1-Score, ROC-AUC)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TASK 124: Model Evaluation on Spark Test DataFrame")
    print("=" * 70)

    # Predictions
    lr_preds = spark_lr_model.transform(test_data)
    rf_preds = spark_rf_model.transform(test_data)

    roc_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    pr_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    
    acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    prec_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    rec_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

    def get_spark_metrics(pred_df):
        return {
            "accuracy": round(float(acc_evaluator.evaluate(pred_df)), 4),
            "precision": round(float(prec_evaluator.evaluate(pred_df)), 4),
            "recall": round(float(rec_evaluator.evaluate(pred_df)), 4),
            "f1_score": round(float(f1_evaluator.evaluate(pred_df)), 4),
            "roc_auc": round(float(roc_evaluator.evaluate(pred_df)), 4),
            "pr_auc": round(float(pr_evaluator.evaluate(pred_df)), 4)
        }

    lr_metrics = get_spark_metrics(lr_preds)
    rf_metrics = get_spark_metrics(rf_preds)

    print("\n--- Spark MLlib Logistic Regression Metrics ---")
    for k, v in lr_metrics.items():
        print(f"  {k:12s}: {v}")

    print("\n--- Spark MLlib Random Forest Metrics ---")
    for k, v in rf_metrics.items():
        print(f"  {k:12s}: {v}")

    # =========================================================================
    # TASK 125: Generate Return Predictions with Selected Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("TASK 125: Generating Predictions with Best Spark Model")
    print("=" * 70)

    if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"]:
        selected_spark_name = "Spark MLlib Random Forest"
        selected_preds = rf_preds
        best_metrics = rf_metrics
    else:
        selected_spark_name = "Spark MLlib Logistic Regression"
        selected_preds = lr_preds
        best_metrics = lr_metrics

    print(f"Selected Spark Model: {selected_spark_name}")
    print("\nSample Return Predictions (First 10 rows):")
    selected_preds.select("prediction", "probability", "label").show(10, truncate=False)

    # Save Spark metrics summary
    spark_results = {
        "framework": f"PySpark {spark.version}",
        "execution_mode": "Spark Local Multi-Core Distributed DataFrame",
        "timestamp": datetime.now().isoformat(),
        "total_records": raw_count,
        "clean_records": dedup_count,
        "selected_model": selected_spark_name,
        "models": {
            "Logistic Regression": lr_metrics,
            "Random Forest": rf_metrics
        }
    }

    metrics_save_path = os.path.join(output_dir, "spark_metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(spark_results, f, indent=2)
    print(f"Spark evaluation results persisted to {metrics_save_path}")

    # Export a sample of predictions to CSV for verification
    pred_sample_pd = selected_preds.select("label", "prediction").limit(100).toPandas()
    pred_sample_pd.to_csv(os.path.join(predictions_dir, "spark_sample_predictions.csv"), index=False)
    print("Sample Spark predictions exported to output/spark_sample_predictions.csv")

    spark.stop()
    print("Spark Session cleanly stopped.")
    return spark_results


if __name__ == "__main__":
    run_spark_pipeline()
