"""
Model Training, Tuning, and Selection Module for E-Commerce Return Intelligence.

Implements stratified splitting (70% Train, 15% Validation, 15% Test),
imbalance handling with balanced class weighting, validation-driven hyperparameter tuning,
threshold optimization, and comprehensive evaluation on untouched test data.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import prepare_training_data, TARGET_COLUMN
from src.feature_engineering import extract_features, build_preprocessor, get_feature_names
from src.evaluation import evaluate_predictions, find_optimal_threshold, compute_curve_data


def train_and_evaluate(
    data_path: str = "dataset/Ecommerce_2000.csv",
    output_dir: str = "models",
    predictions_dir: str = "output"
) -> Dict[str, Any]:
    """
    Complete end-to-end training pipeline:
    1. Preprocess raw dataset and extract features.
    2. Split into Stratified Train (70%), Validation (15%), and Test (15%).
    3. Fit feature preprocessor on training data only.
    4. Train & tune Logistic Regression and Random Forest.
    5. Optimize classification thresholds on validation data.
    6. Select the superior model based on validation F1 and ROC-AUC.
    7. Perform final evaluation on the untouched Test set.
    8. Persist all models, metadata, curve data, and evaluation reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    print("=" * 60)
    print("STEP 1: Data Preparation & Feature Extraction")
    print("=" * 60)
    modeling_df, raw_dedup_df, dup_count = prepare_training_data(data_path)
    print(f"Loaded records: {len(raw_dedup_df)} unique records ({dup_count} duplicates removed)")
    
    # Feature engineering on modeling data
    df_features = extract_features(modeling_df)
    
    X = df_features.drop(columns=[TARGET_COLUMN])
    y = df_features[TARGET_COLUMN].values
    
    print(f"Total features: {X.shape[1]} | Class Distribution: Negative={np.sum(y==0)}, Positive={np.sum(y==1)} ({np.mean(y):.2%})")
    
    # STEP 2: Stratified Split: 70% Train, 15% Validation, 15% Test
    print("\n" + "=" * 60)
    print("STEP 2: Stratified Train / Validation / Test Split")
    print("=" * 60)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    print(f"Train set: {len(y_train)} samples ({np.sum(y_train)} returns, {np.mean(y_train):.2%})")
    print(f"Val set:   {len(y_val)} samples ({np.sum(y_val)} returns, {np.mean(y_val):.2%})")
    print(f"Test set:  {len(y_test)} samples ({np.sum(y_test)} returns, {np.mean(y_test):.2%}) [Untouched]")
    
    # STEP 3: Fit Preprocessor on Training Split
    print("\n" + "=" * 60)
    print("STEP 3: Fitting Preprocessor (StandardScaler + OneHotEncoder)")
    print("=" * 60)
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    feature_names = get_feature_names(preprocessor)
    print(f"Transformed feature vector dimensionality: {X_train.shape[1]}")
    
    # STEP 4: Train & Tune Logistic Regression
    print("\n" + "=" * 60)
    print("STEP 4: Training & Tuning Logistic Regression")
    print("=" * 60)
    best_lr_score = -1.0
    best_lr_c = 1.0
    best_lr_model = None
    
    # Hyperparameter tuning over C using validation data
    for c_val in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        lr = LogisticRegression(
            C=c_val,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        )
        lr.fit(X_train, y_train)
        val_probs = lr.predict_proba(X_val)[:, 1]
        _, f1 = find_optimal_threshold(y_val, val_probs)
        if f1 > best_lr_score:
            best_lr_score = f1
            best_lr_c = c_val
            best_lr_model = lr
            
    print(f"Best Logistic Regression: C={best_lr_c} | Validation F1={best_lr_score:.4f}")
    
    # Optimal threshold for LR on validation set
    lr_val_probs = best_lr_model.predict_proba(X_val)[:, 1]
    lr_opt_thresh, _ = find_optimal_threshold(y_val, lr_val_probs)
    lr_val_preds = (lr_val_probs >= lr_opt_thresh).astype(int)
    lr_val_metrics = evaluate_predictions(y_val, lr_val_preds, lr_val_probs)
    
    # STEP 5: Train & Tune Random Forest
    print("\n" + "=" * 60)
    print("STEP 5: Training & Tuning Random Forest Classifier")
    print("=" * 60)
    best_rf_score = -1.0
    best_rf_params = {}
    best_rf_model = None
    
    rf_param_grid = [
        {'n_estimators': 150, 'max_depth': 6, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 8, 'min_samples_split': 5},
        {'n_estimators': 250, 'max_depth': 10, 'min_samples_split': 4},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 5},
        {'n_estimators': 300, 'max_depth': 8, 'min_samples_split': 2}
    ]
    
    for params in rf_param_grid:
        rf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        val_probs = rf.predict_proba(X_val)[:, 1]
        _, f1 = find_optimal_threshold(y_val, val_probs)
        if f1 > best_rf_score:
            best_rf_score = f1
            best_rf_params = params
            best_rf_model = rf
            
    print(f"Best Random Forest: {best_rf_params} | Validation F1={best_rf_score:.4f}")
    
    # Optimal threshold for RF on validation set
    rf_val_probs = best_rf_model.predict_proba(X_val)[:, 1]
    rf_opt_thresh, _ = find_optimal_threshold(y_val, rf_val_probs)
    rf_val_preds = (rf_val_probs >= rf_opt_thresh).astype(int)
    rf_val_metrics = evaluate_predictions(y_val, rf_val_preds, rf_val_probs)
    
    # STEP 6: Model Selection based on Validation Metrics
    print("\n" + "=" * 60)
    print("STEP 6: Model Selection Analysis")
    print("=" * 60)
    print(f"Logistic Regression Validation -> F1: {lr_val_metrics['f1_score']}, ROC-AUC: {lr_val_metrics['roc_auc']}, PR-AUC: {lr_val_metrics['pr_auc']}")
    print(f"Random Forest Validation       -> F1: {rf_val_metrics['f1_score']}, ROC-AUC: {rf_val_metrics['roc_auc']}, PR-AUC: {rf_val_metrics['pr_auc']}")
    
    # Selection criteria: Higher validation F1-score, with ROC-AUC as tie-breaker
    if (rf_val_metrics['f1_score'] > lr_val_metrics['f1_score']) or (
        rf_val_metrics['f1_score'] == lr_val_metrics['f1_score'] and rf_val_metrics['roc_auc'] >= lr_val_metrics['roc_auc']
    ):
        selected_model_name = "Random Forest Classifier"
        selected_model = best_rf_model
        selected_threshold = rf_opt_thresh
        selection_reason = (
            f"Random Forest selected based on superior Validation F1-score ({rf_val_metrics['f1_score']} vs {lr_val_metrics['f1_score']}) "
            f"and ROC-AUC ({rf_val_metrics['roc_auc']} vs {lr_val_metrics['roc_auc']}), effectively capturing non-linear interactions."
        )
    else:
        selected_model_name = "Logistic Regression"
        selected_model = best_lr_model
        selected_threshold = lr_opt_thresh
        selection_reason = (
            f"Logistic Regression selected based on higher Validation F1-score ({lr_val_metrics['f1_score']} vs {rf_val_metrics['f1_score']}) "
            f"and better generalization balance under linear decision boundaries."
        )
        
    print(f"==> Selected: {selected_model_name}")
    print(f"Selection Reason: {selection_reason}")
    print(f"Optimized Decision Threshold: {selected_threshold}")
    
    # STEP 7: Final Test Evaluation on Untouched Test Set
    print("\n" + "=" * 60)
    print("STEP 7: Final Evaluation on Untouched Test Set")
    print("=" * 60)
    
    # LR Test Evaluation
    lr_test_probs = best_lr_model.predict_proba(X_test)[:, 1]
    lr_test_preds = (lr_test_probs >= lr_opt_thresh).astype(int)
    lr_test_metrics = evaluate_predictions(y_test, lr_test_preds, lr_test_probs)
    
    # RF Test Evaluation
    rf_test_probs = best_rf_model.predict_proba(X_test)[:, 1]
    rf_test_preds = (rf_test_probs >= rf_opt_thresh).astype(int)
    rf_test_metrics = evaluate_predictions(y_test, rf_test_preds, rf_test_probs)
    
    # Selected Model Test Metrics
    if selected_model_name == "Random Forest Classifier":
        test_metrics = rf_test_metrics
        test_probs = rf_test_probs
        test_preds = rf_test_preds
    else:
        test_metrics = lr_test_metrics
        test_probs = lr_test_probs
        test_preds = lr_test_preds
        
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"Confusion Matrix (TN, FP, FN, TP): {test_metrics['confusion_matrix']}")
    
    # STEP 8: Feature Importance Extraction
    print("\n" + "=" * 60)
    print("STEP 8: Feature Importance Extraction")
    print("=" * 60)
    if hasattr(best_rf_model, 'feature_importances_'):
        importances = best_rf_model.feature_importances_
    else:
        importances = np.abs(best_lr_model.coef_[0])
        
    # Map back to feature names
    feat_imp_list = []
    for name, imp in zip(feature_names, importances):
        feat_imp_list.append({'feature': name, 'importance': round(float(imp), 6)})
    feat_imp_list = sorted(feat_imp_list, key=lambda x: x['importance'], reverse=True)
    
    # Aggregate importance by source feature for high-level visualization
    aggregated_imp = {}
    for item in feat_imp_list:
        raw_name = item['feature']
        # Clean prefix from ColumnTransformer (e.g., 'num__', 'cat__')
        clean_name = raw_name.split('__')[-1].split('_')[0]
        aggregated_imp[clean_name] = aggregated_imp.get(clean_name, 0.0) + item['importance']
        
    agg_imp_list = [
        {'feature': k, 'importance': round(v, 4)}
        for k, v in sorted(aggregated_imp.items(), key=lambda x: x[1], reverse=True)
    ]
    
    print("Top 10 Important Features:")
    for f in feat_imp_list[:10]:
        print(f"  - {f['feature']}: {f['importance']}")
        
    # STEP 9: Compute ROC and PR Curves Data
    lr_curve_data = compute_curve_data(y_test, lr_test_probs)
    rf_curve_data = compute_curve_data(y_test, rf_test_probs)
    
    # STEP 10: Persist Artifacts
    print("\n" + "=" * 60)
    print("STEP 9: Saving Model Artifacts & Metadata")
    print("=" * 60)
    
    # Deployable Scikit-Learn inference package
    deployable_artifact = {
        'model_name': selected_model_name,
        'model': selected_model,
        'preprocessor': preprocessor,
        'threshold': selected_threshold,
        'feature_names': feature_names,
        'training_date': datetime.now().isoformat()
    }
    
    joblib.dump(deployable_artifact, os.path.join(output_dir, 'best_model.joblib'))
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
    joblib.dump(best_lr_model, os.path.join(output_dir, 'lr_model.joblib'))
    joblib.dump(best_rf_model, os.path.join(output_dir, 'rf_model.joblib'))
    
    # Metadata JSON
    metadata = {
        'project_name': 'E-Commerce Return Intelligence',
        'training_timestamp': datetime.now().isoformat(),
        'selected_model': selected_model_name,
        'optimal_threshold': selected_threshold,
        'selection_reason': selection_reason,
        'dataset': {
            'total_clean_samples': len(y),
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'positive_rate': round(float(np.mean(y)), 4)
        },
        'hyperparameters': {
            'Logistic Regression': {'C': best_lr_c, 'class_weight': 'balanced'},
            'Random Forest': best_rf_params
        }
    }
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    # Evaluation Results JSON
    eval_results = {
        'selected_model': selected_model_name,
        'threshold': selected_threshold,
        'validation_metrics': {
            'Logistic Regression': lr_val_metrics,
            'Random Forest': rf_val_metrics
        },
        'test_metrics': {
            'Logistic Regression': lr_test_metrics,
            'Random Forest': rf_test_metrics
        },
        'curves': {
            'Logistic Regression': lr_curve_data,
            'Random Forest': rf_curve_data
        }
    }
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
        
    # Feature Importance JSON
    with open(os.path.join(output_dir, 'feature_importance.json'), 'w') as f:
        json.dump({
            'detailed': feat_imp_list[:30],
            'aggregated': agg_imp_list
        }, f, indent=2)
        
    # Save Test Set Predictions for demonstration and bulk validation
    test_pred_df = X_test_raw.copy()
    test_pred_df['Actual_Returned'] = y_test
    test_pred_df['Predicted_Probability'] = np.round(test_probs, 4)
    test_pred_df['Predicted_Returned'] = test_preds
    test_pred_df.to_csv(os.path.join(predictions_dir, 'test_predictions.csv'), index=False)
    
    print("All artifacts successfully saved to 'models/' and 'output/'!")
    return eval_results


if __name__ == '__main__':
    train_and_evaluate()
