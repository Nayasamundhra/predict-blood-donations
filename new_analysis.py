import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from operator import itemgetter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    try:
        # Load dataset
        print("Loading dataset...")
        transfusion = pd.read_csv(r"C:/Users/user/Downloads/Data Analyst/Give Life_ Predict Blood Donations/datasets/transfusion.data")
        
        # Display basic info about the dataset
        print(f"Dataset shape: {transfusion.shape}")
        print(f"Columns: {list(transfusion.columns)}")
        
        # Rename target column
        transfusion.rename(
            columns={'whether he/she donated blood in March 2007': 'target'},
            inplace=True
        )
        
        print(f"Target distribution:\n{transfusion['target'].value_counts(normalize=True).round(3)}")
        
        # Split the DataFrame
        X_train, X_test, y_train, y_test = train_test_split(
            transfusion.drop(columns='target'),  # features (X)
            transfusion['target'],               # target (y)
            test_size=0.25,                      # 25% test, 75% train
            random_state=42,                     # for reproducibility
            stratify=transfusion['target']       # preserve target distribution
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # TASK 8: Check X_train's variance
        print("\n" + "="*50)
        print("TASK 8: CHECKING X_TRAIN VARIANCE")
        print("="*50)
        
        # Calculate variance for each feature in X_train, rounded to 3 decimal places
        variance_results = X_train.var().round(3)
        
        print("X_train's variance, rounding the output to 3 decimal places:")
        for feature, variance in variance_results.items():
            print(f"{feature}: {variance}")
        
        # Additional analysis: Check if normalization is needed
        print(f"\nVariance summary:")
        print(f"Min variance: {variance_results.min():.3f}")
        print(f"Max variance: {variance_results.max():.3f}")
        print(f"Variance ratio (max/min): {(variance_results.max()/variance_results.min()):.3f}")
        
        if variance_results.max() / variance_results.min() > 10:
            print("⚠️  High variance difference detected! Normalization recommended for linear models.")
        else:
            print("✅ Variance differences are acceptable.")
        
        # TASK 9: Log normalization
        print("\n" + "="*50)
        print("TASK 9: LOG NORMALIZATION")
        print("="*50)
        
        # Import numpy
        import numpy as np
        
        # Copy X_train and X_test into X_train_normed and X_test_normed
        X_train_normed = X_train.copy()
        X_test_normed = X_test.copy()
        
        # Specify which column to normalize
        col_to_normalize = 'Monetary (c.c. blood)'
        
        # Log normalization
        for df_ in [X_train_normed, X_test_normed]:
            # Add log normalized column
            df_['monetary_log'] = np.log(df_[col_to_normalize])
            # Drop the original column
            df_.drop(columns=col_to_normalize, inplace=True)
        
        # Check the variance for X_train_normed
        print("Checking the variance for X_train_normed:")
        variance_normed = X_train_normed.var().round(3)
        for feature, variance in variance_normed.items():
            print(f"{feature}: {variance}")
        
        print(f"\nVariance comparison:")
        print(f"Original 'Monetary (c.c. blood)' variance: {variance_results['Monetary (c.c. blood)']}")
        print(f"Log-normalized 'monetary_log' variance: {variance_normed['monetary_log']}")
        print(f"Variance reduction: {((variance_results['Monetary (c.c. blood)'] - variance_normed['monetary_log']) / variance_results['Monetary (c.c. blood)'] * 100):.2f}%")
        
        # TASK 10: Training the linear regression model
        print("\n" + "="*50)
        print("TASK 10: TRAINING THE LINEAR REGRESSION MODEL")
        print("="*50)
        
        # Importing modules (LogisticRegression is already imported at the top)
        # from sklearn.linear_model import LogisticRegression
        
        # Instantiate LogisticRegression
        logreg = LogisticRegression(
            solver='liblinear',
            random_state=42
        )
        
        # Train the model
        logreg.fit(X_train_normed, y_train)
        
        # AUC score for the trained model
        logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
        print(f'\nAUC score: {logreg_auc_score:.4f}')
        
        # Additional model evaluation
        y_pred_logreg = logreg.predict(X_test_normed)
        print(f"\nLogistic Regression Model Performance:")
        print(f"AUC Score: {logreg_auc_score:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_logreg))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_logreg))
        
        # Try TPOT first, but with error handling
        try:
            from tpot import TPOTClassifier
            print("\nAttempting TPOT analysis...")
            
            # Use a more conservative configuration
            tpot = TPOTClassifier(
                generations=3,              # Reduced generations
                population_size=10,         # Reduced population
                verbosity=1,               # Reduced verbosity
                scoring='roc_auc',
                random_state=42,
                disable_update_check=True,
                config_dict='TPOT light',   # Use light config
                max_time_mins=5,           # Add time limit
                n_jobs=1                   # Single thread to avoid issues
            )
            
            # Fit the model using log-normalized data
            tpot.fit(X_train_normed, y_train)
            
            # Calculate AUC score using log-normalized data
            tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test_normed)[:, 1])
            print(f'\nTPOT AUC score: {tpot_auc_score:.4f}')
            
            # Print best pipeline steps
            print('\nBest pipeline steps:')
            for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
                print(f'{idx}. {name}: {transform}')
                
            # Export the best pipeline
            tpot.export('tpot_blood_donation_pipeline.py')
            print("\nBest pipeline exported to 'tpot_blood_donation_pipeline.py'")
            
        except Exception as tpot_error:
            print(f"\nTPOT failed with error: {str(tpot_error)}")
            print("Falling back to manual model comparison...")
            
            # Manual model comparison as fallback using log-normalized data
            print("\nComparing models with original vs log-normalized data...")
            
            # Test with both original and log-normalized data
            datasets = {
                'Original Data': (X_train, X_test),
                'Log-Normalized Data': (X_train_normed, X_test_normed)
            }
            
            all_results = {}
            
            for data_type, (X_tr, X_te) in datasets.items():
                print(f"\n--- {data_type} ---")
                
                models = {
                    'Logistic Regression': Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
                    ]),
                    'Random Forest': RandomForestClassifier(
                        n_estimators=100, 
                        random_state=42,
                        class_weight='balanced'
                    )
                }
                
                for name, model in models.items():
                    print(f"\nTraining {name} with {data_type}...")
                    
                    # Fit the model
                    model.fit(X_tr, y_train)
                    
                    # Make predictions
                    y_pred_proba = model.predict_proba(X_te)[:, 1]
                    y_pred = model.predict(X_te)
                    
                    # Calculate metrics
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    model_key = f"{name} ({data_type})"
                    all_results[model_key] = auc_score
                    
                    print(f"{name} AUC Score: {auc_score:.4f}")
                    print(f"\nClassification Report for {name}:")
                    print(classification_report(y_test, y_pred))
                    print(f"\nConfusion Matrix for {name}:")
                    print(confusion_matrix(y_test, y_pred))
            
            # Find best performing model overall
            best_model = max(all_results, key=all_results.get)
            print(f"\nBest performing model overall: {best_model} with AUC score: {all_results[best_model]:.4f}")
            
            # TASK 11: Sort models by AUC score from highest to lowest
            print("\n" + "="*50)
            print("TASK 11: SORTING MODELS BY AUC SCORE")
            print("="*50)
            
            # Create list of (model_name, model_score) pairs
            model_score_pairs = [(model_name, score) for model_name, score in all_results.items()]
            
            # Sort the list from highest to lowest using reverse=True parameter
            sorted_models = sorted(model_score_pairs, key=itemgetter(1), reverse=True)
            
            print("Models sorted by AUC score (highest to lowest):")
            for rank, (model_name, score) in enumerate(sorted_models, 1):
                print(f"{rank}. {model_name}: {score:.4f}")
            
            # Show comparison summary
            print(f"\n--- MODEL COMPARISON SUMMARY ---")
            for model_name, score in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
                print(f"{model_name}: {score:.4f}")
        
        # Feature importance analysis
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Use Random Forest for feature importance
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance (Random Forest):")
        for idx, row in feature_importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        print(f"\nRandom Forest AUC Score: {roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]):.4f}")
        
    except FileNotFoundError:
        print("Error: Could not find the dataset file.")
        print("Please check the file path: C:/Users/user/Downloads/Data Analyst/Give Life_ Predict Blood Donations/datasets/transfusion.data")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("Please check your data and try again.")

if __name__ == "__main__":
    main()