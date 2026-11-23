"""Compare adaptive fallback against a fixed Enhanced RF with train-only imputation.

The same three-class KDDTest+ rows and fitted RF models are used throughout.
Only the six additional features are hidden, on a reproducible random subset
of rows; all four Basic fields remain valid. This is a local stress test,
not reproduction of any published paper or a claim about real traffic.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score


def run(detector, columns, map_label, additional, output_dir):
    def read(name):
        df = pd.read_csv(Path(detector.data_dir) / name, header=None, names=columns)
        df['category'] = df['label'].map(map_label)
        return df[df['category'].isin(detector.target_classes)].reset_index(drop=True)

    train, test = read('KDDTrain+.txt'), read('KDDTest+.txt')
    y = test['category'].to_numpy()

    def predict(df, enhanced):
        if enhanced:
            x = detector.prepare_enhanced_features(df)
            cols = detector.enhanced_numerical_cols
            x[cols] = detector.enhanced_scaler.transform(x[cols])
            return detector.enhanced_rf.predict(x)
        x = detector.prepare_features(df)
        cols = detector.numerical_cols
        x[cols] = detector.scaler.transform(x[cols])
        return detector.active_model.predict(x)

    # Fill with statistics learned only from the training partition; do not
    # alter any labels, tune on test data, or invent production predictions.
    imputed = test.copy()
    for col in additional:
        fill = train[col].mode().iloc[0] if col == 'flag' else train[col].median()
        imputed[col] = fill
    pred_basic, pred_enhanced = predict(test, False), predict(test, True)
    pred_imputed = predict(imputed, True)
    random_values = np.random.default_rng(42).random(len(test))
    rows = []
    for rate in (0, .25, .5, .75, 1):
        mask = random_values < rate
        for name, missing_pred in [('Adaptive SentinelNet', pred_basic), ('Fixed Enhanced RF (imputed)', pred_imputed)]:
            pred = np.where(mask, missing_pred, pred_enhanced)
            rows.append({'Missing additional fields (%)': round(mask.mean()*100, 2),
                         'Model': name, 'Accuracy': round(accuracy_score(y, pred), 6),
                         'Macro F1': round(f1_score(y, pred, average='macro', zero_division=0), 6),
                         'Probe Recall': round(recall_score(y, pred, labels=['Probe'], average='macro', zero_division=0), 6)})
    output = Path(output_dir) / 'missing_features_comparison.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    return result


if __name__ == '__main__':
    from app import detector, NSL_KDD_COLUMNS, map_nsl_label, ENHANCED_ADDITIONAL_COLS
    run(detector, NSL_KDD_COLUMNS, map_nsl_label, ENHANCED_ADDITIONAL_COLS,
        Path(detector.base_dir) / 'data' / 'benchmark_results')
