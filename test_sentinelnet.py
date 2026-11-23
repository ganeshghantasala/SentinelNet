"""
Comprehensive Automated Test Suite for SentinelNet Intrusion Detection System
Validates NSL-KDD dataset loading, attack mapping, 3-class scope, preprocessing,
model training, independent test evaluation, confusion matrix ordering, cache verification,
XSS hardening, probability terminology, explainable predictions, and CSV upload validation.
"""

import os
import io
import json
import unittest
import numpy as np
import pandas as pd
import sklearn

import app

class TestSentinelNetNSLKDD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = app.detector
        cls.client = app.app.test_client()

    def test_01_training_data_loading(self):
        """Test 1: Successful NSL-KDD training-data loading."""
        train_stats = self.detector.dataset_stats.get('train', {})
        self.assertEqual(train_stats.get('total'), 125973)
        self.assertTrue(os.path.exists(os.path.join(self.detector.data_dir, "KDDTrain+.txt")))

    def test_02_test_data_loading(self):
        """Test 2: Successful NSL-KDD test-data loading."""
        test_stats = self.detector.dataset_stats.get('test', {})
        self.assertEqual(test_stats.get('total'), 22544)
        self.assertTrue(os.path.exists(os.path.join(self.detector.data_dir, "KDDTest+.txt")))

    def test_03_correct_mapping_normal_dos_probe(self):
        """Test 3: Correct mapping of Normal, DoS and Probe."""
        self.assertEqual(app.map_nsl_label('normal'), 'Normal')
        self.assertEqual(app.map_nsl_label('neptune'), 'DoS')
        self.assertEqual(app.map_nsl_label('smurf'), 'DoS')
        self.assertEqual(app.map_nsl_label('back'), 'DoS')
        self.assertEqual(app.map_nsl_label('satan'), 'Probe')
        self.assertEqual(app.map_nsl_label('ipsweep'), 'Probe')
        self.assertEqual(app.map_nsl_label('nmap'), 'Probe')

    def test_04_correct_exclusion_r2l_u2r(self):
        """Test 4: Correct exclusion of R2L and U2R (never mapped to Normal)."""
        self.assertEqual(app.map_nsl_label('guess_passwd'), 'R2L')
        self.assertEqual(app.map_nsl_label('warezclient'), 'R2L')
        self.assertEqual(app.map_nsl_label('buffer_overflow'), 'U2R')
        self.assertEqual(app.map_nsl_label('rootkit'), 'U2R')
        self.assertNotEqual(app.map_nsl_label('buffer_overflow'), 'Normal')
        self.assertNotEqual(app.map_nsl_label('guess_passwd'), 'Normal')

        train_stats = self.detector.dataset_stats.get('train', {})
        self.assertEqual(train_stats.get('excluded'), 1047) # 995 R2L + 52 U2R
        self.assertEqual(train_stats.get('included'), 124926)

        test_stats = self.detector.dataset_stats.get('test', {})
        self.assertEqual(test_stats.get('excluded'), 2952) # 2885 R2L + 67 U2R
        self.assertEqual(test_stats.get('included'), 19592)

    def test_05_correct_feature_preprocessing(self):
        """Test 5: Correct feature preprocessing and scaler properties."""
        self.assertEqual(len(self.detector.feature_columns), 6)
        self.assertIn('proto_TCP', self.detector.feature_columns)
        self.assertIn('proto_UDP', self.detector.feature_columns)
        self.assertIn('proto_ICMP', self.detector.feature_columns)
        self.assertTrue(hasattr(self.detector.scaler, 'mean_'))
        self.assertEqual(len(self.detector.scaler.mean_), 3)

    def test_06_three_model_training_processes(self):
        """Test 6: All three model training processes in validation benchmark."""
        bench = self.detector.val_benchmark_results
        self.assertEqual(len(bench), 3)
        model_names = [m['Model'] for m in bench]
        self.assertIn('Logistic Regression', model_names)
        self.assertIn('K-Nearest Neighbors', model_names)
        self.assertIn('Random Forest', model_names)
        for m in bench:
            self.assertGreater(m['Accuracy'], 0.50)
            self.assertIn('F1-Score', m)
            self.assertIn('per_class', m)

    def test_07_independent_kddtest_evaluation(self):
        """Test 7: Independent KDDTest+ evaluation."""
        test_eval = self.detector.test_evaluation_results
        self.assertIn('Accuracy', test_eval)
        self.assertIn('F1-Score', test_eval)
        self.assertIn('confusion_matrix', test_eval)
        self.assertIn('feature_importances', test_eval)
        self.assertGreater(test_eval['Accuracy'], 0.75)
        self.assertEqual(len(test_eval['confusion_matrix']), 3)

    def test_08_valid_csv_upload(self):
        """Test 8: Valid CSV upload."""
        csv_content = """duration,protocol_type,src_bytes,dst_bytes,Timestamp
0.0,tcp,215,45096,2026-11-17 10:00:00
0.0,udp,105,146,2026-11-17 10:01:00
0.5,icmp,0,0,2026-11-17 10:02:00
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'test_valid.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertTrue(json_data['success'])
        self.assertEqual(json_data['total_records'], 3)
        self.assertEqual(json_data['valid_records'], 3)
        self.assertEqual(json_data['rejected_records'], 0)

    def test_09_missing_column_csv_rejection(self):
        """Test 9: Missing-column CSV rejection."""
        csv_content = """duration,src_bytes,dst_bytes
0.0,215,45096
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'missing_proto.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertFalse(json_data['success'])
        self.assertIn('Missing required CSV columns', json_data['error_message'])

    def test_10_invalid_numeric_values_rejection(self):
        """Test 10: Invalid numeric values (non-numeric, NaN, infinity)."""
        csv_content = """duration,protocol_type,src_bytes,dst_bytes
invalid_num,tcp,215,45096
0.0,udp,nan,146
inf,icmp,0,0
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'invalid_nums.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertFalse(json_data['success'])
        self.assertEqual(json_data['valid_records'], 0)
        self.assertEqual(json_data['rejected_records'], 3)

    def test_11_unsupported_protocol_rejection(self):
        """Test 11: Unsupported protocol rejection (never convert to TCP)."""
        csv_content = """duration,protocol_type,src_bytes,dst_bytes
0.0,sctp,215,45096
0.0,gre,100,200
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'unsupported_proto.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertFalse(json_data['success'])
        self.assertEqual(json_data['valid_records'], 0)

    def test_12_negative_value_rejection(self):
        """Test 12: Negative value rejection."""
        csv_content = """duration,protocol_type,src_bytes,dst_bytes
-1.5,tcp,215,45096
0.0,udp,-100,146
0.0,icmp,10,-5
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'negative_vals.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertFalse(json_data['success'])
        self.assertEqual(json_data['rejected_records'], 3)

    def test_13_batch_predictions(self):
        """Test 13: Batch predictions produce expected classification fields."""
        csv_content = """duration,protocol_type,src_bytes,dst_bytes
0.0,tcp,215,45096
0.0,tcp,0,0
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'batch_test.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        preview = json_data['preview_records']
        self.assertEqual(len(preview), 2)
        for r in preview:
            self.assertIn(r['predicted_classification'], ['Normal', 'DoS', 'Probe'])
            self.assertIn('confidence_score', r)
            self.assertIn('risk_level', r)

    def test_14_probability_terminology_and_outputs(self):
        """Test 14: Probability calculations across all three classes and proper terminology."""
        flow = {
            "src_bytes": 1500,
            "duration": 0.5,
            "protocol_type": "TCP",
            "dst_bytes": 4500
        }
        res = self.client.post('/api/scan-packet', json=flow)
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        r = json_data['result']

        # Verify model probability estimates
        probs = r['model_probability_estimates']
        self.assertIn('Normal', probs)
        self.assertIn('DoS', probs)
        self.assertIn('Probe', probs)
        total_p = sum(probs.values())
        self.assertAlmostEqual(total_p, 1.0, places=2)
        self.assertEqual(r['confidence'], max(probs.values()))

        # Verify risk mapping note explains deterministic mapping
        self.assertIn('risk_mapping_note', r)
        self.assertIn('categorical severity mapping', r['risk_mapping_note'].lower())

    def test_15_annotated_csv_download(self):
        """Test 15: Annotated CSV generation and structure."""
        csv_content = """duration,protocol_type,src_bytes,dst_bytes,src_ip
0.0,tcp,215,45096,192.168.1.50
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'dl_test.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        csv_out = res.get_json()['csv_data']
        df_out = pd.read_csv(io.StringIO(csv_out))
        self.assertIn('src_ip', df_out.columns)
        self.assertIn('predicted_classification', df_out.columns)
        self.assertIn('confidence_score', df_out.columns)
        self.assertIn('probability_normal', df_out.columns)
        self.assertIn('probability_dos', df_out.columns)
        self.assertIn('probability_probe', df_out.columns)
        self.assertIn('risk_level', df_out.columns)
        self.assertIn('explanation', df_out.columns)

    def test_16_scan_history(self):
        """Test 16: Scan history maintenance and retrieval."""
        res = self.client.get('/api/scan-history')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertTrue(json_data['success'])
        self.assertIsInstance(json_data['scan_history'], list)

    def test_17_dashboard_rendering(self):
        """Test 17: Dashboard endpoints rendering."""
        res_index = self.client.get('/')
        self.assertEqual(res_index.status_code, 200)
        self.assertIn(b'SentinelNet', res_index.data)
        self.assertIn(b'NSL-KDD', res_index.data)

        res_api = self.client.get('/api/dashboard')
        self.assertEqual(res_api.status_code, 200)
        data = res_api.get_json()
        self.assertIn('kpis', data)
        self.assertIn('charts', data)
        self.assertIn('classification_dist', data['charts'])
        self.assertIn('model_comparison', data['charts'])
        self.assertIn('feature_importance', data['charts'])
        self.assertIn('confusion_matrix', data['charts'])
        self.assertIn('per_class', data['charts'])

    def test_18_confusion_matrix_ordering(self):
        """Test 18: Consistent Normal, DoS, Probe ordering in confusion matrix and per-class metrics."""
        self.assertEqual(self.detector.target_classes, ['Normal', 'DoS', 'Probe'])
        eval_res = self.detector.test_evaluation_results

        # Labels order explicitly specified
        labels = eval_res.get('confusion_matrix_labels', [])
        self.assertEqual(labels, ['Normal', 'DoS', 'Probe'])

        cm = np.array(eval_res['confusion_matrix'])
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(cm.sum(), 19592)

        # Diagonal values corresponding strictly to [Normal, DoS, Probe]
        self.assertEqual(cm[0, 0], 9333) # True Normal -> Pred Normal
        self.assertEqual(cm[1, 1], 5981) # True DoS -> Pred DoS
        self.assertEqual(cm[2, 2], 728)  # True Probe -> Pred Probe

        # Per-class metrics dictionary keys must follow canonical order
        per_class_keys = list(eval_res['per_class'].keys())
        self.assertEqual(per_class_keys, ['Normal', 'DoS', 'Probe'])

    def test_19_safe_rendering_of_malicious_csv_fields(self):
        """Test 19: Safe handling and sanitization of malicious HTML-like CSV fields (XSS prevention)."""
        malicious_csv = """duration,protocol_type,src_bytes,dst_bytes,Timestamp,src_ip
0.0,tcp,215,45096,<script>alert('xss_time')</script>,<img src=x onerror=alert('xss_ip')>
"""
        malicious_filename = '<script>alert("filename_xss")</script>.csv'
        data = {'file': (io.BytesIO(malicious_csv.encode('utf-8')), malicious_filename)}

        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()

        # Filename must be sanitized (no angle brackets or script tags)
        self.assertNotIn('<script>', json_data['filename'])
        self.assertNotIn('</script>', json_data['filename'])
        self.assertTrue(json_data['filename'].endswith('.csv'))

        # Preview record contains literal string, safe for textContent insertion
        preview = json_data['preview_records'][0]
        self.assertIn('xss_time', preview['Timestamp'])
        self.assertIn('xss_ip', preview['src_ip'])

    def test_20_cache_version_mismatch(self):
        """Test 20: Rejection of model cache when scikit-learn version mismatches."""
        train_path = os.path.join(self.detector.data_dir, "KDDTrain+.txt")
        test_path = os.path.join(self.detector.data_dir, "KDDTest+.txt")

        # Mock cache with outdated sklearn version
        mock_cache = {
            'metadata': {
                'sklearn_version': '0.24.0', # Incompatible version
                'feature_schema': self.detector.feature_columns,
                'target_classes': self.detector.target_classes,
                'train_dataset_sha256': app.get_dataset_fingerprint(train_path),
                'test_dataset_sha256': app.get_dataset_fingerprint(test_path)
            },
            'scaler': self.detector.scaler,
            'models': self.detector.models,
            'dataset_stats': self.detector.dataset_stats,
            'val_benchmark_results': self.detector.val_benchmark_results,
            'test_evaluation_results': self.detector.test_evaluation_results,
            'charts_b64': self.detector.charts_b64,
            'sample_flows': self.detector.sample_flows
        }

        is_valid, reason = self.detector.validate_cache(mock_cache, train_path, test_path)
        self.assertFalse(is_valid)
        self.assertIn('Scikit-learn version mismatch', reason)

    def test_21_dataset_fingerprint_mismatch(self):
        """Test 21: Rejection of model cache when dataset SHA-256 fingerprint mismatches."""
        train_path = os.path.join(self.detector.data_dir, "KDDTrain+.txt")
        test_path = os.path.join(self.detector.data_dir, "KDDTest+.txt")

        # Mock cache with altered dataset SHA-256
        mock_cache = {
            'metadata': {
                'sklearn_version': sklearn.__version__,
                'feature_schema': self.detector.feature_columns,
                'target_classes': self.detector.target_classes,
                'train_dataset_sha256': 'tampered_sha256_hash_value',
                'test_dataset_sha256': app.get_dataset_fingerprint(test_path)
            },
            'scaler': self.detector.scaler,
            'models': self.detector.models,
            'dataset_stats': self.detector.dataset_stats,
            'val_benchmark_results': self.detector.val_benchmark_results,
            'test_evaluation_results': self.detector.test_evaluation_results,
            'charts_b64': self.detector.charts_b64,
            'sample_flows': self.detector.sample_flows
        }

        is_valid, reason = self.detector.validate_cache(mock_cache, train_path, test_path)
        self.assertFalse(is_valid)
        self.assertIn('KDDTrain+.txt SHA-256 fingerprint mismatch', reason)

    def test_22_batch_prediction_explanations(self):
        """Test 22: Plain-language explanations and low-confidence warnings in batch predictions."""
        # 1 DoS (large payload), 1 Probe (small ICMP probe), 1 ambiguous flow
        csv_content = """duration,protocol_type,src_bytes,dst_bytes
1.5,tcp,45000,100
0.1,icmp,40,0
0.2,tcp,800,500
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'test_exp.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()

        records = json_data['preview_records']
        self.assertEqual(len(records), 3)

        for r in records:
            # Each record must have a cautious explanation
            self.assertIn('explanation', r)
            self.assertTrue(len(r['explanation']) > 20)
            self.assertIn('low_confidence_warning', r)
            self.assertIsInstance(r['low_confidence_warning'], bool)
            self.assertIn('risk_level', r)

    def test_23_force_fresh_model_training_pipeline(self):
        """Test 23: Force a fresh model-training pipeline without relying on cache."""
        # Run force retrain
        self.detector.load_and_train_datasets(force_retrain=True)
        self.assertTrue(self.detector.is_trained)
        self.assertEqual(len(self.detector.val_benchmark_results), 3)
        self.assertIn('Accuracy', self.detector.test_evaluation_results)

        # Verify cache metadata was created and populated
        self.assertIn('sklearn_version', self.detector.cache_metadata)
        self.assertIn('train_dataset_sha256', self.detector.cache_metadata)
        self.assertEqual(self.detector.cache_metadata['train_total'], 125973)

    def test_24_enhanced_mode_kddtest_evaluation(self):
        """Test 24: Enhanced Mode KDDTest+ evaluation reproduces accuracy and probe recall improvement."""
        enh_eval = self.detector.enhanced_test_evaluation
        self.assertIn('Accuracy', enh_eval)
        self.assertIn('F1-Score', enh_eval)
        self.assertIn('macro_f1', enh_eval)
        self.assertIn('per_class', enh_eval)
        self.assertIn('confusion_matrix', enh_eval)

        # Accuracy >= 85.0% and Weighted F1 >= 84.0%
        self.assertGreaterEqual(enh_eval['Accuracy'], 0.84)
        self.assertGreaterEqual(enh_eval['F1-Score'], 0.84)

        # Probe recall significantly improved over basic mode (30.07% -> >60.0%)
        probe_rec = enh_eval['per_class']['Probe']['recall']
        self.assertGreater(probe_rec, 0.55)
        self.assertAlmostEqual(probe_rec, 0.6080, places=2)

        # Confusion matrix labels and structure
        self.assertEqual(enh_eval['confusion_matrix_labels'], ['Normal', 'DoS', 'Probe'])
        cm_e = np.array(enh_eval['confusion_matrix'])
        self.assertEqual(cm_e.shape, (3, 3))
        self.assertEqual(cm_e.sum(), 19592)

    def test_25_uncertainty_decision_status_and_threshold(self):
        """Test 25: Confidence thresholding and decision status (ACCEPTED vs REVIEW REQUIRED)."""
        # High confidence flow -> ACCEPTED
        high_conf_flow = {
            "src_bytes": 0,
            "duration": 0.0,
            "dst_bytes": 0,
            "protocol_type": "TCP",
            "count": 250,
            "srv_count": 250,
            "flag": "S0",
            "same_srv_rate": 1.0,
            "serror_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0
        }
        res_high = self.client.post('/api/scan-packet', json=high_conf_flow)
        self.assertEqual(res_high.status_code, 200)
        data_high = res_high.get_json()['result']
        self.assertEqual(data_high['decision_status'], 'ACCEPTED')
        self.assertFalse(data_high['low_confidence_warning'])
        self.assertGreaterEqual(data_high['confidence'], 0.65)

        # Review required metrics in test evaluation
        enh_eval = self.detector.enhanced_test_evaluation
        self.assertIn('coverage', enh_eval)
        self.assertIn('accuracy_accepted', enh_eval)
        self.assertIn('review_count', enh_eval)
        self.assertGreater(enh_eval['coverage'], 0.90)
        self.assertGreater(enh_eval['accuracy_accepted'], enh_eval['Accuracy'])

    def test_26_automatic_model_selection_enhanced_csv(self):
        """Test 26: Automatic selection of Enhanced Mode when all 10 features are valid."""
        csv_content = """duration,protocol_type,src_bytes,dst_bytes,count,srv_count,flag,same_srv_rate,serror_rate,dst_host_diff_srv_rate
0.0,tcp,215,45096,1,1,SF,1.0,0.0,0.0
0.0,tcp,0,0,250,250,S0,1.0,1.0,0.0
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'test_enhanced_upload.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertTrue(json_data['success'])
        self.assertEqual(json_data['enhanced_count'], 2)
        self.assertEqual(json_data['basic_count'], 0)
        self.assertEqual(json_data['fallback_count'], 0)

        preview = json_data['preview_records']
        self.assertEqual(preview[0]['detection_mode'], 'Enhanced')
        self.assertEqual(preview[0]['fallback_reason'], '')
        self.assertEqual(preview[1]['detection_mode'], 'Enhanced')

    def test_27_automatic_model_selection_basic_csv(self):
        """Test 27: Automatic selection of Basic Mode when only 4 features are present."""
        csv_content = """duration,protocol_type,src_bytes,dst_bytes
0.0,tcp,215,45096
0.0,udp,105,146
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'test_basic_upload.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertTrue(json_data['success'])
        self.assertEqual(json_data['enhanced_count'], 0)
        self.assertEqual(json_data['basic_count'], 2)
        self.assertEqual(json_data['fallback_count'], 0)

        preview = json_data['preview_records']
        self.assertEqual(preview[0]['detection_mode'], 'Basic')
        self.assertEqual(preview[0]['fallback_reason'], '')

    def test_28_automatic_model_selection_partial_enhanced_fallback(self):
        """Test 28: Fallback to Basic Mode when enhanced columns are partially present."""
        # Has count and flag, but missing same_srv_rate, serror_rate, etc.
        csv_content = """duration,protocol_type,src_bytes,dst_bytes,count,flag
0.0,tcp,215,45096,5,SF
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'partial_enh.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertTrue(json_data['success'])
        self.assertEqual(json_data['basic_count'], 1)
        self.assertEqual(json_data['enhanced_count'], 0)
        self.assertEqual(json_data['fallback_count'], 1)

        preview = json_data['preview_records']
        self.assertEqual(preview[0]['detection_mode'], 'Basic')
        self.assertIn('Partial enhanced columns in CSV header', preview[0]['fallback_reason'])

    def test_29_row_level_invalid_enhanced_fallback(self):
        """Test 29: Row-level fallback to Basic Mode when an individual row has invalid enhanced features."""
        # Row 1 is fully valid enhanced
        # Row 2 has invalid flag ('INVALID_FLAG')
        # Row 3 has negative count (-5)
        csv_content = """duration,protocol_type,src_bytes,dst_bytes,count,srv_count,flag,same_srv_rate,serror_rate,dst_host_diff_srv_rate
0.0,tcp,215,45096,1,1,SF,1.0,0.0,0.0
0.0,tcp,500,200,5,5,INVALID_FLAG,1.0,0.0,0.0
0.0,udp,100,20,-5,1,SF,1.0,0.0,0.0
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'row_fallback.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertTrue(json_data['success'])
        self.assertEqual(json_data['total_records'], 3)
        self.assertEqual(json_data['valid_records'], 3)
        self.assertEqual(json_data['enhanced_count'], 1)
        self.assertEqual(json_data['basic_count'], 2)
        self.assertEqual(json_data['fallback_count'], 2)

        preview = json_data['preview_records']
        # Row 1: Enhanced
        self.assertEqual(preview[0]['detection_mode'], 'Enhanced')
        self.assertEqual(preview[0]['fallback_reason'], '')

        # Row 2: Fallback to Basic due to unsupported flag
        self.assertEqual(preview[1]['detection_mode'], 'Basic')
        self.assertIn("Unsupported flag 'INVALID_FLAG'", preview[1]['fallback_reason'])

        # Row 3: Fallback to Basic due to negative count
        self.assertEqual(preview[2]['detection_mode'], 'Basic')
        self.assertIn('Invalid count (-5.0)', preview[2]['fallback_reason'])

    def test_30_row_level_invalid_basic_rejection(self):
        """Test 30: Rejection of row when basic features are invalid, regardless of enhanced features."""
        csv_content = """duration,protocol_type,src_bytes,dst_bytes,count,srv_count,flag,same_srv_rate,serror_rate,dst_host_diff_srv_rate
0.0,tcp,-50,45096,1,1,SF,1.0,0.0,0.0
-1.0,tcp,100,100,1,1,SF,1.0,0.0,0.0
0.0,UNKNOWN_PROTO,100,100,1,1,SF,1.0,0.0,0.0
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'bad_basic.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertFalse(json_data['success'])
        self.assertEqual(json_data['valid_records'], 0)
        self.assertEqual(json_data['rejected_records'], 3)

    def test_31_enhanced_feature_validation_rules(self):
        """Test 31: Enhanced feature validation: integer check on count, bounds on rate, flag vocabulary."""
        # Non-integer count float (2.5)
        flow_float_count = {
            "src_bytes": 100, "duration": 0.0, "dst_bytes": 0, "protocol_type": "TCP",
            "count": 2.5, "srv_count": 2, "flag": "SF", "same_srv_rate": 1.0, "serror_rate": 0.0, "dst_host_diff_srv_rate": 0.0
        }
        res = self.client.post('/api/scan-packet', json=flow_float_count)
        self.assertEqual(res.status_code, 200)
        r = res.get_json()['result']
        self.assertEqual(r['detection_mode'], 'Basic') # Falls back to Basic
        self.assertIn('must be a non-negative integer', r['fallback_reason'])

        # Out-of-bounds rate (1.5)
        flow_bad_rate = {
            "src_bytes": 100, "duration": 0.0, "dst_bytes": 0, "protocol_type": "TCP",
            "count": 2, "srv_count": 2, "flag": "SF", "same_srv_rate": 1.5, "serror_rate": 0.0, "dst_host_diff_srv_rate": 0.0
        }
        res2 = self.client.post('/api/scan-packet', json=flow_bad_rate)
        self.assertEqual(res2.status_code, 200)
        r2 = res2.get_json()['result']
        self.assertEqual(r2['detection_mode'], 'Basic')
        self.assertIn('must be a finite float between 0.0 and 1.0', r2['fallback_reason'])

    def test_32_single_packet_dual_mode(self):
        """Test 32: Single flow API supports both Basic and Enhanced modes."""
        # Basic flow
        flow_b = {"src_bytes": 200, "duration": 0.1, "dst_bytes": 500, "protocol_type": "TCP"}
        res_b = self.client.post('/api/scan-packet', json=flow_b)
        self.assertEqual(res_b.status_code, 200)
        r_b = res_b.get_json()['result']
        self.assertEqual(r_b['detection_mode'], 'Basic')
        self.assertEqual(len(r_b['features_used']), 4)

        # Enhanced flow
        flow_e = {
            "src_bytes": 200, "duration": 0.1, "dst_bytes": 500, "protocol_type": "TCP",
            "count": 10, "srv_count": 10, "flag": "SF", "same_srv_rate": 1.0, "serror_rate": 0.0, "dst_host_diff_srv_rate": 0.0
        }
        res_e = self.client.post('/api/scan-packet', json=flow_e)
        self.assertEqual(res_e.status_code, 200)
        r_e = res_e.get_json()['result']
        self.assertEqual(r_e['detection_mode'], 'Enhanced')
        self.assertEqual(len(r_e['features_used']), 10)

    def test_33_sample_csv_endpoints(self):
        """Test 33: Sample CSV download endpoints for Basic and Enhanced schemas."""
        res_b = self.client.get('/api/sample-csv?mode=basic')
        self.assertEqual(res_b.status_code, 200)
        self.assertIn(b'src_bytes,dst_bytes', res_b.data)
        self.assertIn(b'sentinelnet_sample_basic_telemetry.csv', res_b.headers.get('Content-Disposition', '').encode('utf-8'))

        res_e = self.client.get('/api/sample-csv?mode=enhanced')
        self.assertEqual(res_e.status_code, 200)
        self.assertIn(b'dst_host_diff_srv_rate', res_e.data)
        self.assertIn(b'sentinelnet_sample_enhanced_telemetry.csv', res_e.headers.get('Content-Disposition', '').encode('utf-8'))

    def test_34_annotated_csv_exported_fields(self):
        """Test 34: Annotated CSV output includes detection_mode, decision_status, and fallback_reason."""
        csv_content = """duration,protocol_type,src_bytes,dst_bytes,count,srv_count,flag,same_srv_rate,serror_rate,dst_host_diff_srv_rate
0.0,tcp,215,45096,1,1,SF,1.0,0.0,0.0
0.0,tcp,100,50,1,1,BAD_FLAG,1.0,0.0,0.0
"""
        data = {'file': (io.BytesIO(csv_content.encode('utf-8')), 'test_annotated.csv')}
        res = self.client.post('/api/scan-batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        csv_str = res.get_json()['csv_data']
        df_annotated = pd.read_csv(io.StringIO(csv_str))

        self.assertIn('detection_mode', df_annotated.columns)
        self.assertIn('decision_status', df_annotated.columns)
        self.assertIn('fallback_reason', df_annotated.columns)
        self.assertIn('predicted_classification', df_annotated.columns)
        self.assertIn('confidence_score', df_annotated.columns)

        self.assertEqual(df_annotated.loc[0, 'detection_mode'], 'Enhanced')
        self.assertEqual(df_annotated.loc[1, 'detection_mode'], 'Basic')
        self.assertIn('BAD_FLAG', str(df_annotated.loc[1, 'fallback_reason']))

    def test_35_preview_strict_json_with_missing_optional_fields(self):
        csv = ('src_bytes,dst_bytes,duration,protocol_type,count,srv_count,flag,same_srv_rate,serror_rate,dst_host_diff_srv_rate,src_ip\n'
               '20,30,1,tcp,,2,SF,1,0,0,\n'
               '20,30,1,INVALID,1,2,SF,1,0,0,\n')
        res = self.client.post('/api/scan-batch', data={'file': (io.BytesIO(csv.encode()), 'mixed.csv')}, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        result = res.get_json()
        json.dumps(result, allow_nan=False)
        self.assertEqual(result['valid_records'], 1)
        self.assertEqual(result['rejected_records'], 1)
        self.assertEqual(result['fallback_count'], 1)
        self.assertIsNone(result['preview_records'][0]['count'])

    def test_36_simplified_page_and_benchmark_summary(self):
        page = self.client.get('/')
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'Analyze Network Logs', page.data)
        self.assertIn(b'Download Analysis Report', page.data)
        res = self.client.get('/api/benchmark-summary')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data['models']), 6)
        self.assertEqual(len(data['missing_comparison']), 10)
        self.assertEqual(len(data['paper_comparison']), 10)
        half = next(row for row in data['paper_comparison'] if row['missing_target_pct'] == 50 and row['model'] == 'Adaptive SentinelNet')
        self.assertAlmostEqual(half['accuracy_mean'], 0.8345753368721928)

if __name__ == '__main__':
    unittest.main()
