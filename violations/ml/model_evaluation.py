# violations/ml/model_evaluation.py
# COMPREHENSIVE EVALUATION SUITE - All metrics for thesis

import numpy as np
import cv2
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    classification_report, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive ML model evaluator with ALL metrics for thesis:
    - Accuracy, Precision, Recall, F1
    - Confusion Matrix
    - ROC & AUC
    - Precision-Recall Curve
    - Matthews Correlation Coefficient
    - Per-class metrics
    - Visualizations
    """
    
    def __init__(self, model, inference_engine):
        """
        Args:
            model: Trained LSTM model
            inference_engine: InferenceEngine instance
        """
        self.model = model
        self.inference_engine = inference_engine
        self.results = {}
        
        logger.info("✅ ModelEvaluator initialized")
    
    def evaluate_on_videos(self, test_videos_dir, labels_file=None):
        """
        Evaluate model on test video directory.
        
        Args:
            test_videos_dir: Path to test videos
                            Structure:
                            test_videos/
                              littering/
                                video1.mp4
                                video2.mp4
                              normal/
                                video1.mp4
                                video2.mp4
            labels_file: Optional JSON file with labels
                        {"video_name.mp4": "littering" or "normal"}
        
        Returns:
            Evaluation results dictionary
        """
        
        print("\n" + "="*70)
        print("🧪 EVALUATING MODEL ON TEST SET")
        print("="*70)
        
        y_true = []  # Ground truth labels
        y_pred = []  # Model predictions
        y_pred_prob = []  # Model probabilities
        video_results = []  # Detailed results per video
        
        # Load videos from directory structure
        videos_to_test = []
        
        # Check if directory structure: littering/ and normal/
        littering_dir = os.path.join(test_videos_dir, 'littering')
        normal_dir = os.path.join(test_videos_dir, 'normal')
        
        if os.path.exists(littering_dir):
            for video_file in os.listdir(littering_dir):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    videos_to_test.append({
                        'path': os.path.join(littering_dir, video_file),
                        'label': 1,  # littering
                        'label_name': 'littering'
                    })
        
        if os.path.exists(normal_dir):
            for video_file in os.listdir(normal_dir):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    videos_to_test.append({
                        'path': os.path.join(normal_dir, video_file),
                        'label': 0,  # normal
                        'label_name': 'normal'
                    })
        
        if not videos_to_test:
            logger.error("No videos found in test directory")
            print("❌ No test videos found")
            return None
        
        print(f"\n📹 Found {len(videos_to_test)} test videos:")
        print(f"   Littering: {sum(1 for v in videos_to_test if v['label'] == 1)}")
        print(f"   Normal: {sum(1 for v in videos_to_test if v['label'] == 0)}")
        
        # Process each video
        print(f"\n🎬 Processing videos...")
        print("="*70)
        
        for i, video_info in enumerate(videos_to_test, 1):
            video_path = video_info['path']
            true_label = video_info['label']
            label_name = video_info['label_name']
            
            print(f"\n[{i}/{len(videos_to_test)}] {os.path.basename(video_path)}")
            print(f"   True label: {label_name}")
            
            # Run inference on video
            pred_label, pred_prob, details = self._evaluate_video(video_path)
            
            print(f"   Predicted: {'littering' if pred_label == 1 else 'normal'} ({pred_prob:.3f})")
            
            # Store results
            y_true.append(true_label)
            y_pred.append(pred_label)
            y_pred_prob.append(pred_prob)
            
            video_results.append({
                'video': os.path.basename(video_path),
                'true_label': true_label,
                'pred_label': pred_label,
                'pred_prob': pred_prob,
                'correct': pred_label == true_label,
                'details': details
            })
            
            if pred_label == true_label:
                print(f"   ✅ CORRECT")
            else:
                print(f"   ❌ WRONG (False {'Positive' if pred_label == 1 else 'Negative'})")
        
        # Calculate all metrics
        print("\n" + "="*70)
        print("📊 CALCULATING EVALUATION METRICS")
        print("="*70)
        
        self.results = self._calculate_all_metrics(
            y_true, y_pred, y_pred_prob, video_results
        )
        
        return self.results
    
    def _evaluate_video(self, video_path):
        """
        Evaluate single video and return prediction.
        
        Returns:
            (predicted_label, predicted_probability, details)
        """
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        confidences = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            results = self.inference_engine.process_frame(frame)
            
            if results['prediction']:
                confidence = results['prediction']['confidence']
                confidences.append(confidence)
        
        cap.release()
        
        # Determine prediction based on peak confidence
        if not confidences:
            pred_label = 0
            pred_prob = 0.0
        else:
            peak_conf = max(confidences)
            avg_conf = np.mean(confidences)
            
            # Use peak confidence for prediction
            pred_prob = peak_conf
            pred_label = 1 if peak_conf > 0.5 else 0
        
        details = {
            'frames_processed': frame_count,
            'peak_confidence': max(confidences) if confidences else 0.0,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'min_confidence': min(confidences) if confidences else 0.0,
            'max_confidence': max(confidences) if confidences else 0.0,
            'std_confidence': np.std(confidences) if confidences else 0.0
        }
        
        return pred_label, pred_prob, details
    
    def _calculate_all_metrics(self, y_true, y_pred, y_pred_prob, video_results):
        """
        Calculate ALL evaluation metrics for thesis.
        
        Returns comprehensive metrics dictionary
        """
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_prob = np.array(y_pred_prob)
        
        results = {}
        
        # ========== BASIC METRICS ==========
        print("\n1️⃣ BASIC METRICS:")
        print("-" * 70)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results['basic_metrics'] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        
        # ========== CONFUSION MATRIX ==========
        print("\n2️⃣ CONFUSION MATRIX:")
        print("-" * 70)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results['confusion_matrix'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        print(f"   True Negatives (TN):  {tn}")
        print(f"   False Positives (FP): {fp}")
        print(f"   False Negatives (FN): {fn}")
        print(f"   True Positives (TP):  {tp}")
        print(f"\n   Confusion Matrix:")
        print(f"   [[{tn}  {fp}]")
        print(f"    [{fn}  {tp}]]")
        
        # ========== SENSITIVITY & SPECIFICITY ==========
        print("\n3️⃣ SENSITIVITY & SPECIFICITY:")
        print("-" * 70)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results['sensitivity_specificity'] = {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity)
        }
        
        print(f"   Sensitivity (Recall): {sensitivity:.4f}")
        print(f"   Specificity:          {specificity:.4f}")
        
        # ========== ROC & AUC ==========
        print("\n4️⃣ ROC & AUC:")
        print("-" * 70)
        
        if len(np.unique(y_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            
            results['roc_auc'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc_score': float(roc_auc)
            }
            
            print(f"   AUC Score: {roc_auc:.4f}")
        else:
            print(f"   ⚠️  Cannot calculate ROC (only one class in test set)")
        
        # ========== PRECISION-RECALL CURVE ==========
        print("\n5️⃣ PRECISION-RECALL CURVE:")
        print("-" * 70)
        
        if len(np.unique(y_true)) > 1:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
            ap = average_precision_score(y_true, y_pred_prob)
            
            results['precision_recall'] = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'average_precision': float(ap)
            }
            
            print(f"   Average Precision: {ap:.4f}")
        
        # ========== MATTHEWS CORRELATION COEFFICIENT ==========
        print("\n6️⃣ MATTHEWS CORRELATION COEFFICIENT (MCC):")
        print("-" * 70)
        
        mcc = matthews_corrcoef(y_true, y_pred)
        
        results['mcc'] = float(mcc)
        print(f"   MCC: {mcc:.4f}")
        print(f"   (Range: -1 to 1, where 1 = perfect prediction)")
        
        # ========== PER-CLASS METRICS ==========
        print("\n7️⃣ PER-CLASS METRICS:")
        print("-" * 70)
        
        class_report = classification_report(y_true, y_pred, 
                                            target_names=['Normal', 'Littering'],
                                            output_dict=True,
                                            zero_division=0)
        
        results['per_class_metrics'] = class_report
        
        print(f"\n   Normal Class:")
        print(f"     Precision: {class_report['Normal']['precision']:.4f}")
        print(f"     Recall:    {class_report['Normal']['recall']:.4f}")
        print(f"     F1-Score:  {class_report['Normal']['f1-score']:.4f}")
        
        print(f"\n   Littering Class:")
        print(f"     Precision: {class_report['Littering']['precision']:.4f}")
        print(f"     Recall:    {class_report['Littering']['recall']:.4f}")
        print(f"     F1-Score:  {class_report['Littering']['f1-score']:.4f}")
        
        # ========== FALSE POSITIVE & FALSE NEGATIVE RATES ==========
        print("\n8️⃣ FALSE POSITIVE & FALSE NEGATIVE RATES:")
        print("-" * 70)
        
        fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        results['error_rates'] = {
            'false_positive_rate': float(fpr_rate),
            'false_negative_rate': float(fnr_rate)
        }
        
        print(f"   FPR (False Positive Rate): {fpr_rate:.4f} ({fpr_rate*100:.2f}%)")
        print(f"   FNR (False Negative Rate): {fnr_rate:.4f} ({fnr_rate*100:.2f}%)")
        
        # ========== VIDEO-LEVEL RESULTS ==========
        print("\n9️⃣ VIDEO-LEVEL RESULTS:")
        print("-" * 70)
        
        correct = sum(1 for v in video_results if v['correct'])
        total = len(video_results)
        
        print(f"   Videos Correct: {correct}/{total}")
        
        results['video_results'] = video_results
        results['summary'] = {
            'total_videos': total,
            'correct_predictions': correct,
            'incorrect_predictions': total - correct
        }
        
        return results
    
    def generate_visualizations(self, output_dir='./evaluation_results'):
        """
        Generate publication-ready visualizations for thesis.
        
        Creates:
        - Confusion Matrix heatmap
        - ROC Curve
        - Precision-Recall Curve
        - Metrics comparison bar chart
        """
        
        if not self.results:
            logger.warning("No results to visualize. Run evaluate_on_videos first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("📊 GENERATING VISUALIZATIONS")
        print("="*70)
        
        # Set style for thesis
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        
        # 1. Confusion Matrix Heatmap
        print("\n1. Confusion Matrix Heatmap...")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = np.array([
            [self.results['confusion_matrix']['true_negatives'],
             self.results['confusion_matrix']['false_positives']],
            [self.results['confusion_matrix']['false_negatives'],
             self.results['confusion_matrix']['true_positives']]
        ])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Normal', 'Littering'],
                   yticklabels=['Normal', 'Littering'],
                   ax=ax)
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '01_confusion_matrix.png'), dpi=300)
        print("   ✅ Saved: 01_confusion_matrix.png")
        plt.close()
        
        # 2. Metrics Bar Chart
        print("\n2. Metrics Comparison Bar Chart...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        values = [
            self.results['basic_metrics']['accuracy'],
            self.results['basic_metrics']['precision'],
            self.results['basic_metrics']['recall'],
            self.results['basic_metrics']['f1_score'],
            self.results.get('roc_auc', {}).get('auc_score', 0)
        ]
        
        bars = ax.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '02_metrics_comparison.png'), dpi=300)
        print("   ✅ Saved: 02_metrics_comparison.png")
        plt.close()
        
        # 3. ROC Curve
        if 'roc_auc' in self.results:
            print("\n3. ROC Curve...")
            fig, ax = plt.subplots(figsize=(8, 8))
            
            fpr = self.results['roc_auc']['fpr']
            tpr = self.results['roc_auc']['tpr']
            auc_score = self.results['roc_auc']['auc_score']
            
            ax.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ROC Curve (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '03_roc_curve.png'), dpi=300)
            print("   ✅ Saved: 03_roc_curve.png")
            plt.close()
        
        print(f"\n✅ All visualizations saved to: {output_dir}")
    
    def generate_report(self, output_file='evaluation_report.json'):
        """
        Generate comprehensive JSON report for thesis.
        """
        
        if not self.results:
            logger.warning("No results to report")
            return
        
        # Convert arrays to lists for JSON serialization
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_evaluation_results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Report saved to: {output_file}")
        
        return report
    
    def print_summary(self):
        """Print beautiful summary for thesis."""
        
        if not self.results:
            print("❌ No results to print")
            return
        
        print("\n" + "="*70)
        print("📋 EVALUATION SUMMARY")
        print("="*70)
        
        print(f"\n{'Metric':<30} {'Value':<15} {'Status'}")
        print("-" * 70)
        
        acc = self.results['basic_metrics']['accuracy']
        print(f"{'Accuracy':<30} {acc:.4f} ({acc*100:.2f}%) {'✅' if acc > 0.8 else '⚠️'}")
        
        prec = self.results['basic_metrics']['precision']
        print(f"{'Precision':<30} {prec:.4f} {'✅' if prec > 0.8 else '⚠️'}")
        
        rec = self.results['basic_metrics']['recall']
        print(f"{'Recall':<30} {rec:.4f} {'✅' if rec > 0.8 else '⚠️'}")
        
        f1 = self.results['basic_metrics']['f1_score']
        print(f"{'F1-Score':<30} {f1:.4f} {'✅' if f1 > 0.8 else '⚠️'}")
        
        if 'roc_auc' in self.results:
            auc_val = self.results['roc_auc']['auc_score']
            print(f"{'AUC':<30} {auc_val:.4f} {'✅' if auc_val > 0.8 else '⚠️'}")
        
        mcc = self.results['mcc']
        print(f"{'Matthews Corr. Coef.':<30} {mcc:.4f}")
        
        tn = self.results['confusion_matrix']['true_negatives']
        fp = self.results['confusion_matrix']['false_positives']
        fn = self.results['confusion_matrix']['false_negatives']
        tp = self.results['confusion_matrix']['true_positives']
        
        print(f"\n{'True Negatives':<30} {tn}")
        print(f"{'False Positives':<30} {fp}")
        print(f"{'False Negatives':<30} {fn}")
        print(f"{'True Positives':<30} {tp}")
        
        print("\n" + "="*70)