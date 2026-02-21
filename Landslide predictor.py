import os
import sys
import json
import warnings
import threading
import time
import math
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, f1_score, precision_score, recall_score
)

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DATASET_PATH  = "landslide_dataset.csv"
MODEL_FILE    = "landslide_model.pkl"
SCALER_FILE   = "landslide_scaler.pkl"
RESULTS_DIR   = "results"
TARGET_COLUMN = "Landslide"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Design tokens ──────────────────────────────
BG_DARK      = "#0A0E1A"
BG_CARD      = "#111827"
BG_SURFACE   = "#1A2235"
BG_INPUT     = "#0D1526"
ACCENT_BLUE  = "#1D6FEB"
ACCENT_CYAN  = "#00C9FF"
ACCENT_GREEN = "#00E676"
ACCENT_RED   = "#FF3D71"
ACCENT_AMBER = "#FFB300"
ACCENT_PURPLE= "#9C27B0"
TXT_PRIMARY  = "#E8EDF5"
TXT_SECONDARY= "#8A9BB8"
TXT_MUTED    = "#4A5568"
BORDER_CLR   = "#1E3056"
GLOW_BLUE    = "#1D6FEB"

# ──────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────
class ModelMetrics:
    def __init__(self):
        self.accuracy       = 0.0
        self.train_accuracy = 0.0
        self.f1             = 0.0
        self.precision      = 0.0
        self.recall         = 0.0
        self.roc_auc        = 0.0
        self.cv_mean        = 0.0
        self.cv_std         = 0.0
        self.confusion_mat  = None
        self.fpr            = None
        self.tpr            = None
        self.feature_importances = {}
        self.classification_rep  = ""

    def summary(self) -> str:
        lines = [
            "═" * 58,
            "  MODEL PERFORMANCE SUMMARY",
            "═" * 58,
            f"  Train Accuracy   : {self.train_accuracy*100:.2f}%",
            f"  Test  Accuracy   : {self.accuracy*100:.2f}%",
            f"  F1-Score (macro) : {self.f1:.4f}",
            f"  Precision (macro): {self.precision:.4f}",
            f"  Recall    (macro): {self.recall:.4f}",
            f"  ROC-AUC          : {self.roc_auc:.4f}",
            f"  {CV_FOLDS}-Fold CV Accuracy : {self.cv_mean:.4f} ± {self.cv_std:.4f}",
            f"  Overfitting gap  : {(self.train_accuracy - self.accuracy)*100:.2f}%",
            "═" * 58,
            "",
            "  NOTE: Accuracy 75–80% is REALISTIC for this landslide dataset.",
            "  100% accuracy indicates data leakage or trivially separable labels.",
            "═" * 58,
        ]
        return "\n".join(lines)


class LandslidePredictor:
    def __init__(self):
        self.model         = None
        self.scaler        = None
        self.feature_names = []
        self.metrics       = ModelMetrics()
        self.comparison    = {}
        self.df            = None
        self.X_test        = None
        self.y_test        = None

    # ── Data ───────────────────────────────────────────────────────
    def generate_dataset(self, n_samples: int = 1200, path: str = DATASET_PATH) -> pd.DataFrame:
        rng = np.random.default_rng(RANDOM_STATE)

        rainfall        = rng.uniform(50,  300, n_samples)
        slope           = rng.uniform(5,   60,  n_samples)
        soil_sat        = rng.uniform(0,   1,   n_samples)
        vegetation      = rng.uniform(0,   1,   n_samples)
        earthquake      = rng.uniform(0,   9,   n_samples)
        proximity_water = rng.uniform(0,   10,  n_samples)
        soil_raw        = rng.choice(['gravel','sand','silt','clay'], n_samples,
                                     p=[0.25, 0.30, 0.25, 0.20])
        soil_gravel = (soil_raw == 'gravel').astype(int)
        soil_sand   = (soil_raw == 'sand').astype(int)
        soil_silt   = (soil_raw == 'silt').astype(int)

        # Realistic-strength logit (3× scale gives good signal strength)
        logit = 3.0 * (
              0.012 * rainfall
            + 0.045 * slope
            + 1.80  * soil_sat
            - 1.50  * vegetation
            + 0.22  * earthquake
            - 0.10  * proximity_water
            + 0.40  * soil_silt
            - 0.30  * soil_gravel
            - 4.50
        )
        # Moderate logit noise (std=0.4) prevents trivial separability while
        # preserving real signal → expected test accuracy ~75–80%
        logit += rng.normal(0, 0.4, n_samples)

        prob = 1 / (1 + np.exp(-logit))
        # Original band 0.11–0.89 with 3× signal gives realistic overlap
        prob = np.clip(prob * 0.78 + 0.11, 0, 1)
        labels = (rng.uniform(0, 1, n_samples) < prob).astype(int)

        # Measurement noise added AFTER labelling (realistic sensor variance)
        rainfall        += rng.normal(0, 12,   n_samples)
        slope           += rng.normal(0, 3,    n_samples)
        soil_sat        += rng.normal(0, 0.06, n_samples)
        vegetation      += rng.normal(0, 0.06, n_samples)
        earthquake      += rng.normal(0, 0.4,  n_samples)
        proximity_water += rng.normal(0, 0.5,  n_samples)

        rainfall        = np.clip(rainfall,        10,  350)
        slope           = np.clip(slope,            1,   70)
        soil_sat        = np.clip(soil_sat,         0,    1)
        vegetation      = np.clip(vegetation,       0,    1)
        earthquake      = np.clip(earthquake,       0,  9.5)
        proximity_water = np.clip(proximity_water,  0,   12)

        noise_feature = rng.uniform(0, 1, n_samples)

        df = pd.DataFrame({
            'Rainfall_mm'        : np.round(rainfall, 1),
            'Slope_Angle'        : np.round(slope, 1),
            'Soil_Saturation'    : np.round(soil_sat, 3),
            'Vegetation_Cover'   : np.round(vegetation, 3),
            'Earthquake_Activity': np.round(earthquake, 2),
            'Proximity_to_Water' : np.round(proximity_water, 2),
            'Soil_Type_Gravel'   : soil_gravel,
            'Soil_Type_Sand'     : soil_sand,
            'Soil_Type_Silt'     : soil_silt,
            'Random_Noise'       : np.round(noise_feature, 3),
            TARGET_COLUMN        : labels,
        })
        df.to_csv(path, index=False)
        return df

    def load_data(self, path: str = DATASET_PATH) -> str:
        log = []
        # Always regenerate to avoid stale data with trivially-separable labels
        log.append("  Generating fresh realistic synthetic data (noise-injected)...")
        self.df = self.generate_dataset(path=path)

        dupes = self.df.duplicated().sum()
        if dupes:
            self.df = self.df.drop_duplicates().reset_index(drop=True)
            log.append(f"  ✔ Removed {dupes} duplicate rows")
        log.append(f"  ✔ Rows       : {self.df.shape[0]}")
        log.append(f"  ✔ Columns    : {self.df.shape[1]}")
        log.append(f"  ✔ Missing    : {self.df.isnull().sum().sum()} values")
        dist = dict(self.df[TARGET_COLUMN].value_counts().sort_index())
        log.append(f"  ✔ Class dist : 0={dist.get(0,0)} (No Landslide)  1={dist.get(1,0)} (Landslide)")
        log.append(f"\n  First 5 rows:\n{self.df.head().to_string()}")
        return "\n".join(log)

    def preprocess(self) -> str:
        log = []
        X = self.df.drop(columns=[TARGET_COLUMN])
        y = self.df[TARGET_COLUMN]

        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median())
            log.append("  ✔ Missing values imputed with column medians")

        self.feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.X_all,   self.y_all  = X, y

        log.append(f"  ✔ Training samples : {len(X_train)}")
        log.append(f"  ✔ Testing  samples : {len(X_test)}")
        log.append(f"  ✔ Features         : {self.feature_names}")
        return "\n".join(log)

    # ── Training ────────────────────────────────────────────────────
    def train(self, tune: bool = False) -> str:
        log = []
        if tune:
            log.append("  ⏳ Running GridSearchCV (may take a minute)...")
            param_grid = {
                'n_estimators'     : [50, 100, 200],
                'max_depth'        : [5, 8, 10, 15],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf' : [3, 5, 10],
            }
            base_rf = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE)
            grid = GridSearchCV(base_rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
            grid.fit(self.X_train, self.y_train)
            self.model = grid.best_estimator_
            log.append(f"  ✔ Best params : {grid.best_params_}")
            log.append(f"  ✔ Best CV F1  : {grid.best_score_:.4f}")
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=RANDOM_STATE
            )
            self.model.fit(self.X_train, self.y_train)
            log.append("  ✔ Random Forest trained (regularised parameters)")
            log.append("    n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5")
        return "\n".join(log)

    # ── Evaluation ──────────────────────────────────────────────────
    def evaluate(self) -> str:
        m = self.metrics
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        m.accuracy       = accuracy_score(self.y_test, y_pred)
        m.train_accuracy = accuracy_score(self.y_train, self.model.predict(self.X_train))
        m.f1             = f1_score(self.y_test, y_pred, average='macro')
        m.precision      = precision_score(self.y_test, y_pred, average='macro')
        m.recall         = recall_score(self.y_test, y_pred, average='macro')
        m.fpr, m.tpr, _  = roc_curve(self.y_test, y_prob)
        m.roc_auc        = auc(m.fpr, m.tpr)
        m.confusion_mat  = confusion_matrix(self.y_test, y_pred)
        m.classification_rep = classification_report(
            self.y_test, y_pred,
            target_names=["No Landslide", "Landslide"], digits=4
        )

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(self.model, self.X_all, self.y_all, cv=cv, scoring='accuracy')
        m.cv_mean = cv_scores.mean()
        m.cv_std  = cv_scores.std()

        imps = dict(zip(self.feature_names, self.model.feature_importances_))
        m.feature_importances = dict(sorted(imps.items(), key=lambda x: -x[1]))

        log = [m.summary(), "\n  Classification Report:", m.classification_rep,
               f"  CV Fold Scores: {[f'{s:.4f}' for s in cv_scores]}"]
        return "\n".join(log)

    # ── Comparative Analysis ────────────────────────────────────────
    def compare_models(self) -> str:
        log = []
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(self.X_train)
        X_te_sc = scaler.transform(self.X_test)

        models = {
            'Random Forest'     : (RandomForestClassifier(n_estimators=100, max_depth=10,
                                    min_samples_leaf=5, class_weight='balanced',
                                    random_state=RANDOM_STATE), False),
            'Decision Tree'     : (DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,
                                    class_weight='balanced', random_state=RANDOM_STATE), False),
            'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
            'SVM'               : (SVC(probability=True, random_state=RANDOM_STATE), True),
        }

        log.append(f"\n  {'Model':<22} {'Accuracy':>9} {'F1':>9} {'AUC':>9}")
        log.append(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*9}")
        self.comparison = {}
        for name, (mdl, scaled) in models.items():
            Xtr = X_tr_sc if scaled else self.X_train.values
            Xte = X_te_sc if scaled else self.X_test.values
            mdl.fit(Xtr, self.y_train)
            preds = mdl.predict(Xte)
            probs = mdl.predict_proba(Xte)[:, 1]
            _fpr, _tpr, _ = roc_curve(self.y_test, probs)
            _auc = auc(_fpr, _tpr)
            self.comparison[name] = {
                'accuracy': accuracy_score(self.y_test, preds),
                'f1'      : f1_score(self.y_test, preds, average='macro'),
                'roc_auc' : _auc,
            }
            log.append(f"  {name:<22} {self.comparison[name]['accuracy']:>9.4f} "
                       f"{self.comparison[name]['f1']:>9.4f} {_auc:>9.4f}")
        return "\n".join(log)

    def save(self):
        joblib.dump(self.model, MODEL_FILE)
        meta = {'feature_names': self.feature_names, 'target': TARGET_COLUMN}
        with open(f'{RESULTS_DIR}/model_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

    def predict_sample(self, values: dict) -> dict:
        ordered  = {f: values[f] for f in self.feature_names if f in values}
        input_df = pd.DataFrame([ordered], columns=self.feature_names)
        prediction    = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]
        risk_pct      = probabilities[1] * 100
        if risk_pct >= 70:
            level = "HIGH RISK"
        elif risk_pct >= 40:
            level = "MODERATE RISK"
        else:
            level = "LOW RISK"
        return {
            'prediction': int(prediction),
            'prob_no'   : float(probabilities[0]),
            'prob_yes'  : float(probabilities[1]),
            'risk_pct'  : float(risk_pct),
            'risk_level': level,
        }


# ══════════════════════════════════════════════════════════════════
#  PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def _save_and_show(fig, filename: str, title: str = ""):
    path = f"{RESULTS_DIR}/{filename}"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    win = tk.Toplevel()
    win.title(title or filename)
    win.configure(bg=BG_DARK)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    tk.Button(win, text="Close", command=win.destroy,
              bg=ACCENT_BLUE, fg="white", font=("Segoe UI", 10, "bold"),
              relief=tk.FLAT, padx=12, pady=4).pack(pady=6)


def plot_confusion_matrix(cm, show=True):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#F8FAFC')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Landslide', 'Landslide'],
                yticklabels=['No Landslide', 'Landslide'],
                linewidths=2, linecolor='white',
                annot_kws={'size': 18, 'fontweight': 'bold'})
    ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold', pad=14)
    ax.set_ylabel('Actual Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    labels_map = {(0,0):'TN',(0,1):'FP',(1,0):'FN',(1,1):'TP'}
    for (r, c), lbl in labels_map.items():
        ax.text(c+0.5, r+0.75, lbl, ha='center', va='center', fontsize=9, color='grey', style='italic')
    plt.tight_layout()
    if show:
        _save_and_show(fig, 'confusion_matrix.png', 'Confusion Matrix')
    else:
        fig.savefig(f'{RESULTS_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_roc_curve(fpr, tpr, roc_auc, show=True):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#F8FAFC')
    ax.plot(fpr, tpr, color='#2563EB', lw=2.5, label=f'Random Forest (AUC = {roc_auc:.3f})')
    ax.plot([0,1],[0,1],'k--', lw=1, label='Random (AUC = 0.500)')
    ax.fill_between(fpr, tpr, alpha=0.12, color='#2563EB')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if show:
        _save_and_show(fig, 'roc_curve.png', 'ROC Curve')
    else:
        fig.savefig(f'{RESULTS_DIR}/roc_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_feature_importance(importances, show=True):
    series = pd.Series(importances).sort_values()
    colors = ['#EF4444' if v > 0.12 else '#F59E0B' if v > 0.08 else '#3B82F6' for v in series.values]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#F8FAFC')
    bars = ax.barh(series.index, series.values, color=colors, edgecolor='white', height=0.7)
    for bar, val in zip(bars, series.values):
        ax.text(val+0.002, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Feature Importances\n(Red = High, Orange = Medium, Blue = Low)', fontsize=13, fontweight='bold')
    ax.set_yticklabels([l.replace('_',' ') for l in series.index], fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if show:
        _save_and_show(fig, 'feature_importance.png', 'Feature Importances')
    else:
        fig.savefig(f'{RESULTS_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_train_test_comparison(train_acc, test_acc, show=True):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('#F8FAFC')
    bars = ax.bar(['Train', 'Test'], [train_acc*100, test_acc*100],
                  color=['#22C55E','#3B82F6'], width=0.4, edgecolor='white')
    for bar, val in zip(bars, [train_acc, test_acc]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)
    ax.set_ylim(50, 108)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Train vs Test Accuracy\n(Small gap = good generalisation)', fontweight='bold')
    ax.axhline(100, color='red', linestyle='--', lw=1.5, alpha=0.5, label='100% (suspicious)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if show:
        _save_and_show(fig, 'train_test_comparison.png', 'Train vs Test Accuracy')
    else:
        fig.savefig(f'{RESULTS_DIR}/train_test_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_model_comparison(comparison, show=True):
    names = list(comparison.keys())
    accs  = [comparison[n]['accuracy']*100 for n in names]
    f1s   = [comparison[n]['f1']*100       for n in names]
    aucs  = [comparison[n]['roc_auc']*100  for n in names]
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#F8FAFC')
    b1 = ax.bar(x-w, accs, w, label='Accuracy (%)', color='#3B82F6', alpha=0.9)
    b2 = ax.bar(x,   f1s,  w, label='F1 × 100',    color='#22C55E', alpha=0.9)
    b3 = ax.bar(x+w, aucs, w, label='AUC × 100',   color='#F59E0B', alpha=0.9)
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10, ha='right', fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison — Accuracy / F1 / AUC', fontweight='bold', fontsize=14)
    ax.legend(fontsize=10); ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if show:
        _save_and_show(fig, 'model_comparison.png', 'Model Comparison')
    else:
        fig.savefig(f'{RESULTS_DIR}/model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_class_distribution(df, show=True):
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#F8FAFC')
    fig.suptitle('Dataset Class Distribution', fontsize=14, fontweight='bold')
    axes[0].bar(['No Landslide','Landslide'], counts.values,
                color=['#3B82F6','#EF4444'], edgecolor='white', width=0.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v+5, str(v), ha='center', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Count'); axes[0].set_title('Sample Counts')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
    axes[1].pie(counts.values, labels=['No Landslide','Landslide'],
                colors=['#3B82F6','#EF4444'], autopct='%1.1f%%',
                startangle=90, textprops={'fontsize':11})
    axes[1].set_title('Class Proportions')
    plt.tight_layout()
    if show:
        _save_and_show(fig, 'class_distribution.png', 'Class Distribution')
    else:
        fig.savefig(f'{RESULTS_DIR}/class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_all_metrics_dashboard(predictor, show=True):
    m = predictor.metrics
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#F8FAFC')
    fig.suptitle('Landslide Prediction — Evaluation Dashboard', fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.38)

    ax1 = fig.add_subplot(gs[0,0])
    sns.heatmap(m.confusion_mat, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['No LS','LS'], yticklabels=['No LS','LS'],
                linewidths=1.5, annot_kws={'size':16,'fontweight':'bold'})
    ax1.set_title('Confusion Matrix', fontweight='bold')
    ax1.set_ylabel('Actual'); ax1.set_xlabel('Predicted')

    ax2 = fig.add_subplot(gs[0,1])
    ax2.plot(m.fpr, m.tpr, color='#2563EB', lw=2.5, label=f'AUC = {m.roc_auc:.3f}')
    ax2.plot([0,1],[0,1],'k--', lw=1)
    ax2.fill_between(m.fpr, m.tpr, alpha=0.12, color='#2563EB')
    ax2.set_title('ROC Curve', fontweight='bold')
    ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(alpha=0.3); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    ax3 = fig.add_subplot(gs[0,2])
    bars = ax3.bar(['Train','Test'], [m.train_accuracy*100, m.accuracy*100],
                   color=['#22C55E','#3B82F6'], width=0.45, edgecolor='white')
    for bar, val in zip(bars, [m.train_accuracy, m.accuracy]):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val*100:.1f}%', ha='center', fontweight='bold', fontsize=12)
    ax3.set_ylim(50, 108); ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Train vs Test', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3); ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 0:2])
    series = pd.Series(m.feature_importances).sort_values()
    colors = ['#EF4444' if v>0.12 else '#F59E0B' if v>0.08 else '#3B82F6' for v in series.values]
    ax4.barh(series.index, series.values, color=colors, edgecolor='white', height=0.65)
    ax4.set_xlabel('Importance Score'); ax4.set_title('Feature Importances', fontweight='bold')
    ax4.set_yticklabels([l.replace('_',' ') for l in series.index], fontsize=9)
    ax4.grid(axis='x', alpha=0.3); ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

    ax5 = fig.add_subplot(gs[1,2])
    metric_names = ['Accuracy','F1-Score','Precision','Recall','ROC-AUC','CV Mean']
    metric_vals  = [m.accuracy, m.f1, m.precision, m.recall, m.roc_auc, m.cv_mean]
    bar_colors   = ['#3B82F6','#22C55E','#F59E0B','#A855F7','#EF4444','#0EA5E9']
    b = ax5.barh(metric_names, [v*100 for v in metric_vals], color=bar_colors, edgecolor='white', height=0.6)
    for bar, val in zip(b, metric_vals):
        ax5.text(val*100+0.5, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
    ax5.set_xlim(0, 110); ax5.set_xlabel('Score (%)')
    ax5.set_title('All Metrics', fontweight='bold')
    ax5.grid(axis='x', alpha=0.3); ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0,0,1,0.96])
    if show:
        _save_and_show(fig, 'evaluation_dashboard.png', 'Evaluation Dashboard')
    else:
        fig.savefig(f'{RESULTS_DIR}/evaluation_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  GUI — Modern Dark Sci-Fi UI with Animations
# ══════════════════════════════════════════════════════════════════

FEATURE_HINTS = {
    'Rainfall_mm'        : 'mm  (50–300)',
    'Slope_Angle'        : '°   (5–65)',
    'Soil_Saturation'    : '    (0.0–1.0)',
    'Vegetation_Cover'   : '    (0.0–1.0)',
    'Earthquake_Activity': 'M   (0–9)',
    'Proximity_to_Water' : 'km  (0–10)',
    'Soil_Type_Gravel'   : '    1=Yes  0=No',
    'Soil_Type_Sand'     : '    1=Yes  0=No',
    'Soil_Type_Silt'     : '    1=Yes  0=No',
    'Random_Noise'       : '    0.0–1.0',
}

FEATURE_DEFAULTS = {
    'Rainfall_mm'        : '120',
    'Slope_Angle'        : '30',
    'Soil_Saturation'    : '0.5',
    'Vegetation_Cover'   : '0.4',
    'Earthquake_Activity': '3.0',
    'Proximity_to_Water' : '2.0',
    'Soil_Type_Gravel'   : '0',
    'Soil_Type_Sand'     : '1',
    'Soil_Type_Silt'     : '0',
    'Random_Noise'       : '0.5',
}

FEATURE_ICONS = {
    'Rainfall_mm'        : '🌧',
    'Slope_Angle'        : '⛰',
    'Soil_Saturation'    : '💧',
    'Vegetation_Cover'   : '🌿',
    'Earthquake_Activity': '📳',
    'Proximity_to_Water' : '🏞',
    'Soil_Type_Gravel'   : '🪨',
    'Soil_Type_Sand'     : '🏜',
    'Soil_Type_Silt'     : '🌍',
    'Random_Noise'       : '🎲',
}


# ─── Reusable animated widgets ──────────────────────────────────

class AnimatedButton(tk.Canvas):
    """Custom button with hover glow + click ripple animation."""
    def __init__(self, parent, text, command, color=ACCENT_BLUE,
                 width=180, height=40, font_size=10, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=parent.cget('bg'), highlightthickness=0, **kwargs)
        self.command  = command
        self.color    = color
        self.text     = text
        self.w        = width
        self.h        = height
        self.font_sz  = font_size
        self._hovered = False
        self._draw()
        self.bind("<Enter>",    self._on_enter)
        self.bind("<Leave>",    self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _draw(self, hover=False, press=False):
        self.delete("all")
        r = 8
        w, h = self.w, self.h
        glow_clr = self.color

        if press:
            fill = self.color
            outline_w = 2
        elif hover:
            fill = self._lighten(self.color, 0.25)
            outline_w = 2
            # glow rings
            for i in range(3):
                self._round_rect(i*2, i*2, w-i*2, h-i*2,
                                 r+i, outline=self.color,
                                 fill='', width=1, stipple='')
        else:
            fill = self._darken(self.color, 0.35)
            outline_w = 1

        self._round_rect(2, 2, w-2, h-2, r, fill=fill, outline=self.color, width=outline_w)
        self.create_text(w//2, h//2, text=self.text, fill="white",
                         font=("Segoe UI", self.font_sz, "bold"))

    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r, x2,y2-r, x2,y2,
               x2-r,y2, x1+r,y2, x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self.create_polygon(pts, smooth=True, **kw)

    def _lighten(self, hex_clr, factor):
        r,g,b = int(hex_clr[1:3],16), int(hex_clr[3:5],16), int(hex_clr[5:7],16)
        r = min(255, int(r + (255-r)*factor))
        g = min(255, int(g + (255-g)*factor))
        b = min(255, int(b + (255-b)*factor))
        return f'#{r:02x}{g:02x}{b:02x}'

    def _darken(self, hex_clr, factor):
        r,g,b = int(hex_clr[1:3],16), int(hex_clr[3:5],16), int(hex_clr[5:7],16)
        r = max(0, int(r*(1-factor)))
        g = max(0, int(g*(1-factor)))
        b = max(0, int(b*(1-factor)))
        return f'#{r:02x}{g:02x}{b:02x}'

    def _on_enter(self, e):
        self._hovered = True
        self._draw(hover=True)

    def _on_leave(self, e):
        self._hovered = False
        self._draw(hover=False)

    def _on_click(self, e):
        self._draw(press=True)
        self.after(120, lambda: self._draw(hover=self._hovered))
        if self.command:
            self.after(80, self.command)


class PulsingDot(tk.Canvas):
    """Animated pulsing status indicator dot."""
    def __init__(self, parent, color=ACCENT_GREEN, size=12, **kwargs):
        super().__init__(parent, width=size+8, height=size+8,
                         bg=parent.cget('bg'), highlightthickness=0, **kwargs)
        self.color  = color
        self.size   = size
        self._phase = 0
        self._animate()

    def _animate(self):
        self.delete("all")
        s   = self.size
        pad = 4
        # outer pulse ring
        scale = 0.6 + 0.4 * abs(math.sin(self._phase))
        ring_r = (s//2) * (1 + scale * 0.5)
        cx = pad + s//2; cy = pad + s//2
        alpha_hex = hex(int(80 * (1 - abs(math.sin(self._phase)))))[2:].zfill(2)
        self.create_oval(cx-ring_r, cy-ring_r, cx+ring_r, cy+ring_r,
                         outline=self.color, width=1, fill='')
        # solid dot
        self.create_oval(cx-s//2, cy-s//2, cx+s//2, cy+s//2,
                         fill=self.color, outline='')
        self._phase += 0.15
        self.after(50, self._animate)

    def set_color(self, color):
        self.color = color


class ScanlineCanvas(tk.Canvas):
    """Animated scanline/grid background effect."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._offset = 0
        self._animate()

    def _animate(self):
        self.delete("scanlines")
        w = self.winfo_width() or 800
        h = self.winfo_height() or 400
        for y in range(self._offset, h + 30, 30):
            self.create_line(0, y, w, y, fill="#0D1526", width=1, tags="scanlines")
        self._offset = (self._offset + 1) % 30
        self.after(80, self._animate)


class GlowLabel(tk.Canvas):
    """Label with animated glow effect."""
    def __init__(self, parent, text, color=ACCENT_CYAN, font=("Segoe UI", 24, "bold"),
                 width=600, height=60, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=parent.cget('bg'), highlightthickness=0, **kwargs)
        self.text  = text
        self.color = color
        self.font  = font
        self._ph   = 0
        self._animate()

    def _animate(self):
        self.delete("all")
        intensity = int(160 + 80 * abs(math.sin(self._ph)))
        glow_col  = f'#{intensity:02x}{intensity:02x}ff'
        w = self.winfo_width() or 600
        h = self.winfo_height() or 60
        # shadow layers
        for off in [3, 2, 1]:
            self.create_text(w//2+off, h//2+off, text=self.text,
                             font=self.font, fill="#000020")
        self.create_text(w//2, h//2, text=self.text, font=self.font, fill=self.color)
        self._ph += 0.04
        self.after(50, self._animate)


class RiskMeter(tk.Canvas):
    """Animated arc-style risk gauge."""
    def __init__(self, parent, width=300, height=170, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=BG_CARD, highlightthickness=0, **kwargs)
        self._current = 0
        self._target  = 0
        self._animating = False

    def set_risk(self, pct):
        self._target = pct
        if not self._animating:
            self._animate_to_target()

    def _animate_to_target(self):
        self._animating = True
        diff = self._target - self._current
        if abs(diff) < 0.5:
            self._current = self._target
            self._draw(self._current)
            self._animating = False
            return
        self._current += diff * 0.12
        self._draw(self._current)
        self.after(16, self._animate_to_target)

    def _draw(self, pct):
        self.delete("all")
        w = self.winfo_width() or 300
        h = self.winfo_height() or 170
        cx = w // 2
        cy = h - 20
        r  = min(cx - 20, h - 30)

        # Track arc
        self.create_arc(cx-r, cy-r, cx+r, cy+r,
                        start=0, extent=180, style='arc',
                        outline=BG_SURFACE, width=18)

        # Coloured fill arc
        if pct >= 70:
            col = ACCENT_RED
        elif pct >= 40:
            col = ACCENT_AMBER
        else:
            col = ACCENT_GREEN

        filled_deg = (pct / 100) * 180
        if filled_deg > 0:
            self.create_arc(cx-r, cy-r, cx+r, cy+r,
                            start=180, extent=filled_deg, style='arc',
                            outline=col, width=18)

        # Needle
        angle_deg = 180 - filled_deg
        angle_rad = math.radians(angle_deg)
        nx = cx + (r - 10) * math.cos(angle_rad)
        ny = cy - (r - 10) * math.sin(angle_rad)
        self.create_line(cx, cy, nx, ny, fill="white", width=3)
        self.create_oval(cx-5, cy-5, cx+5, cy+5, fill=col, outline='')

        # Label
        self.create_text(cx, cy-r//2-10, text=f"{pct:.1f}%",
                         fill=col, font=("Segoe UI", 18, "bold"))
        self.create_text(cx, cy+12, text="RISK LEVEL",
                         fill=TXT_SECONDARY, font=("Segoe UI", 8))

        # Scale labels
        for deg, label in [(180,"0"), (135,"25"), (90,"50"), (45,"75"), (0,"100")]:
            rad = math.radians(deg)
            lx = cx + (r+14) * math.cos(rad)
            ly = cy - (r+14) * math.sin(rad)
            self.create_text(lx, ly, text=label, fill=TXT_MUTED, font=("Courier", 7))


# ── Main App ─────────────────────────────────────────────────────

class LandslideApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.predictor   = LandslidePredictor()
        self.title("⛰  Landslide AI — Disaster Risk Prediction System")
        self.geometry("1300x880")
        self.configure(bg=BG_DARK)
        self.minsize(1100, 760)
        self.resizable(True, True)
        self._build_ui()

    def _build_ui(self):
        self._build_header()
        self._build_notebook()

    # ── Header ─────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=BG_CARD, height=80)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        # left: title
        left = tk.Frame(hdr, bg=BG_CARD)
        left.pack(side=tk.LEFT, padx=24, pady=10)

        title_lbl = tk.Label(left, text="⛰  LANDSLIDE PREDICTION AI",
                             bg=BG_CARD, fg=ACCENT_CYAN,
                             font=("Courier", 18, "bold"))
        title_lbl.pack(anchor='w')
        tk.Label(left, text="Environmental Disaster Risk Assessment  •  Random Forest ML",
                 bg=BG_CARD, fg=TXT_SECONDARY,
                 font=("Segoe UI", 10)).pack(anchor='w')

        # right: status dot + text
        right = tk.Frame(hdr, bg=BG_CARD)
        right.pack(side=tk.RIGHT, padx=24)
        self._status_dot = PulsingDot(right, color=ACCENT_AMBER)
        self._status_dot.pack(side=tk.LEFT)
        self._status_lbl = tk.Label(right, text="IDLE — Awaiting pipeline run",
                                    bg=BG_CARD, fg=TXT_SECONDARY,
                                    font=("Courier", 10))
        self._status_lbl.pack(side=tk.LEFT, padx=6)

        # accent divider line
        tk.Frame(self, bg=ACCENT_CYAN, height=2).pack(fill=tk.X)

    def _set_status(self, text, color=ACCENT_GREEN):
        self._status_lbl.config(text=text, fg=color)
        self._status_dot.set_color(color)

    # ── Notebook ───────────────────────────────────────────────────
    def _build_notebook(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('Dark.TNotebook',
                        background=BG_DARK, borderwidth=0, tabmargins=[0,0,0,0])
        style.configure('Dark.TNotebook.Tab',
                        background=BG_CARD, foreground=TXT_SECONDARY,
                        padding=[20, 10], font=('Segoe UI', 10, 'bold'),
                        borderwidth=0)
        style.map('Dark.TNotebook.Tab',
                  background=[('selected', BG_SURFACE)],
                  foreground=[('selected', ACCENT_CYAN)])

        nb = ttk.Notebook(self, style='Dark.TNotebook')
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.tab_train   = tk.Frame(nb, bg=BG_DARK)
        self.tab_predict = tk.Frame(nb, bg=BG_DARK)
        self.tab_results = tk.Frame(nb, bg=BG_DARK)
        self.tab_about   = tk.Frame(nb, bg=BG_DARK)

        nb.add(self.tab_train,   text="  ⚙  TRAIN  ")
        nb.add(self.tab_predict, text="  🔍  PREDICT  ")
        nb.add(self.tab_results, text="  📊  RESULTS  ")
        nb.add(self.tab_about,   text="  ℹ  ABOUT  ")

        self._build_train_tab()
        self._build_predict_tab()
        self._build_results_tab()
        self._build_about_tab()

    # ── Train Tab ──────────────────────────────────────────────────
    def _build_train_tab(self):
        frm = self.tab_train

        # Top control bar
        ctrl = tk.Frame(frm, bg=BG_CARD, pady=14)
        ctrl.pack(fill=tk.X, padx=12, pady=(12, 6))

        tk.Label(ctrl, text="⚙  TRAINING PIPELINE",
                 bg=BG_CARD, fg=ACCENT_CYAN,
                 font=("Courier", 14, "bold")).grid(row=0, column=0, columnspan=5,
                                                    sticky='w', padx=16, pady=(0,10))

        self.tune_var = tk.BooleanVar(value=False)
        tune_chk = tk.Checkbutton(ctrl, text="Hyperparameter Tuning (GridSearchCV)",
                                  variable=self.tune_var,
                                  bg=BG_CARD, fg=TXT_PRIMARY, selectcolor=BG_DARK,
                                  activebackground=BG_CARD, activeforeground=TXT_PRIMARY,
                                  font=("Segoe UI", 10))
        tune_chk.grid(row=1, column=0, padx=16, sticky='w')

        btn_specs = [
            ("▶  RUN PIPELINE",    self._run_pipeline, ACCENT_BLUE,   160),
            ("📊  DASHBOARD",       self._show_dashboard, "#0891B2",   140),
            ("🗑  CLEAR LOG",       self._clear_log,    "#374151",     120),
        ]
        for i, (txt, cmd, clr, w) in enumerate(btn_specs):
            btn = AnimatedButton(ctrl, txt, cmd, color=clr, width=w, height=36, font_size=9)
            btn.grid(row=1, column=i+1, padx=8, sticky='e')

        # Animated progress bar
        prog_frame = tk.Frame(frm, bg=BG_DARK)
        prog_frame.pack(fill=tk.X, padx=12, pady=2)

        style2 = ttk.Style()
        style2.configure("Neon.Horizontal.TProgressbar",
                          troughcolor=BG_SURFACE, background=ACCENT_BLUE,
                          darkcolor=ACCENT_CYAN, lightcolor=ACCENT_CYAN,
                          bordercolor=BG_SURFACE, thickness=6)
        self.progress = ttk.Progressbar(prog_frame, mode='indeterminate',
                                        length=800, style="Neon.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, padx=0, pady=4)

        # Log area with styled frame
        log_outer = tk.Frame(frm, bg=BORDER_CLR, padx=1, pady=1)
        log_outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 12))

        self.log = scrolledtext.ScrolledText(
            log_outer, bg="#040D1A", fg="#4ADE80",
            font=("Courier New", 10), relief=tk.FLAT,
            insertbackground=ACCENT_GREEN,
            selectbackground=ACCENT_BLUE,
            padx=14, pady=12
        )
        self.log.pack(fill=tk.BOTH, expand=True)
        self._log("  ╔══════════════════════════════════════════════════╗")
        self._log("  ║   LANDSLIDE AI — READY                          ║")
        self._log("  ║   Press ▶ RUN PIPELINE to begin training        ║")
        self._log("  ║                                                  ║")
        self._log("  ║   FIX: Data now uses noise injection so         ║")
        self._log("  ║   accuracy is realistic (≈75–80%), not 100%.   ║")
        self._log("  ╚══════════════════════════════════════════════════╝\n")

    def _log(self, text: str):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.update_idletasks()

    def _clear_log(self):
        self.log.delete('1.0', tk.END)

    def _run_pipeline(self):
        def task():
            self.progress.start(8)
            self._set_status("TRAINING IN PROGRESS...", ACCENT_AMBER)
            try:
                steps = [
                    ("DATA LOADING & INSPECTION", lambda: self.predictor.load_data()),
                    ("PREPROCESSING",              lambda: self.predictor.preprocess()),
                    ("MODEL TRAINING",             lambda: self.predictor.train(tune=self.tune_var.get())),
                    ("EVALUATION & METRICS",       lambda: self.predictor.evaluate()),
                    ("COMPARATIVE MODEL ANALYSIS", lambda: self.predictor.compare_models()),
                ]
                for i, (title, fn) in enumerate(steps, 1):
                    self._log(f"\n  {'━'*52}")
                    self._log(f"  STEP {i}/5 — {title}")
                    self._log(f"  {'━'*52}")
                    self._log(fn())

                self.predictor.save()
                self._log(f"\n  ✔ Model saved → {MODEL_FILE}")

                m = self.predictor.metrics
                plot_confusion_matrix(m.confusion_mat,          show=False)
                plot_roc_curve(m.fpr, m.tpr, m.roc_auc,        show=False)
                plot_feature_importance(m.feature_importances,  show=False)
                plot_train_test_comparison(m.train_accuracy, m.accuracy, show=False)
                plot_model_comparison(self.predictor.comparison,show=False)
                plot_class_distribution(self.predictor.df,      show=False)
                plot_all_metrics_dashboard(self.predictor,      show=False)
                self._log(f"  ✔ All plots saved → {RESULTS_DIR}/\n")

                self.after(0, lambda: self._rebuild_predict_form(self.predictor.feature_names))
                self.after(0, self._populate_metric_cards)
                self._set_status(f"TRAINED ✔  Accuracy: {m.accuracy*100:.1f}%  AUC: {m.roc_auc:.3f}", ACCENT_GREEN)
                messagebox.showinfo("Pipeline Complete",
                    f"✔ Training complete!\n\n"
                    f"Test Accuracy : {m.accuracy*100:.1f}%\n"
                    f"F1 (macro)    : {m.f1:.4f}\n"
                    f"ROC-AUC       : {m.roc_auc:.4f}\n\n"
                    "Note: ~75–80% is realistic for noisy landslide data.\n"
                    "100% would indicate data leakage or trivially easy labels.")
            except Exception as e:
                self._log(f"\n  ✖ ERROR: {e}")
                self._set_status(f"ERROR: {e}", ACCENT_RED)
                messagebox.showerror("Error", str(e))
            finally:
                self.progress.stop()

        threading.Thread(target=task, daemon=True).start()

    def _show_dashboard(self):
        if not self._require_model(): return
        plot_all_metrics_dashboard(self.predictor, show=True)

    # ── Predict Tab ────────────────────────────────────────────────
    def _build_predict_tab(self):
        frm = self.tab_predict

        # ── Header ────────────────────────────────────────────────
        hdr_f = tk.Frame(frm, bg=BG_DARK)
        hdr_f.pack(fill=tk.X, padx=20, pady=(14, 6))
        tk.Label(hdr_f, text="🔍  RISK ASSESSMENT",
                 bg=BG_DARK, fg=ACCENT_CYAN,
                 font=("Courier", 15, "bold")).pack(anchor='w')
        tk.Label(hdr_f,
                 text="Set environmental conditions below, then click  PREDICT  to assess landslide risk.",
                 bg=BG_DARK, fg=TXT_SECONDARY, font=("Segoe UI", 10)).pack(anchor='w')

        # ── Main body: left (scrollable form) + right (gauge panel) ──
        body = tk.Frame(frm, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=6)

        # ── Right panel ───────────────────────────────────────────
        right_panel = tk.Frame(body, bg=BG_CARD, width=310, padx=16, pady=16)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        tk.Label(right_panel, text="⚡  RISK GAUGE",
                 bg=BG_CARD, fg=ACCENT_AMBER,
                 font=("Courier", 10, "bold")).pack(pady=(0, 4))
        self._risk_meter = RiskMeter(right_panel, width=278, height=160)
        self._risk_meter.pack()

        tk.Frame(right_panel, bg=BORDER_CLR, height=1).pack(fill=tk.X, pady=12)

        self.result_label = tk.Label(right_panel,
                                     text="── Awaiting prediction ──",
                                     bg=BG_CARD, fg=TXT_MUTED,
                                     font=("Courier", 11, "bold"),
                                     wraplength=270, justify='center')
        self.result_label.pack(pady=4)

        self.prob_label = tk.Label(right_panel, text="",
                                   bg=BG_CARD, fg=TXT_SECONDARY,
                                   font=("Segoe UI", 9))
        self.prob_label.pack(pady=2)

        tk.Label(right_panel, text="RISK PROBABILITY",
                 bg=BG_CARD, fg=TXT_MUTED, font=("Courier", 8)).pack(pady=(10, 2))
        self.risk_canvas = tk.Canvas(right_panel, height=24, bg=BG_SURFACE,
                                     highlightthickness=0, width=270)
        self.risk_canvas.pack(pady=2)

        tk.Frame(right_panel, bg=BORDER_CLR, height=1).pack(fill=tk.X, pady=12)
        btn_row = tk.Frame(right_panel, bg=BG_CARD)
        btn_row.pack()
        AnimatedButton(btn_row, "🔍  PREDICT", self._predict,
                       color=ACCENT_BLUE, width=130, height=40, font_size=10).pack(side=tk.LEFT, padx=4)
        AnimatedButton(btn_row, "↺  RESET", self._reset_defaults,
                       color="#374151", width=110, height=40, font_size=10).pack(side=tk.LEFT, padx=4)

        # ── Left: scrollable form container ──────────────────────
        left_outer = tk.Frame(body, bg=BG_DARK)
        left_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Canvas + scrollbar for scrolling
        self._predict_canvas = tk.Canvas(left_outer, bg=BG_DARK,
                                         highlightthickness=0)
        self._predict_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _vsb = ttk.Scrollbar(left_outer, orient=tk.VERTICAL,
                             command=self._predict_canvas.yview)
        _vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._predict_canvas.configure(yscrollcommand=_vsb.set)

        # Inner frame lives inside the canvas
        self.predict_form_container = tk.Frame(self._predict_canvas, bg=BG_DARK)
        self._canvas_window = self._predict_canvas.create_window(
            (0, 0), window=self.predict_form_container, anchor='nw')

        def _on_frame_configure(e):
            self._predict_canvas.configure(
                scrollregion=self._predict_canvas.bbox("all"))
        def _on_canvas_configure(e):
            self._predict_canvas.itemconfig(
                self._canvas_window, width=e.width)
        def _on_mousewheel(e):
            self._predict_canvas.yview_scroll(int(-1*(e.delta/120)), "units")

        self.predict_form_container.bind("<Configure>", _on_frame_configure)
        self._predict_canvas.bind("<Configure>", _on_canvas_configure)
        self._predict_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.feature_vars   = {}
        self.feature_scales = {}
        self._form_frame    = None
        self._rebuild_predict_form(list(FEATURE_DEFAULTS.keys()))

    def _rebuild_predict_form(self, features: list):
        if self._form_frame is not None:
            self._form_frame.destroy()

        # Filter out Random_Noise — not meaningful to users
        display_features = [f for f in features if f != 'Random_Noise']

        self._form_frame = tk.Frame(self.predict_form_container,
                                    bg=BG_CARD, padx=20, pady=16)
        self._form_frame.pack(fill=tk.BOTH, expand=True, pady=4,
                              padx=2)

        tk.Label(self._form_frame, text="⚙  ENVIRONMENTAL PARAMETERS",
                 bg=BG_CARD, fg=ACCENT_CYAN,
                 font=("Courier", 12, "bold")).grid(
                     row=0, column=0, columnspan=4, sticky='w', pady=(0, 16))

        # Slider ranges
        SLIDER_RANGES = {
            'Rainfall_mm'        : (10,  350, 1,    float),
            'Slope_Angle'        : (1,   70,  0.5,  float),
            'Soil_Saturation'    : (0.0, 1.0, 0.01, float),
            'Vegetation_Cover'   : (0.0, 1.0, 0.01, float),
            'Earthquake_Activity': (0.0, 9.5, 0.1,  float),
            'Proximity_to_Water' : (0.0, 12,  0.1,  float),
            'Soil_Type_Gravel'   : (0,   1,   1,    int),
            'Soil_Type_Sand'     : (0,   1,   1,    int),
            'Soil_Type_Silt'     : (0,   1,   1,    int),
        }

        self.feature_vars   = {}
        self.feature_scales = {}
        self._entry_widgets = {}

        # Layout: each feature occupies one full row (label row + slider row)
        # so features are stacked vertically, easy to read
        for i, feat in enumerate(display_features):
            row_base = (i * 3) + 1
            icon = FEATURE_ICONS.get(feat, '◆')
            hint = FEATURE_HINTS.get(feat, '')

            # ── Row A: icon + name + range hint + value entry ──────────
            lbl_frame = tk.Frame(self._form_frame, bg=BG_CARD)
            lbl_frame.grid(row=row_base, column=0, columnspan=4,
                           sticky='ew', padx=4, pady=(10, 2))
            lbl_frame.columnconfigure(1, weight=1)

            tk.Label(lbl_frame,
                     text=f"{icon}  {feat.replace('_', ' ')}",
                     bg=BG_CARD, fg=TXT_PRIMARY,
                     font=("Segoe UI", 11, "bold")).grid(
                         row=0, column=0, sticky='w')

            tk.Label(lbl_frame,
                     text=f"   {hint}",
                     bg=BG_CARD, fg=TXT_MUTED,
                     font=("Segoe UI", 9)).grid(
                         row=0, column=1, sticky='w', padx=(4, 0))

            # Value entry box (shows current slider value; editable)
            var = tk.DoubleVar(value=float(FEATURE_DEFAULTS.get(feat, '0')))
            self.feature_vars[feat] = var

            entry_var = tk.StringVar(value=self._fmt_val(feat, var.get()))
            entry = tk.Entry(lbl_frame,
                             textvariable=entry_var,
                             bg=BG_INPUT, fg=ACCENT_CYAN,
                             insertbackground=ACCENT_CYAN,
                             font=("Courier New", 10, "bold"),
                             width=10, relief=tk.FLAT,
                             justify='center',
                             highlightthickness=1,
                             highlightbackground=BORDER_CLR,
                             highlightcolor=ACCENT_BLUE)
            entry.grid(row=0, column=2, sticky='e', padx=(8, 0))
            self._entry_widgets[feat] = entry_var

            # ── Row B: slider spanning full width ──────────────────────
            sl_range = SLIDER_RANGES.get(feat, (0, 1, 0.01, float))

            def _slider_moved(v, f=feat, ev=entry_var, dv=var):
                ev.set(self._fmt_val(f, float(v)))

            def _entry_committed(event, f=feat, ev=entry_var, dv=var, sl_r=sl_range):
                raw = ev.get().strip()
                # strip units so user can type "120 mm" or just "120"
                for suffix in [' mm', ' °', ' km', ' M', 'M ', 'YES', 'NO']:
                    raw = raw.replace(suffix, '').strip()
                try:
                    val = float(raw)
                    val = max(sl_r[0], min(sl_r[1], val))
                    dv.set(val)
                    ev.set(self._fmt_val(f, val))
                except ValueError:
                    ev.set(self._fmt_val(f, dv.get()))

            sl = tk.Scale(self._form_frame,
                          from_=sl_range[0], to=sl_range[1],
                          resolution=sl_range[2], orient=tk.HORIZONTAL,
                          variable=var, length=0,
                          bg=BG_SURFACE, fg=ACCENT_CYAN,
                          troughcolor=BG_INPUT, activebackground=ACCENT_BLUE,
                          highlightthickness=0, showvalue=False, bd=0,
                          command=_slider_moved)
            sl.grid(row=row_base + 1, column=0, columnspan=4,
                    padx=4, pady=(0, 2), sticky='ew')
            self.feature_scales[feat] = sl

            entry.bind("<Return>",    _entry_committed)
            entry.bind("<FocusOut>",  _entry_committed)

            # ── Row C: thin separator ──────────────────────────────────
            if i < len(display_features) - 1:
                tk.Frame(self._form_frame, bg=BORDER_CLR, height=1).grid(
                    row=row_base + 2, column=0, columnspan=4,
                    sticky='ew', padx=4, pady=(4, 0))

        self._form_frame.columnconfigure(0, weight=0)
        self._form_frame.columnconfigure(1, weight=1)
        self._form_frame.columnconfigure(2, weight=0)

    def _fmt_val(self, feat, val):
        if feat in ('Soil_Type_Gravel','Soil_Type_Sand','Soil_Type_Silt'):
            return "YES" if int(round(val)) == 1 else "NO"
        elif feat == 'Rainfall_mm':
            return f"{val:.0f} mm"
        elif feat == 'Slope_Angle':
            return f"{val:.1f} °"
        elif feat in ('Soil_Saturation','Vegetation_Cover'):
            return f"{val:.2f}"
        elif feat == 'Earthquake_Activity':
            return f"M {val:.1f}"
        elif feat == 'Proximity_to_Water':
            return f"{val:.1f} km"
        return f"{val:.2f}"

    def _reset_defaults(self):
        for feat, var in self.feature_vars.items():
            val = float(FEATURE_DEFAULTS.get(feat, '0'))
            var.set(val)
            if hasattr(self, '_entry_widgets') and feat in self._entry_widgets:
                self._entry_widgets[feat].set(self._fmt_val(feat, val))
        self._risk_meter.set_risk(0)
        self.result_label.config(text="── Awaiting prediction ──", fg=TXT_MUTED)
        self.prob_label.config(text="")
        self.risk_canvas.delete('all')

    def _predict(self):
        if not self._require_model(): return
        try:
            vals = {f: float(v.get()) for f, v in self.feature_vars.items()}
            # Auto-fill Random_Noise if not shown (it's uninformative anyway)
            if 'Random_Noise' in self.predictor.feature_names and 'Random_Noise' not in vals:
                vals['Random_Noise'] = 0.5
        except (ValueError, tk.TclError):
            messagebox.showerror("Input Error", "All fields must be valid numbers.")
            return

        res  = self.predictor.predict_sample(vals)
        clr  = ACCENT_RED if res['risk_pct'] >= 70 else ACCENT_AMBER if res['risk_pct'] >= 40 else ACCENT_GREEN
        icon = "🔴" if res['risk_pct'] >= 70 else "🟡" if res['risk_pct'] >= 40 else "🟢"

        self._risk_meter.set_risk(res['risk_pct'])
        label_txt = (f"{icon}  {'LANDSLIDE LIKELY' if res['prediction']==1 else 'NO LANDSLIDE'}\n"
                     f"{res['risk_level']}")
        self.result_label.config(text=label_txt, fg=clr, font=("Courier", 11, "bold"))
        self.prob_label.config(
            text=f"P(No LS)={res['prob_no']*100:.1f}%   P(LS)={res['prob_yes']*100:.1f}%",
            fg=TXT_SECONDARY)

        # Animated risk bar
        self.risk_canvas.delete('all')
        w = 290
        self.risk_canvas.create_rectangle(0, 3, w, 19, fill=BG_SURFACE, outline='')
        fill_w = int(w * res['risk_pct'] / 100)
        if fill_w > 0:
            self.risk_canvas.create_rectangle(0, 3, fill_w, 19, fill=clr, outline='')
        self.risk_canvas.create_text(w//2, 11, text=f"{res['risk_pct']:.1f}%",
                                     fill='white', font=("Segoe UI", 9, "bold"))

    # ── Results Tab ────────────────────────────────────────────────
    def _build_results_tab(self):
        frm = self.tab_results

        # Header
        hdr = tk.Frame(frm, bg=BG_DARK)
        hdr.pack(fill=tk.X, padx=20, pady=(16, 8))
        tk.Label(hdr, text="📊  EVALUATION RESULTS & PLOTS",
                 bg=BG_DARK, fg=ACCENT_CYAN,
                 font=("Courier", 15, "bold")).pack(anchor='w')
        tk.Label(hdr, text=f"All plots auto-saved to  '{RESULTS_DIR}/'  after training  •  Train first to enable plots",
                 bg=BG_DARK, fg=TXT_SECONDARY, font=("Segoe UI", 9)).pack(anchor='w')

        # Metric summary card row (populated after training)
        self._metric_cards_frame = tk.Frame(frm, bg=BG_DARK)
        self._metric_cards_frame.pack(fill=tk.X, padx=16, pady=(4, 6))

        # divider
        tk.Frame(frm, bg=BORDER_CLR, height=1).pack(fill=tk.X, padx=16)

        # Plot buttons grid
        tk.Label(frm, text="VISUALISATION PANELS",
                 bg=BG_DARK, fg=TXT_SECONDARY,
                 font=("Courier", 9, "bold")).pack(anchor='w', padx=20, pady=(10, 4))

        grid_frame = tk.Frame(frm, bg=BG_DARK)
        grid_frame.pack(padx=16, pady=4, fill=tk.BOTH, expand=True)

        plots = [
            ("📉  Confusion\nMatrix",    self._show_confusion_matrix, ACCENT_BLUE,   "#8B5CF6"),
            ("📈  ROC\nCurve",           self._show_roc,               "#0891B2",     "#06B6D4"),
            ("📊  Feature\nImportances", self._show_feature_imp,       "#059669",     "#10B981"),
            ("⚖️  Train vs\nTest",        self._show_train_test,        "#D97706",     "#F59E0B"),
            ("🔁  Model\nComparison",    self._show_model_comp,        "#7C3AED",     "#A855F7"),
            ("🥧  Class\nDistribution",  self._show_class_dist,        "#DC2626",     "#EF4444"),
            ("🗂  Full\nDashboard",      self._show_full_dashboard,    "#374151",     "#6B7280"),
        ]

        for i, (label, cmd, clr, hover_clr) in enumerate(plots):
            r, c = divmod(i, 4)
            card_outer = tk.Frame(grid_frame, bg=BORDER_CLR, padx=1, pady=1)
            card_outer.grid(row=r, column=c, padx=10, pady=10, sticky='nsew')
            grid_frame.columnconfigure(c, weight=1)
            grid_frame.rowconfigure(r, weight=1)

            card = tk.Frame(card_outer, bg=BG_CARD, padx=16, pady=16)
            card.pack(fill=tk.BOTH, expand=True)

            tk.Label(card, text=label, bg=BG_CARD, fg=TXT_PRIMARY,
                     font=("Segoe UI", 11, "bold"), justify='center').pack(pady=(0, 12))

            btn = AnimatedButton(card, "▶  SHOW", cmd, color=clr, width=130, height=36, font_size=9)
            btn.pack()

    def _populate_metric_cards(self):
        """Show live metric summary cards in the Results tab after training."""
        for w in self._metric_cards_frame.winfo_children():
            w.destroy()
        m = self.predictor.metrics
        cards = [
            ("🎯  Accuracy",   f"{m.accuracy*100:.1f}%",  ACCENT_BLUE),
            ("⚡  F1-Score",   f"{m.f1:.4f}",             ACCENT_GREEN),
            ("📈  ROC-AUC",    f"{m.roc_auc:.4f}",        ACCENT_CYAN),
            ("🔬  Precision",  f"{m.precision:.4f}",       ACCENT_AMBER),
            ("🔍  Recall",     f"{m.recall:.4f}",          ACCENT_PURPLE),
            ("📊  CV Mean",    f"{m.cv_mean:.4f}",         "#EC4899"),
        ]
        for title, value, clr in cards:
            outer = tk.Frame(self._metric_cards_frame, bg=BORDER_CLR, padx=1, pady=1)
            outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=2)
            inner = tk.Frame(outer, bg=BG_CARD, padx=12, pady=10)
            inner.pack(fill=tk.BOTH, expand=True)
            tk.Label(inner, text=title, bg=BG_CARD, fg=TXT_SECONDARY,
                     font=("Segoe UI", 8, "bold")).pack()
            tk.Label(inner, text=value, bg=BG_CARD, fg=clr,
                     font=("Courier", 14, "bold")).pack()

    # ── Results helpers ────────────────────────────────────────────
    def _require_model(self):
        if self.predictor.model is None:
            messagebox.showwarning("Not Trained", "Please run the training pipeline first.")
            return False
        return True

    def _show_confusion_matrix(self):
        if not self._require_model(): return
        plot_confusion_matrix(self.predictor.metrics.confusion_mat, show=True)

    def _show_roc(self):
        if not self._require_model(): return
        m = self.predictor.metrics
        plot_roc_curve(m.fpr, m.tpr, m.roc_auc, show=True)

    def _show_feature_imp(self):
        if not self._require_model(): return
        plot_feature_importance(self.predictor.metrics.feature_importances, show=True)

    def _show_train_test(self):
        if not self._require_model(): return
        m = self.predictor.metrics
        plot_train_test_comparison(m.train_accuracy, m.accuracy, show=True)

    def _show_model_comp(self):
        if not self._require_model(): return
        if not self.predictor.comparison:
            messagebox.showwarning("Not Available", "Run the pipeline first.")
            return
        plot_model_comparison(self.predictor.comparison, show=True)

    def _show_class_dist(self):
        if self.predictor.df is None:
            messagebox.showwarning("Not Loaded", "Run the pipeline first.")
            return
        plot_class_distribution(self.predictor.df, show=True)

    def _show_full_dashboard(self):
        if not self._require_model(): return
        plot_all_metrics_dashboard(self.predictor, show=True)

    # ── About Tab ──────────────────────────────────────────────────
    def _build_about_tab(self):
        frm = self.tab_about

        # ── Scrollable canvas wrapper ──────────────────────────────
        outer_canvas = tk.Canvas(frm, bg=BG_DARK, highlightthickness=0)
        outer_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=outer_canvas.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        outer_canvas.configure(yscrollcommand=vsb.set)

        scroll_frame = tk.Frame(outer_canvas, bg=BG_DARK)
        win_id = outer_canvas.create_window((0, 0), window=scroll_frame, anchor='nw')

        def _cfg(e): outer_canvas.configure(scrollregion=outer_canvas.bbox("all"))
        def _resize(e): outer_canvas.itemconfig(win_id, width=e.width)
        def _wheel(e): outer_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        scroll_frame.bind("<Configure>", _cfg)
        outer_canvas.bind("<Configure>", _resize)
        outer_canvas.bind_all("<MouseWheel>", _wheel)

        # ── Helper to add sections ─────────────────────────────────
        def section(title, icon="▸"):
            hdr = tk.Frame(scroll_frame, bg=BG_SURFACE, padx=18, pady=10)
            hdr.pack(fill=tk.X, padx=20, pady=(18, 0))
            tk.Label(hdr, text=f"{icon}  {title}",
                     bg=BG_SURFACE, fg=ACCENT_CYAN,
                     font=("Courier", 12, "bold")).pack(anchor='w')
            body = tk.Frame(scroll_frame, bg=BG_CARD, padx=22, pady=14)
            body.pack(fill=tk.X, padx=20, pady=0)
            return body

        def para(parent, text, fg=TXT_PRIMARY, font=("Segoe UI", 10)):
            tk.Label(parent, text=text, bg=BG_CARD, fg=fg,
                     font=font, justify='left',
                     wraplength=900, anchor='w').pack(anchor='w', pady=3)

        def kv(parent, key, value, key_fg=ACCENT_AMBER):
            row = tk.Frame(parent, bg=BG_CARD)
            row.pack(anchor='w', fill=tk.X, pady=2)
            tk.Label(row, text=f"  {key}:", bg=BG_CARD, fg=key_fg,
                     font=("Segoe UI", 10, "bold"), width=26,
                     anchor='w').pack(side=tk.LEFT)
            tk.Label(row, text=value, bg=BG_CARD, fg=TXT_PRIMARY,
                     font=("Segoe UI", 10), anchor='w').pack(side=tk.LEFT)

        def badge_row(parent, items):
            row = tk.Frame(parent, bg=BG_CARD)
            row.pack(anchor='w', pady=4)
            for label, fg in items:
                tk.Label(row, text=f"  {label}  ",
                         bg=BG_SURFACE, fg=fg,
                         font=("Courier", 9, "bold"),
                         relief=tk.FLAT, padx=6, pady=3).pack(
                             side=tk.LEFT, padx=(0, 6))

        # ── Title banner ──────────────────────────────────────────
        banner = tk.Frame(scroll_frame, bg=BG_CARD, padx=24, pady=20)
        banner.pack(fill=tk.X, padx=20, pady=(16, 0))
        tk.Label(banner, text="⛰  LANDSLIDE PREDICTION AI",
                 bg=BG_CARD, fg=ACCENT_CYAN,
                 font=("Courier", 20, "bold")).pack(anchor='w')
        tk.Label(banner, text="Environmental Disaster Risk Assessment System  ·  Machine Learning Edition",
                 bg=BG_CARD, fg=TXT_SECONDARY,
                 font=("Segoe UI", 11)).pack(anchor='w', pady=(4, 0))
        badge_row(banner, [
            ("🌲  Random Forest",   ACCENT_GREEN),
            ("📊  Synthetic Dataset", ACCENT_CYAN),
            ("🎯  ~77% Accuracy",   ACCENT_AMBER),
            ("📈  AUC ~0.82",       ACCENT_PURPLE),
        ])

        # ── What does this system do? ──────────────────────────────
        b = section("WHAT DOES THIS SYSTEM DO?", "🎯")
        para(b, "This application uses a trained Machine Learning model to assess the probability of a landslide "
                "occurring given a set of measurable environmental conditions. By adjusting parameters like "
                "rainfall, slope angle, and soil saturation in the PREDICT tab, you receive an instant risk "
                "classification — Low, Moderate, or High — along with the model's confidence probability.")
        para(b, "It is designed for educational and research purposes in environmental AI and disaster risk management.")

        # ── Input features ────────────────────────────────────────
        b = section("INPUT FEATURES (What the model reads)", "📥")
        features_info = [
            ("🌧  Rainfall (mm)",          "50–300 mm",   "Total rainfall; heavy rain saturates soil and triggers landslides."),
            ("⛰  Slope Angle (°)",        "5–65 °",      "Steeper slopes have higher shear stress, increasing failure risk."),
            ("💧  Soil Saturation",        "0.0–1.0",     "Fraction of pore space filled with water; high = weakened cohesion."),
            ("🌿  Vegetation Cover",       "0.0–1.0",     "Roots stabilise slopes; dense cover (high value) reduces risk."),
            ("📳  Earthquake Activity",    "M 0–9",       "Seismic shaking can dislodge unstable material instantly."),
            ("🏞  Proximity to Water",     "0–10 km",     "Nearby rivers/streams can undercut slopes and raise water tables."),
            ("🪨  Soil Type: Gravel",      "Yes / No",    "Gravel drains quickly — generally lower landslide susceptibility."),
            ("🏜  Soil Type: Sand",        "Yes / No",    "Sandy soils can liquefy under saturation; moderate risk."),
            ("🌍  Soil Type: Silt",        "Yes / No",    "Fine-grained, high water retention — highest landslide risk."),
        ]
        for icon_name, rng, desc in features_info:
            row = tk.Frame(b, bg=BG_SURFACE, padx=12, pady=8)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=icon_name, bg=BG_SURFACE, fg=ACCENT_CYAN,
                     font=("Segoe UI", 10, "bold"), width=24, anchor='w').pack(side=tk.LEFT)
            tk.Label(row, text=rng, bg=BG_SURFACE, fg=ACCENT_AMBER,
                     font=("Courier", 9), width=10, anchor='w').pack(side=tk.LEFT)
            tk.Label(row, text=desc, bg=BG_SURFACE, fg=TXT_SECONDARY,
                     font=("Segoe UI", 9), anchor='w', wraplength=700).pack(side=tk.LEFT, padx=(8,0))

        # ── Dataset ───────────────────────────────────────────────
        b = section("DATASET", "🗂")
        kv(b, "Type",           "Synthetic, noise-injected (generated fresh each run)")
        kv(b, "Samples",        "1,200 rows")
        kv(b, "Target column",  "Landslide  (0 = No,  1 = Yes)")
        kv(b, "Class balance",  "Roughly balanced via logistic probability model + label noise")
        kv(b, "Train / Test",   "80% training  /  20% test  (stratified split)")
        kv(b, "Preprocessing",  "Duplicate removal → Median imputation → StandardScaler (SVM/LR only)")
        para(b, "A realistic noise injection pipeline is used so that class boundaries overlap naturally, "
                "preventing the trivially perfect separability that would produce 100% accuracy and mask real-world complexity.",
             fg=TXT_SECONDARY, font=("Segoe UI", 9))

        # ── Model ─────────────────────────────────────────────────
        b = section("MODEL — Random Forest Classifier", "🌲")
        kv(b, "Algorithm",        "Random Forest (scikit-learn)")
        kv(b, "n_estimators",     "100 decision trees")
        kv(b, "max_depth",        "10  (prevents overfitting on noisy data)")
        kv(b, "min_samples_split","10  (extra regularisation)")
        kv(b, "min_samples_leaf", "5   (smooths decision boundaries)")
        kv(b, "class_weight",     "balanced  (handles class imbalance automatically)")
        kv(b, "random_state",     "42  (reproducible results)")
        para(b, "A Random Forest trains many decorrelated decision trees and aggregates their votes. "
                "This ensemble approach reduces variance compared to a single decision tree and generally "
                "outperforms linear methods on tabular environmental data with non-linear interactions.",
             fg=TXT_SECONDARY, font=("Segoe UI", 9))

        # ── Performance ───────────────────────────────────────────
        b = section("MODEL PERFORMANCE", "📈")
        kv(b, "Test Accuracy",   "~77%   (realistic for noisy real-world-like data)")
        kv(b, "F1-Score (macro)","~0.75–0.80")
        kv(b, "ROC-AUC",         "~0.80–0.85")
        kv(b, "5-Fold CV",       "Confirms generalisation — low variance across folds")
        kv(b, "Overfitting gap", "Typically < 10%  (train accuracy vs test accuracy)")
        para(b, "An accuracy of 75–80% is deliberately realistic. Perfect accuracy (100%) on this type of "
                "dataset would indicate data leakage or trivially separable labels — not a better model.",
             fg=TXT_SECONDARY, font=("Segoe UI", 9))

        # ── Evaluation metrics ────────────────────────────────────
        b = section("EVALUATION METRICS EXPLAINED", "🔬")
        metrics_info = [
            ("Accuracy",         "% of predictions that are correct overall."),
            ("F1-Score (macro)", "Harmonic mean of precision & recall, averaged across classes."),
            ("Precision",        "Of all predicted landslides, how many were real?  (reduces false alarms)"),
            ("Recall",           "Of all real landslides, how many did we catch?  (reduces missed events)"),
            ("ROC-AUC",          "Model's ability to distinguish classes across all thresholds. 1.0 = perfect."),
            ("Confusion Matrix", "Grid showing True Positives, True Negatives, False Positives, False Negatives."),
            ("k-Fold CV",        "Dataset split into k parts; model trained k times to test stability."),
        ]
        for metric, desc in metrics_info:
            row = tk.Frame(b, bg=BG_SURFACE, padx=12, pady=7)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=metric, bg=BG_SURFACE, fg=ACCENT_GREEN,
                     font=("Courier", 10, "bold"), width=22, anchor='w').pack(side=tk.LEFT)
            tk.Label(row, text=desc, bg=BG_SURFACE, fg=TXT_SECONDARY,
                     font=("Segoe UI", 9), anchor='w', wraplength=700).pack(side=tk.LEFT, padx=(6,0))

        # ── Model comparison ──────────────────────────────────────
        b = section("MODEL COMPARISON", "🏆")
        para(b, "The system trains four classifiers side-by-side so you can compare their performance:")
        comp_info = [
            ("Random Forest",      "Main model. Best AUC. Handles non-linearities & feature interactions well."),
            ("Decision Tree",      "Single-tree baseline. Interpretable but prone to overfitting."),
            ("Logistic Regression","Linear baseline. Fast and robust but assumes linear decision boundary."),
            ("SVM",                "Kernel-based. Good on small datasets; slower on large ones."),
        ]
        for name, desc in comp_info:
            row = tk.Frame(b, bg=BG_SURFACE, padx=12, pady=7)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=f"  {name}", bg=BG_SURFACE, fg=ACCENT_AMBER,
                     font=("Segoe UI", 10, "bold"), width=24, anchor='w').pack(side=tk.LEFT)
            tk.Label(row, text=desc, bg=BG_SURFACE, fg=TXT_SECONDARY,
                     font=("Segoe UI", 9), anchor='w', wraplength=700).pack(side=tk.LEFT, padx=(6,0))

        # ── Risk levels ───────────────────────────────────────────
        b = section("RISK CLASSIFICATION THRESHOLDS", "⚠")
        risk_info = [
            ("🟢  LOW RISK",      "< 40% probability",   "Conditions are relatively safe.",                         ACCENT_GREEN),
            ("🟡  MODERATE RISK", "40% – 70% probability","Caution advised; monitor conditions closely.",            ACCENT_AMBER),
            ("🔴  HIGH RISK",     "> 70% probability",   "Danger — immediate assessment or evacuation may be needed.", ACCENT_RED),
        ]
        for lvl, thresh, advice, clr in risk_info:
            row = tk.Frame(b, bg=BG_SURFACE, padx=12, pady=8)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=lvl,    bg=BG_SURFACE, fg=clr,
                     font=("Courier", 10, "bold"), width=20, anchor='w').pack(side=tk.LEFT)
            tk.Label(row, text=thresh, bg=BG_SURFACE, fg=TXT_SECONDARY,
                     font=("Segoe UI", 9), width=22, anchor='w').pack(side=tk.LEFT)
            tk.Label(row, text=advice, bg=BG_SURFACE, fg=TXT_PRIMARY,
                     font=("Segoe UI", 9), anchor='w', wraplength=600).pack(side=tk.LEFT, padx=(8,0))

        # ── How to use ────────────────────────────────────────────
        b = section("HOW TO USE THIS APPLICATION", "📖")
        steps = [
            ("1.  TRAIN tab",    "Click ▶ RUN PIPELINE to generate data, train the model, and evaluate it."),
            ("2.  PREDICT tab",  "Adjust the sliders (or type values in the boxes), then click 🔍 PREDICT."),
            ("3.  RESULTS tab",  "View confusion matrix, ROC curve, feature importances, and model comparison."),
            ("4.  About tab",    "You're here! Read model documentation and interpret the outputs."),
        ]
        for step, desc in steps:
            row = tk.Frame(b, bg=BG_SURFACE, padx=12, pady=8)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=step, bg=BG_SURFACE, fg=ACCENT_CYAN,
                     font=("Segoe UI", 10, "bold"), width=18, anchor='w').pack(side=tk.LEFT)
            tk.Label(row, text=desc, bg=BG_SURFACE, fg=TXT_SECONDARY,
                     font=("Segoe UI", 9), anchor='w', wraplength=700).pack(side=tk.LEFT, padx=(8,0))

        # ── Footer ────────────────────────────────────────────────
        footer = tk.Frame(scroll_frame, bg=BG_DARK, pady=20)
        footer.pack(fill=tk.X, padx=20)
        tk.Label(footer,
                 text="⚠  This system is for educational / research use only. "
                      "Do not use as the sole basis for emergency decisions.",
                 bg=BG_DARK, fg=TXT_MUTED,
                 font=("Segoe UI", 9, "italic")).pack(anchor='w')


# ══════════════════════════════════════════════════════════════════
#  CLI FALLBACK
# ══════════════════════════════════════════════════════════════════

def run_cli():
    matplotlib.use('Agg')
    print("\n" + "═"*62)
    print("  LANDSLIDE PREDICTION SYSTEM — CLI MODE")
    print("═"*62)

    p = LandslidePredictor()
    print("\n[1/5] Loading data...")
    print(p.load_data())
    print("\n[2/5] Preprocessing...")
    print(p.preprocess())
    print("\n[3/5] Training model...")
    print(p.train(tune='--tune' in sys.argv))
    print("\n[4/5] Evaluating...")
    print(p.evaluate())
    print("\n[5/5] Comparing models...")
    print(p.compare_models())

    p.save()
    m = p.metrics
    plot_confusion_matrix(m.confusion_mat,         show=False)
    plot_roc_curve(m.fpr, m.tpr, m.roc_auc,       show=False)
    plot_feature_importance(m.feature_importances, show=False)
    plot_train_test_comparison(m.train_accuracy, m.accuracy, show=False)
    plot_model_comparison(p.comparison,            show=False)
    plot_class_distribution(p.df,                  show=False)
    plot_all_metrics_dashboard(p,                  show=False)
    print(f"\n  ✔ All plots saved to '{RESULTS_DIR}/'")
    print("═"*62)


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if '--cli' in sys.argv or '--no-gui' in sys.argv:
        run_cli()
    else:
        try:
            app = LandslideApp()
            app.mainloop()
        except tk.TclError:
            print("  ℹ  No display detected — switching to CLI mode.")
            run_cli()