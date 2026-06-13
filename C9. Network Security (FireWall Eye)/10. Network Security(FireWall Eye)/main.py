import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import*
from tkinter import filedialog, scrolledtext, messagebox

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from joblib import load
from PIL import Image, ImageTk
import pymysql

# -------------------- CONSTANTS & GLOBALS --------------------
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data state
df = None            # raw dataframe
X = None             # features (encoded)
y = None             # target (encoded ints)
label_encoders = {}  # encoders for object columns (including 'Action' target if object)
X_train = X_test = y_train = y_test = None

# -------------------- UI HELPERS --------------------
def log_output(msg: str):
    output_box.insert(tk.END, msg + "\n")
    output_box.see(tk.END)

def require(condition, msg):
    if not condition:
        messagebox.showerror("Error", msg)
        return False
    return True

# -------------------- WORKFLOW STEPS --------------------
def upload_dataset():
    """Upload CSV and preview."""
    global df
    path = filedialog.askopenfilename(
        title="Select CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not path:
        return
    try:
        df = pd.read_csv(path)
    except Exception as e:
        messagebox.showerror("Read Error", f"Could not read CSV:\n{e}")
        return

    log_output(f"Dataset loaded: {path}")
    log_output(f"Shape: {df.shape}")
    # print first rows safely
    with pd.option_context('display.max_columns', None, 'display.width', 120):
        log_output("Head:")
        log_output(df.head().to_string())

def preprocess_data_train():
    
    global df, X, y, label_encoders

    if not require(df is not None, "Upload dataset first."):
        return
    if not require('Action' in df.columns, "Column 'Action' not found in dataset."):
        return

    log_output("Preprocessing started...")
    df_clean = df.loc[:, ~df.columns.str.contains(r'^Unnamed', na=False)].copy()

    # Fit encoders on object columns (including Action if it is object)
    label_encoders = {}
    for col in df_clean.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
        log_output(f"Encoded column '{col}' with classes: {list(le.classes_)}")

    # Fill numeric NaNs
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    na_before = df_clean[num_cols].isna().sum().sum()
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
    na_after = df_clean[num_cols].isna().sum().sum()
    log_output(f"Filled numeric NaNs: before={na_before}, after={na_after}")

    # Split features/target
    if not require('Action' in df_clean.columns, "Column 'Action' missing after preprocessing."):
        return
    X = df_clean.drop(columns=['Action'])
    y = df_clean['Action'].astype(int)  # already encoded to ints

    log_output(f"Features ready with shape {X.shape}")
    log_output(f"Target ready with shape {y.shape}")

def generate_eda():
    """Single popup figure with all EDA plots, text updates to output box."""
    global df, X, y, label_encoders
    if not require(X is not None and y is not None, "Run Data Preprocessing first."):
        return

    sns.set(style="whitegrid")

    # Build labels for display if we have an encoder for Action
    display_labels = None
    if 'Action' in label_encoders:
        display_labels = list(label_encoders['Action'].classes_)

    plt.figure(figsize=(18, 10))

    # 1. Action distribution
    plt.subplot(2, 3, 1)
    # y is encoded ints; map to labels if available
    y_display = y.map(lambda v: display_labels[v]) if display_labels else y
    sns.countplot(x=y_display, palette='viridis')
    plt.title("Distribution of Action")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.xticks(rotation=20)

    # Safely pick expected columns if present
    def safe_col(name):
        return name if name in X.columns else None

    col_bytes = safe_col('Bytes')
    col_packets = safe_col('Packets')
    col_elapsed = safe_col('Elapsed Time (sec)')
    col_bytes_sent = safe_col('Bytes Sent')
    col_bytes_recv = safe_col('Bytes Received')

    # 2. Box: Bytes vs Action
    plt.subplot(2, 3, 2)
    if col_bytes:
        sns.boxplot(x=y_display, y=X[col_bytes], palette='Set2')
        plt.title("Bytes vs Action")
        plt.xlabel("Action"); plt.ylabel("Bytes")
    else:
        plt.text(0.5, 0.5, "'Bytes' not found", ha='center'); plt.axis('off')

    # 3. Violin: Packets vs Action
    plt.subplot(2, 3, 3)
    if col_packets:
        sns.violinplot(x=y_display, y=X[col_packets], palette='coolwarm')
        plt.title("Packets vs Action")
        plt.xlabel("Action"); plt.ylabel("Packets")
    else:
        plt.text(0.5, 0.5, "'Packets' not found", ha='center'); plt.axis('off')

    # 4. Box: Elapsed Time vs Action
    plt.subplot(2, 3, 4)
    if col_elapsed:
        sns.boxplot(x=y_display, y=X[col_elapsed], palette='Set1')
        plt.title("Elapsed Time vs Action")
        plt.xlabel("Action"); plt.ylabel("Elapsed Time (sec)")
    else:
        plt.text(0.5, 0.5, "'Elapsed Time (sec)' not found", ha='center'); plt.axis('off')

    # 5. Scatter: Bytes Sent vs Bytes Received
    plt.subplot(2, 3, 5)
    if col_bytes_sent and col_bytes_recv:
        sns.scatterplot(x=X[col_bytes_sent], y=X[col_bytes_recv], hue=y_display, alpha=0.6, legend=False)
        plt.title("Bytes Sent vs Bytes Received")
        plt.xlabel("Bytes Sent"); plt.ylabel("Bytes Received")
    else:
        plt.text(0.5, 0.5, "'Bytes Sent'/'Bytes Received' not found", ha='center'); plt.axis('off')

    # 6. Correlation heatmap (numeric only + encoded Action)
    plt.subplot(2, 3, 6)
    df_corr = X.copy()
    df_corr['Action'] = y.values  # numeric
    # Older pandas: avoid numeric_only kw; select only numeric columns explicitly
    corr = df_corr.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', cbar=True)
    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.show()
    log_output("EDA generated (one popup).")

def split_data_smote():
    
    global X, y, X_train, X_test, y_train, y_test
    if not require(X is not None and y is not None, "Run Data Preprocessing first."):
        return

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
    )

    log_output(f"SMOTE applied. Class counts after: {np.bincount(y_res)}")
    log_output(f"Data split done. Train: {X_train.shape}, Test: {X_test.shape}")

# -------------------- MODEL EVALUATION --------------------
def evaluate_loaded_model(algo_name: str, model):
    """Compute metrics, print to output, show one combined popup (CM + ROC if available)."""
    global X_test, y_test, label_encoders
    if not require(X_test is not None and y_test is not None, "Run Data Splitting first."):
        return

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        messagebox.showerror("Model Error", f"Prediction failed:\n{e}")
        return

    # Scores
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    # Prepare display labels for reports/plots
    if 'Action' in label_encoders:
        classes = list(label_encoders['Action'].classes_)
    else:
        # fallback: sorted unique ints as strings
        classes = [str(c) for c in sorted(np.unique(np.concatenate([y_test, y_pred])))]

    log_output(f"=== {algo_name} ===")
    log_output(f"Accuracy:   {acc:.2f}%")
    log_output(f"Precision:  {prec:.2f}%")
    log_output(f"Recall:     {rec:.2f}%")
    log_output(f"F1-Score:   {f1:.2f}%")
    try:
        log_output("Classification Report:\n" + classification_report(y_test, y_pred, target_names=classes, zero_division=0))
    except ValueError:
        # In case of class mismatch
        log_output("Classification Report:\n" + classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Combined popup: CM + ROC
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CM heatmap
    sns.heatmap(cm, annot=True, fmt="g", cmap="viridis",
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title(f"{algo_name} - Confusion Matrix")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    # fix cut-off issue
    axes[0].set_ylim(len(cm), 0)

    # ROC
    y_score = None
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_test)  # shape (n_samples, n_classes)
        except Exception:
            y_score = None

    if y_score is not None:
        # Build consistent class order 0..(n-1)
        n_classes = y_score.shape[1]
        # binarize using numeric labels 0..n-1 (y_test is encoded ints)
        y_bin = label_binarize(y_test, classes=list(range(n_classes)))
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                axes[1].plot(fpr, tpr, label=f"{classes[i]} (AUC={auc(fpr, tpr):.2f})")
            except Exception:
                pass
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_title(f"{algo_name} - ROC")
        axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
        axes[1].legend(loc="lower right")
    else:
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, "ROC not available for this model", ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.show()

def run_algorithm(algo_name: str, model_filename: str):
    """Load trained model from models/ and evaluate."""
    path = os.path.join(MODEL_DIR, model_filename)
    if not os.path.exists(path):
        messagebox.showerror("Model Missing", f"'{model_filename}' not found in '{MODEL_DIR}'.")
        return
    try:
        model = load(path)
    except Exception as e:
        messagebox.showerror("Load Error", f"Could not load model:\n{e}")
        return

    evaluate_loaded_model(algo_name, model)

# -------------------- PREDICTION --------------------
def predict_on_test():
    """User selects test CSV; we preprocess with the SAME encoders and append predictions."""
    global label_encoders, X, y
    if not require(label_encoders and len(label_encoders) > 0, "Run Data Preprocessing on training data first (to fit encoders)."):
        return

    path = filedialog.askopenfilename(
        title="Select Test CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not path:
        return

    try:
        test_df = pd.read_csv(path)
    except Exception as e:
        messagebox.showerror("Read Error", f"Could not read CSV:\n{e}")
        return

    # Preprocess test with existing encoders
    test_proc = test_df.loc[:, ~test_df.columns.str.contains(r'^Unnamed', na=False)].copy()

    # If test has 'Action', drop it (in case of labels present)
    if 'Action' in test_proc.columns:
        test_proc = test_proc.drop(columns=['Action'])

    # Encode object columns using fitted encoders if available
    for col in test_proc.select_dtypes(include='object').columns:
        if col in label_encoders:
            test_proc[col] = label_encoders[col].transform(test_proc[col].astype(str))
        else:
            # unseen object column -> error
            messagebox.showerror("Encoding Error", f"Missing encoder for column '{col}'. Run training preprocessing with that column present.")
            return

    # Fill numeric NaNs
    num_cols = test_proc.select_dtypes(include=[np.number]).columns
    test_proc[num_cols] = test_proc[num_cols].fillna(test_proc[num_cols].mean())

    # Choose the best model (Hybrid) by default for prediction
    best_model_file = 'mlp_decision_tree_voting.joblib'
    model_path = os.path.join(MODEL_DIR, best_model_file)
    if not os.path.exists(model_path):
        messagebox.showerror("Model Missing", f"'{best_model_file}' not found in '{MODEL_DIR}'.")
        return

    try:
        model = load(model_path)
        y_pred = model.predict(test_proc)
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Prediction failed:\n{e}")
        return

    # Decode predictions to original Action labels if available
    if 'Action' in label_encoders:
        try:
            y_pred_decoded = label_encoders['Action'].inverse_transform(y_pred)
        except Exception:
            y_pred_decoded = y_pred
    else:
        y_pred_decoded = y_pred

    out_df = test_df.copy()
    out_df['Predicted Label'] = y_pred_decoded
    save_path = os.path.join(RESULTS_DIR, "Predictions_Appended.csv")
    try:
        out_df.to_csv(save_path, index=False)
    except Exception as e:
        messagebox.showerror("Save Error", f"Could not save predictions:\n{e}")
        return

    log_output(f"Predictions saved to: {save_path}")
    with pd.option_context('display.max_columns', None, 'display.width', 120):
        log_output("Predictions preview:")
        log_output(out_df.head().to_string())

# -------------------- TKINTER UI --------------------
root = tk.Tk()
root.title("FireWallEye: Real-Time Packet Inspection and Threat Classification Engine")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set window size to full screen
root.geometry(f"{screen_width}x{screen_height}")
#root.geometry("1150x750")

# Set Background Image
def setBackground():
    global bg_photo, bg_label
    image_path = r"BG_image\IMG.webp" # Update with correct image path
    bg_image = Image.open(image_path)
    bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
    #bg_image = bg_image.resize((900, 600), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)

setBackground()

# Background first
label1 = tk.Label(root, image=bg_photo)
label1.image = bg_photo
label1.place(x=0, y=0)

# Title label
title = tk.Label(root, text="FireWallEye: Real-Time Packet Inspection and Threat Classification Engine",
                 font=("Times New Roman", 24, "bold"),
                 bg="black", fg="red")
title.pack(side=tk.TOP, fill=tk.X)

# Force title to front
title.lift()

def connect_db():
    return pymysql.connect(host='localhost', user='root', password='root', database='sparse_db')

# Signup Functionality
def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)"
                cursor.execute(query, (username, password, role))
                conn.commit()
                conn.close()
                messagebox.showinfo("Success", f"{role} Signup Successful!")
                signup_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    signup_window = tk.Toplevel(root)
    signup_window.geometry("400x300")
    signup_window.title(f"{role} Signup")

    tk.Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)

    tk.Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)

# Login Functionality
def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_db()
                cursor = conn.cursor()
                query = "SELECT * FROM users WHERE username=%s AND password=%s AND role=%s"
                cursor.execute(query, (username, password, role))
                result = cursor.fetchone()
                conn.close()
                if result:
                    messagebox.showinfo("Success", f"{role} Login Successful!")
                    login_window.destroy()
                    if role == "Admin":
                        show_admin_buttons()
                    elif role == "User":
                        show_user_buttons()
                else:
                    messagebox.showerror("Error", "Invalid Credentials!")
            except Exception as e:
                messagebox.showerror("Error", f"Database Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(root)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    tk.Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)

# Admin Button Functions
def show_admin_buttons():
    clear_buttons()
    tk.Button(root, text="Upload Dataset", command=upload_dataset, font=font1).place(x=80, y=150)
    tk.Button(root, text="Data Preprocessing", command=preprocess_data_train, font=font1).place(x=380, y=150)
    tk.Button(root, text="Generate EDA (Popup)", command=generate_eda, font=font1).place(x=600, y=150)
    tk.Button(root, text="Data Splitting (SMOTE)", command=split_data_smote, font=font1).place(x=980, y=150)
    tk.Button(root, text="Ridge Classifier", command=lambda: run_algorithm("Ridge Classifier", "ridge_classifier.joblib"), font=font1).place(x=80, y=200)
    tk.Button(root, text="LDA", command=lambda: run_algorithm("Linear Discriminant Analysis", "lda_classifier.joblib"), font=font1).place(x=380, y=200)
    tk.Button(root, text="Decision Tree", command=lambda: run_algorithm("Decision Tree Classifier", "decision_tree_classifier.joblib"), font=font1).place(x=600, y=200)
    tk.Button(root, text="Neuro Tree Fusion", command=lambda: run_algorithm("MLP + Decision Tree Voting", "mlp_decision_tree_voting.joblib"), font=font1).place(x=980, y=200)



# User Button Functions
def show_user_buttons():
    clear_buttons()
    tk.Button(root, text="Prediction (Upload Test CSV)", command=predict_on_test, font=font1).place(x=550, y=200)


# Clear buttons before adding new ones
def clear_buttons():
    for widget in root.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()



# Admin and User Buttons
font1 = ('times', 14, 'bold')


tk.Button(root, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=20, height=1, bg='Lightpink').place(x=100, y=100)

tk.Button(root, text="User Signup", command=lambda: signup("User"), font=font1, width=20, height=1, bg='Lightpink').place(x=400, y=100)


admin_button = tk.Button(root, text="Admin Login", command=lambda: login("Admin"), font=font1, width=20, height=1, bg='Lightpink')
admin_button.place(x=700, y=100)

user_button = tk.Button(root, text="User Login", command=lambda: login("User"), font=font1, width=20, height=1, bg='Lightpink')
user_button.place(x=1000, y=100)


#root = tk.Tk()

# Define font
font1 = ("Times New Roman", 12, "bold")

# Create Text widget
output_box = Text(root, height=20, width=150, wrap=tk.WORD)
output_box.config(font=font1)

# Attach Scrollbar
# scroll = Scrollbar(root)
# scroll.config(command=output_box.yview)
# output_box.config(yscrollcommand=scroll.set)

# Place widgets
output_box.place(x=100, y=300)
#scroll.place(x=100+805, y=300, height=325)  # Adjust position for scrollbar

root.mainloop()

