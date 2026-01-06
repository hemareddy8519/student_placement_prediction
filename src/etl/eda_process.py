import matplotlib
matplotlib.use("Agg")   # GUI disable

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io
import shutil

# ---------------------------
# CONFIG
# ---------------------------
DATA_PATH = "notebook/final_transformed_data.csv"
EDA_OUTPUT_DIR = "etl/eda_outputs"
REPORT_DIR = "notebook/eda_reports"  # <-- DIFFERENT from EDA_OUTPUT_DIR

os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv(DATA_PATH)

# ---------------------------
# Feature Engineering (Checklist 3 & 4)
# ---------------------------
df['high_cgpa'] = (df['cgpa'] > 7.5).astype(int)
df['total_skills'] = df['coding_skill'] + df['communication_skill'] + df['aptitude_skill'] + df['problem_solving']
df['experience_score'] = df['internship_count'] + df['projects_count']
df['ready_flag'] = (df['readiness_score'] > 50).astype(int)  # Checklist 4

# ==================================================
# 1️⃣ BASIC DATA SUMMARY (TEXT FILE)
# ==================================================
summary_file = os.path.join(EDA_OUTPUT_DIR, "data_summary.txt")
if not os.path.exists(summary_file):
    buffer = io.StringIO()
    df.info(buf=buffer)
    with open(summary_file,"w") as f:
        f.write("BASIC DATA SUMMARY\n====================\n\n")
        f.write(f"Dataset Shape: {df.shape}\n\n")
        f.write("Column Information:\n")
        f.write(buffer.getvalue())
        f.write("\n\nStatistical Summary:\n")
        f.write(str(df.describe()))
    print("Data summary generated ✅")

# ---------------------------
# HELPER FUNCTION
# ---------------------------
def save_plot(filename, plot_func, title=None):
    filepath = os.path.join(EDA_OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        plot_func()
        if title:
            plt.title(title)
        plt.savefig(filepath)
        plt.close()
        print(f"{filename} generated")

# ==================================================
# 2️⃣ EDA PLOTS
# ==================================================
save_plot("placement_status.png", lambda: sns.countplot(x="placement_status", data=df), "Placement Status Distribution")
save_plot("cgpa_distribution.png", lambda: sns.histplot(df["cgpa"], kde=True), "CGPA Distribution")
save_plot("cgpa_vs_placement.png", lambda: sns.boxplot(x="placement_status", y="cgpa", data=df), "CGPA vs Placement")
save_plot("coding_skill_vs_placement.png", lambda: sns.boxplot(x="placement_status", y="coding_skill", data=df), "Coding Skill vs Placement")
save_plot("communication_skill_vs_placement.png", lambda: sns.boxplot(x="placement_status", y="communication_skill", data=df), "Communication Skill vs Placement")
save_plot("aptitude_skill_vs_placement.png", lambda: sns.boxplot(x="placement_status", y="aptitude_skill", data=df), "Aptitude Skill vs Placement")
save_plot("internship_vs_placement.png", lambda: sns.countplot(x="internship_count", hue="placement_status", data=df), "Internships vs Placement")
save_plot("projects_vs_placement.png", lambda: sns.countplot(x="projects_count", hue="placement_status", data=df), "Projects vs Placement")
save_plot("readiness_score_distribution.png", lambda: sns.histplot(df["readiness_score"], kde=True), "Readiness Score Distribution")

# 🔟 Correlation Heatmap (Checklist 4)
numeric_df = df.select_dtypes(include="number")
save_plot("correlation_heatmap.png", lambda: sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm"), "Feature Correlation Heatmap")

# ---------------------------
# 5️⃣ Copy all artifacts to reports folder (Checklist 5)
# ---------------------------
# Only copy if source and destination are different
dest_summary = os.path.join(REPORT_DIR, "data_summary.txt")
if os.path.abspath(summary_file) != os.path.abspath(dest_summary):
    shutil.copy(summary_file, REPORT_DIR)

# Copy all plots
for file in os.listdir(EDA_OUTPUT_DIR):
    if file.endswith(".png"):
        shutil.copy(os.path.join(EDA_OUTPUT_DIR, file), REPORT_DIR)

print(f"\nAll artifacts copied to {REPORT_DIR} ✅")
print("EDA process completed successfully ✅")
