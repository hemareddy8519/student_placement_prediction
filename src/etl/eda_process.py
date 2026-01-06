import matplotlib
matplotlib.use("Agg")   # Disable GUI backend

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
EDA_OUTPUT_DIR = "src/etl/eda_outputs"
REPORT_DIR = "notebook/eda_reports"

os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv(DATA_PATH)

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
df["high_cgpa"] = (df["cgpa"] > 7.5).astype(int)

df["total_skills"] = (
    df["coding_skill"] +
    df["communication_skill"] +
    df["aptitude_skill"] +
    df["problem_solving"]
)

df["experience_score"] = df["internship_count"] + df["projects_count"]
df["ready_flag"] = (df["readiness_score"] > 50).astype(int)

# ==================================================
# 1️⃣ BASIC DATA SUMMARY
# ==================================================
summary_file = os.path.join(EDA_OUTPUT_DIR, "data_summary.txt")

if not os.path.exists(summary_file):
    buffer = io.StringIO()
    df.info(buf=buffer)

    with open(summary_file, "w") as f:
        f.write("BASIC DATA SUMMARY\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Dataset Shape: {df.shape}\n\n")
        f.write("Column Information:\n")
        f.write(buffer.getvalue())
        f.write("\nStatistical Summary:\n")
        f.write(str(df.describe()))

    print("Data summary generated ✅")

# ==================================================
# 2️⃣ STATISTICAL INSIGHTS (NEW FEATURE ✅)
# ==================================================
insights_file = os.path.join(EDA_OUTPUT_DIR, "statistical_insights.txt")

if not os.path.exists(insights_file):
    with open(insights_file, "w") as f:
        f.write("KEY STATISTICAL INSIGHTS\n")
        f.write("=" * 35 + "\n\n")

        f.write(f"Total Students: {len(df)}\n")
        f.write(f"Placement Rate: {df['placement_status'].mean() * 100:.2f}%\n")
        f.write(f"Average CGPA: {df['cgpa'].mean():.2f}\n")
        f.write(f"Average Readiness Score: {df['readiness_score'].mean():.2f}\n\n")

        f.write("Average Skills:\n")
        f.write(f"- Coding Skill: {df['coding_skill'].mean():.2f}\n")
        f.write(f"- Communication Skill: {df['communication_skill'].mean():.2f}\n")
        f.write(f"- Aptitude Skill: {df['aptitude_skill'].mean():.2f}\n")
        f.write(f"- Problem Solving: {df['problem_solving'].mean():.2f}\n\n")

        f.write("Top Correlated Features with Placement:\n")
        corr = df.corr()["placement_status"].sort_values(ascending=False)
        f.write(str(corr))

    print("Statistical insights documented ✅")

# ---------------------------
# HELPER FUNCTION
# ---------------------------
def save_plot(filename, plot_func, title):
    filepath = os.path.join(EDA_OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        plt.figure(figsize=(6, 4))
        plot_func()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        print(f"{filename} generated")

# ==================================================
# 3️⃣ EDA PLOTS
# ==================================================
save_plot(
    "placement_status.png",
    lambda: sns.countplot(x="placement_status", data=df),
    "Placement Status Distribution"
)

save_plot(
    "cgpa_distribution.png",
    lambda: sns.histplot(df["cgpa"], kde=False),
    "CGPA Distribution"
)

save_plot(
    "cgpa_vs_placement.png",
    lambda: sns.boxplot(x="placement_status", y="cgpa", data=df),
    "CGPA vs Placement"
)

save_plot(
    "coding_skill_vs_placement.png",
    lambda: sns.boxplot(x="placement_status", y="coding_skill", data=df),
    "Coding Skill vs Placement"
)

save_plot(
    "communication_skill_vs_placement.png",
    lambda: sns.boxplot(x="placement_status", y="communication_skill", data=df),
    "Communication Skill vs Placement"
)

save_plot(
    "aptitude_skill_vs_placement.png",
    lambda: sns.boxplot(x="placement_status", y="aptitude_skill", data=df),
    "Aptitude Skill vs Placement"
)

save_plot(
    "internship_vs_placement.png",
    lambda: sns.countplot(x="internship_count", hue="placement_status", data=df),
    "Internships vs Placement"
)

save_plot(
    "projects_vs_placement.png",
    lambda: sns.countplot(x="projects_count", hue="placement_status", data=df),
    "Projects vs Placement"
)

save_plot(
    "readiness_score_distribution.png",
    lambda: sns.histplot(df["readiness_score"], kde=False),
    "Readiness Score Distribution"
)

# ---------------------------
# CORRELATION HEATMAP
# ---------------------------
numeric_df = df.select_dtypes(include="number")

save_plot(
    "correlation_heatmap.png",
    lambda: sns.heatmap(numeric_df.corr(), cmap="coolwarm"),
    "Feature Correlation Heatmap"
)

# ==================================================
# 4️⃣ COPY ARTIFACTS TO REPORT FOLDER
# ==================================================
for file in os.listdir(EDA_OUTPUT_DIR):
    shutil.copy(os.path.join(EDA_OUTPUT_DIR, file), REPORT_DIR)

print(f"\nAll artifacts copied to {REPORT_DIR} ✅")
print("EDA + Statistical insights completed successfully ✅")
