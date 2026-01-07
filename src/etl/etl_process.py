import pandas as pd

# -------------------------------
# ---------- EXTRACT ----------
# -------------------------------
def extract_data():
    print("Extracting data...")
    df = pd.read_csv("notebook/student_placement_prediction_dataset.csv")
    print("Data extracted successfully")
    return df


# -------------------------------
# ---------- TRANSFORM ----------
# -------------------------------
def transform_data(df):
    print("Transforming data...")

    # -------------------------------
    # 0. Remove Duplicate Records
    # -------------------------------
    if "student_id" in df.columns:
        df.drop_duplicates(subset="student_id", inplace=True)
    else:
        df.drop_duplicates(inplace=True)

    # -------------------------------
    # 1. Handle Missing Values
    # -------------------------------
    df.fillna(0, inplace=True)

    # -------------------------------
    # 2. Convert Numeric Columns (FLOAT)
    # -------------------------------
    numeric_cols = [
        "cgpa", "coding_skill", "communication_skill",
        "aptitude_skill", "problem_solving"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # -------------------------------
    # 3. Convert Count Columns (INT)
    # -------------------------------
    count_cols = [
        "projects_count", "internship_count", "certification_count"
    ]

    for col in count_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # -------------------------------
    # 4. Handle Invalid Value Ranges
    # -------------------------------
    df["cgpa"] = df["cgpa"].clip(0, 10)

    skill_cols = [
        "coding_skill", "communication_skill",
        "aptitude_skill", "problem_solving"
    ]
    for col in skill_cols:
        df[col] = df[col].clip(0, 10)

    for col in count_cols:
        df[col] = df[col].clip(0, 100)

    # -------------------------------
    # 5. Encode Placement Status (TARGET)
    # -------------------------------
    if "placement_status" in df.columns:
        df["placement_status"] = df["placement_status"].map({
            "Placed": 1,
            "Not Placed": 0,
            "Yes": 1,
            "No": 0,
            1: 1,
            0: 0
        }).fillna(0).astype(int)

    # -------------------------------
    # 6. Encode Company Levels (INT)
    # -------------------------------
    level_map = {
        "None": 0,
        "Startup": 1,
        "Local": 1,
        "Mid": 2,
        "National": 2,
        "MNC": 3,
        "International": 3
    }

    for col in ["internship_company_level", "certification_company_level"]:
        if col in df.columns:
            df[col] = df[col].map(level_map).fillna(0).astype(int)

    # -------------------------------
    # 7. Feature Engineering – Readiness Score
    # -------------------------------
    df["readiness_score"] = (
        (df["cgpa"] / 10) * 20 +
        (df["coding_skill"] / 10) * 20 +
        (df["aptitude_skill"] / 10) * 15 +
        (df["communication_skill"] / 10) * 10 +
        (df["problem_solving"] / 10) * 15 +
        (df["projects_count"] / 5) * 10 +
        (df["internship_count"] / 5) * 5 +
        (df["certification_count"] / 5) * 5
    ).clip(0, 100)

    # -------------------------------
    # 7a. Extra Feature Engineering
    # -------------------------------

    # High CGPA flag
    df['high_cgpa'] = (df['cgpa'] > 7.5).astype(int)

    # Total skill score
    df['total_skills'] = df['coding_skill'] + df['communication_skill'] + df['aptitude_skill'] + df['problem_solving']

    # Internships + Projects experience
    df['experience_score'] = df['internship_count'] + df['projects_count']

    # Ready to place flag
    df['ready_flag'] = (df['readiness_score'] > 50).astype(int)


    # -------------------------------
    # 8. Final Datatype Enforcement (IMPORTANT)
    # -------------------------------
    df["placement_status"] = df["placement_status"].astype(int)
    df["readiness_score"] = df["readiness_score"].astype(float)

    print("Transformation + datatype correction completed ✅")
    print(df.dtypes)   # DEBUG CHECK

    return df


# -------------------------------
# ---------- LOAD ----------
# -------------------------------
def load_data(df):
    print("Loading data...")
    df.to_csv("notebook/final_transformed_data.csv", index=False)
    print("ETL completed. File saved as final_transformed_data.csv")


# -------------------------------
# ---------- MAIN ----------
# -------------------------------
def main():
    df = extract_data()      # E
    df = transform_data(df)  # T
    load_data(df)            # L


if __name__ == "__main__":
    main()
