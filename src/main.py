# ============================================
# PLACE MATE – STUDENT PLACEMENT SYSTEM
# ============================================

import os
import pandas as pd

# --------------------------------------------
# STEP 1: MODEL TRAINING
# --------------------------------------------
print("\n" + "="*50)
print("🚀 WELCOME TO PLACEMATE")
print("Your Smart Student Placement Assistant")
print("="*50)

print("\n🔄 MODEL TRAINING IN PROGRESS...\n")
os.system("python src/train_model.py")

print("\n✅ MODEL TRAINING COMPLETED SUCCESSFULLY")
print("-"*50)

# --------------------------------------------
# STEP 2: LOAD / CREATE STUDENT DATABASE
# --------------------------------------------
CSV_FILE = "registered_students.csv"

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    df = pd.DataFrame(columns=[
        "student_id", "student_name", "cgpa",
        "coding_skill", "communication_skill", "aptitude_skill",
        "problem_solving", "projects_count", "internship_count",
        "certification_count",
        "internship_company_level", "certification_company_level",
        "technical_skills", "tools_known"
    ])

# --------------------------------------------
# UTILITY FUNCTION
# --------------------------------------------
def calculate_results(student):

    score = (
        (student["cgpa"] / 10) * 20 +
        (student["coding_skill"] / 10) * 20 +
        (student["communication_skill"] / 10) * 10 +
        (student["aptitude_skill"] / 10) * 15 +
        (student["problem_solving"] / 10) * 15 +
        (student["projects_count"] / 5) * 5 +
        (student["internship_count"] / 5) * 5 +
        (student["internship_company_level"] / 3) * 5 +
        (student["certification_count"] / 5) * 3 +
        (student["certification_company_level"] / 3) * 2
    )

    readiness = round(min(score, 100), 2)
    probability = round(readiness / 100, 2)
    placement = "PLACED ✅" if readiness >= 65 else "NOT PLACED ❌"

    skills = {
        "Coding": student["coding_skill"],
        "Aptitude": student["aptitude_skill"],
        "Communication": student["communication_skill"],
        "Problem Solving": student["problem_solving"]
    }

    strength = max(skills, key=skills.get)
    weakness = min(skills, key=skills.get)

    career_map = {
        "Coding": "Software Developer / Data Scientist",
        "Aptitude": "Data Analyst / Research",
        "Communication": "HR / Business Analyst",
        "Problem Solving": "Product Manager / Consultant"
    }

    companies = {
        "Google": 8.0,
        "Microsoft": 8.0,
        "Amazon": 7.5,
        "TCS": 6.5,
        "Infosys": 6.0
    }

    eligible = [c for c, v in companies.items() if student["cgpa"] >= v]

    return readiness, probability, placement, strength, weakness, career_map[strength], eligible

# --------------------------------------------
# STEP 3: STUDENT LOGIN / REGISTRATION
# --------------------------------------------
print("\n🧑‍🎓 STUDENT LOGIN / REGISTRATION")
print("-"*50)

student_id = int(input("👉 Enter Student ID: "))

if student_id in df["student_id"].values:
    student = df[df["student_id"] == student_id].iloc[0]
    print(f"\n👋 Welcome back, {student['student_name']}!\n")
else:
    print("\n📝 NEW STUDENT REGISTRATION")
    print("-"*50)

    student = pd.Series({
        "student_id": student_id,
        "student_name": input("Name: "),
        "cgpa": float(input("CGPA (0-10): ")),
        "coding_skill": float(input("Coding Skill (0-10): ")),
        "communication_skill": float(input("Communication Skill (0-10): ")),
        "aptitude_skill": float(input("Aptitude Skill (0-10): ")),
        "problem_solving": float(input("Problem Solving (0-10): ")),
        "projects_count": int(input("Projects Completed: ")),
        "internship_count": int(input("Internships Done: ")),
        "certification_count": int(input("Certifications: ")),
        "internship_company_level": int(input("Internship Company Level (0-3): ")),
        "certification_company_level": int(input("Certification Company Level (0-3): ")),
        "technical_skills": input("Technical Skills (comma separated): "),
        "tools_known": input("Tools Known (comma separated): ")
    })

    df = pd.concat([df, pd.DataFrame([student])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    print("\n✅ Student Registered Successfully!\n")

# --------------------------------------------
# STEP 4: STUDENT DASHBOARD
# --------------------------------------------
readiness, probability, placement, strength, weakness, career, eligible = calculate_results(student)

print("\n📊 STUDENT PLACEMENT DASHBOARD")
print("="*50)
print(f"👤 Name                  : {student['student_name']}")
print(f"🆔 Student ID            : {student['student_id']}")
print(f"🎓 CGPA                  : {student['cgpa']}")
print(f"📈 Placement Probability : {probability}")
print(f"📊 Readiness Score       : {readiness}")
print(f"💪 Strength              : {strength}")
print(f"⚠️ Weakness              : {weakness}")
print(f"🎯 Career Suggestion     : {career}")
print(f"🏢 Eligible Companies    : {', '.join(eligible)}")
print(f"✅ Placement Status      : {placement}")
print("="*50)
print("\n🎉 Thank you for using PlaceMate!")
