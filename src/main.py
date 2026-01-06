import pandas as pd
import os

CSV_FILE = "registered_students.csv"

# -----------------------------
# Load or create CSV
# -----------------------------
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

# -----------------------------
# Utility Functions
# -----------------------------
def calculate_results(student):
    # -----------------------------
    # Readiness Score (0-100)
    # Weighted contributions similar to ML model
    # -----------------------------
    score = (
        (student["cgpa"] / 10) * 20 +                   # CGPA max 20
        (student["coding_skill"] / 10) * 20 +          # Coding max 20
        (student["communication_skill"] / 10) * 10 +   # Communication max 10
        (student["aptitude_skill"] / 10) * 15 +        # Aptitude max 15
        (student["problem_solving"] / 10) * 15 +       # Problem Solving max 15
        (student["projects_count"] / 5) * 5 +          # Projects max 5
        (student["internship_count"] / 5) * 5 +        # Internships max 5
        (student["internship_company_level"] / 3) * 5 +# Internship Level max 5
        (student["certification_count"] / 5) * 3 +     # Certifications max 3
        (student["certification_company_level"] / 3) * 2 # Certification Level max 2
    )

    readiness = round(min(score, 100), 2)  # Ensure 0-100
    placement = "Placed" if readiness >= 65 else "Not Placed"

    # Strength & Weakness
    skills = {
        "Coding": student["coding_skill"],
        "Aptitude": student["aptitude_skill"],
        "Communication": student["communication_skill"],
        "Problem Solving": student["problem_solving"]
    }
    strength = max(skills, key=skills.get)
    weakness = min(skills, key=skills.get)

    # Career Suggestions
    career_map = {
        "Coding": "Software Developer / Data Scientist",
        "Aptitude": "Data Analyst / Research",
        "Communication": "HR / Business Analyst",
        "Problem Solving": "Product Manager / Consultant"
    }

    # Eligible Companies based on CGPA
    companies = {
        "Google": 8.0,
        "Microsoft": 8.0,
        "Amazon": 7.5,
        "TCS": 6.5,
        "Infosys": 6.0
    }
    eligible = [c for c, v in companies.items() if student["cgpa"] >= v]

    return readiness, placement, strength, weakness, career_map[strength], eligible

# -----------------------------
# Terminal Interface
# -----------------------------
print("===== STUDENT PLACEMENT DASHBOARD =====")
student_id = int(input("Enter your Student ID: "))

if student_id in df["student_id"].values:
    student = df[df["student_id"] == student_id].iloc[0]
    print(f"\nWelcome back {student['student_name']}!\n")
else:
    print("\nNew Student Registration\n")
    student_name = input("Name: ")
    cgpa = float(input("CGPA (0-10): "))
    coding = float(input("Coding Skill (0-10): "))
    communication = float(input("Communication Skill (0-10): "))
    aptitude = float(input("Aptitude Skill (0-10): "))
    problem_solving = float(input("Problem Solving (0-10): "))
    projects = int(input("Projects Completed: "))
    internships = int(input("Internships Done: "))
    certifications = int(input("Certifications: "))
    internship_level = int(input("Internship Company Level (0-3): "))
    certification_level = int(input("Certification Company Level (0-3): "))
    technical_skills = input("Technical Skills (comma separated): ")
    tools_known = input("Tools Known (comma separated): ")

    student = pd.Series({
        "student_id": student_id,
        "student_name": student_name,
        "cgpa": cgpa,
        "coding_skill": coding,
        "communication_skill": communication,
        "aptitude_skill": aptitude,
        "problem_solving": problem_solving,
        "projects_count": projects,
        "internship_count": internships,
        "certification_count": certifications,
        "internship_company_level": internship_level,
        "certification_company_level": certification_level,
        "technical_skills": technical_skills,
        "tools_known": tools_known
    })

    df = pd.concat([df, pd.DataFrame([student])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"\nStudent {student_name} registered successfully!\n")

# -----------------------------
# Calculate Results
# -----------------------------
readiness, placement, strength, weakness, career, eligible = calculate_results(student)

# -----------------------------
# Display Dashboard
# -----------------------------
print("===== DASHBOARD =====")
print(f"Name: {student['student_name']}")
print(f"Student ID: {student['student_id']}")
print(f"CGPA: {student['cgpa']}")
print(f"Technical Skills: {student['technical_skills']}")
print(f"Tools Known: {student['tools_known']}")
print(f"Readiness Score: {readiness}")
print(f"Strength: {strength}")
print(f"Weakness: {weakness}")
print(f"Career Path Suggestion: {career}")
print(f"Eligible Companies: {', '.join(eligible)}")
print(f"Placement Status: {placement}")
print("=======================================")
