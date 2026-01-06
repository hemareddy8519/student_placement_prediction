# app.py

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.secret_key = "placement_secret_key"

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "placement_model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
CSV_FILE = os.path.join(BASE_DIR, "registered_students.csv")

# Ensure artifacts folder exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# -------------------------------
# Load Model & Scaler safely
# -------------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print(f"Warning: Model not found at {MODEL_PATH}. Train the model first.")

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = None
    print(f"Warning: Scaler not found at {SCALER_PATH}. Create and save the scaler first.")

# -------------------------------
# Load / Create CSV
# -------------------------------
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    df = pd.DataFrame(columns=[
        "student_id","student_name","cgpa","coding_skill","communication_skill",
        "aptitude_skill","problem_solving","projects_count","internship_count",
        "internship_company_level","certification_count",
        "certification_company_level","technical_skills","tools_known"
    ])

# -------------------------------
# FEATURE ORDER (MODEL TRAIN ORDER)
# -------------------------------
FEATURE_COLS = [
    "cgpa","coding_skill","communication_skill","aptitude_skill","problem_solving",
    "projects_count","internship_count","internship_company_level",
    "certification_count","certification_company_level"
]

# -------------------------------
# Readiness Score (Rule-based)
# -------------------------------
def calculate_readiness(s):
    score = (
        (s["cgpa"]/10)*20 +
        (s["coding_skill"]/10)*20 +
        (s["communication_skill"]/10)*10 +
        (s["aptitude_skill"]/10)*15 +
        (s["problem_solving"]/10)*15 +
        (s["projects_count"]/5)*5 +
        (s["internship_count"]/5)*5 +
        (s["internship_company_level"]/3)*5 +
        (s["certification_count"]/5)*3 +
        (s["certification_company_level"]/3)*2
    )
    return round(min(score,100),2)

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# -------------------------------
# Login
# -------------------------------
@app.route("/login", methods=["GET","POST"])
def login():
    error = None
    if request.method == "POST":
        role = request.form.get("role")
        user_id = request.form.get("user_id","").strip()

        if role=="student":
            if not user_id:
                error="Enter Student ID"
                return render_template("login.html", error=error)
            try:
                user_id=int(user_id)
            except ValueError:
                error="Student ID must be numeric"
                return render_template("login.html", error=error)

            session["role"]="student"
            session["student_id"]=user_id

            if user_id in df["student_id"].values:
                return redirect(url_for("student_dashboard", student_id=user_id))
            else:
                return redirect(url_for("student_register", new_id=user_id))

        elif role=="faculty":
            session["role"]="faculty"
            return redirect(url_for("faculty_dashboard"))

        elif role=="admin":
            session["role"]="admin"
            return redirect(url_for("admin_dashboard"))

        else:
            error="Select a valid role"

    return render_template("login.html", error=error)

# -------------------------------
# Student Registration
# -------------------------------
@app.route("/student/register", methods=["GET","POST"])
def student_register():
    global df
    new_id = request.args.get("new_id")
    if request.method=="POST":
        student_id = int(request.form["student_id"])
        student = {
            "student_id": student_id,
            "student_name": request.form["student_name"],
            "cgpa": float(request.form["cgpa"]),
            "coding_skill": float(request.form["coding_skill"]),
            "communication_skill": float(request.form["communication_skill"]),
            "aptitude_skill": float(request.form["aptitude_skill"]),
            "problem_solving": float(request.form["problem_solving"]),
            "projects_count": int(request.form["projects_count"]),
            "internship_count": int(request.form["internship_count"]),
            "internship_company_level": int(request.form["internship_company_level"]),
            "certification_count": int(request.form["certification_count"]),
            "certification_company_level": int(request.form["certification_company_level"]),
            "technical_skills": request.form["technical_skills"],
            "tools_known": request.form["tools_known"]
        }
        df = pd.concat([df, pd.DataFrame([student])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        session["student_id"]=student_id
        return redirect(url_for("student_dashboard", student_id=student_id))
    return render_template("student_registration.html", student_id=new_id)

# -------------------------------
# Student Dashboard
# -------------------------------
@app.route("/student/<int:student_id>")
def student_dashboard(student_id):
    if session.get("role")!="student":
        return redirect(url_for("login"))

    student = df[df["student_id"]==student_id]
    if student.empty:
        return redirect(url_for("student_register"))

    s = student.iloc[0]
    readiness = calculate_readiness(s)

    features = pd.DataFrame([{
        "cgpa": s["cgpa"],
        "coding_skill": s["coding_skill"],
        "communication_skill": s["communication_skill"],
        "aptitude_skill": s["aptitude_skill"],
        "problem_solving": s["problem_solving"],
        "projects_count": s["projects_count"],
        "internship_count": s["internship_count"],
        "internship_company_level": s["internship_company_level"],
        "certification_count": s["certification_count"],
        "certification_company_level": s["certification_company_level"]
    }], columns=FEATURE_COLS)

    if model and scaler:
        prob = round(model.predict_proba(scaler.transform(features))[0][1]*100,2)
        placement_status = "Placed" if prob>=60 else "Not Placed"
    else:
        prob = 0
        placement_status = "Model not loaded"

    skills = ["cgpa","coding_skill","communication_skill","aptitude_skill","problem_solving"]
    strengths = [skill.replace("_"," ").title() for skill in skills if s[skill]>=7][:2]
    weaknesses = [skill.replace("_"," ").title() for skill in skills if s[skill]<7][:2]

    if readiness>=80 and prob>=85:
        eligible_companies=["Google","Amazon","Microsoft"]
        career_path="Top Tech Companies"
    elif readiness>=70 and prob>=70:
        eligible_companies=["Infosys","Accenture","TCS Digital"]
        career_path="Mid-level IT Companies"
    elif readiness>=55 and prob>=55:
        eligible_companies=["Wipro","Capgemini"]
        career_path="Entry-level IT Jobs"
    else:
        eligible_companies=["Startup Companies","Internships"]
        career_path="Internships / Startups"

    return render_template("student_dashboard.html",
                           student=s,
                           readiness_score=readiness,
                           placement_probability=prob,
                           placement_status=placement_status,
                           strengths=strengths,
                           weaknesses=weaknesses,
                           eligible_companies=eligible_companies,
                           career_path=career_path)

# -------------------------------
# Faculty Dashboard
# -------------------------------
@app.route("/faculty")
def faculty_dashboard():
    if session.get("role")!="faculty":
        return redirect(url_for("login"))

    students_list=[]
    total_readiness=0
    placed_count=0
    not_placed_count=0

    for _,s in df.iterrows():
        readiness = calculate_readiness(s)
        features=pd.DataFrame([{
            "cgpa": s["cgpa"], "coding_skill": s["coding_skill"],
            "communication_skill": s["communication_skill"], "aptitude_skill": s["aptitude_skill"],
            "problem_solving": s["problem_solving"], "projects_count": s["projects_count"],
            "internship_count": s["internship_count"], "internship_company_level": s["internship_company_level"],
            "certification_count": s["certification_count"], "certification_company_level": s["certification_company_level"]
        }], columns=FEATURE_COLS)

        if model and scaler:
            pred_label = "Placed" if model.predict(scaler.transform(features))[0]==1 else "Not Placed"
        else:
            pred_label = "Model not loaded"

        if pred_label=="Placed":
            placed_count+=1
        else:
            not_placed_count+=1

        total_readiness+=readiness
        students_list.append({
            "name": s["student_name"],
            "cgpa": s["cgpa"],
            "readiness_score": readiness,
            "placement_status": pred_label
        })

    avg_readiness = round(total_readiness/len(students_list),2) if students_list else 0

    return render_template("faculty_dashboard.html",
                           students=students_list,
                           total_students=len(students_list),
                           placed_count=placed_count,
                           not_placed_count=not_placed_count,
                           avg_readiness=avg_readiness)

# -------------------------------
# Admin Dashboard
# -------------------------------
@app.route("/admin")
def admin_dashboard():
    if session.get("role")!="admin":
        return redirect(url_for("login"))

    total_students=len(df)
    placed=0
    not_placed=0
    total_readiness=0

    for _,s in df.iterrows():
        readiness=calculate_readiness(s)
        total_readiness+=readiness
        features=pd.DataFrame([{
            "cgpa": s["cgpa"], "coding_skill": s["coding_skill"],
            "communication_skill": s["communication_skill"], "aptitude_skill": s["aptitude_skill"],
            "problem_solving": s["problem_solving"], "projects_count": s["projects_count"],
            "internship_count": s["internship_count"], "internship_company_level": s["internship_company_level"],
            "certification_count": s["certification_count"], "certification_company_level": s["certification_company_level"]
        }],columns=FEATURE_COLS)

        if model and scaler:
            prob = model.predict_proba(scaler.transform(features))[0][1]*100
        else:
            prob = 0

        if prob>=60:
            placed+=1
        else:
            not_placed+=1

    avg_readiness = round(total_readiness/total_students,2) if total_students>0 else 0
    placement_rate = round((placed/total_students)*100,2) if total_students>0 else 0

    return render_template("admin_dashboard.html",
                           total_students=total_students,
                           placed=placed,
                           not_placed=not_placed,
                           avg_readiness=avg_readiness,
                           placement_rate=placement_rate)

# -------------------------------
# Logout
# -------------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# -------------------------------
# Run App
# -------------------------------
if __name__=="__main__":
    app.run(debug=True)
