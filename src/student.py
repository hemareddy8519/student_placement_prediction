import pandas as pd

class Student:
    def __init__(self, student_name, student_id, cgpa, coding_skill, communication_skill,
                 aptitude_skill, problem_solving, projects_count, internship_count, certification_count):
        self.name = student_name
        self.id = student_id
        self.cgpa = cgpa
        self.coding = coding_skill
        self.communication = communication_skill
        self.aptitude = aptitude_skill
        self.problem_solving = problem_solving
        self.projects = projects_count
        self.internships = internship_count
        self.certifications = certification_count
        self.readiness_score = self.calculate_readiness()
    
    def calculate_readiness(self):
        return round(self.cgpa*10*0.3 + self.coding*0.25 + self.aptitude*0.25 + self.communication*0.2, 2)
    
    def strength_weakness(self):
        scores = {
            "Coding": self.coding,
            "Aptitude": self.aptitude,
            "Communication": self.communication,
            "Problem Solving": self.problem_solving
        }
        return {"Strength": max(scores, key=scores.get),
                "Weakness": min(scores, key=scores.get)}
    
    def career_path(self):
        scores = {
            "Coding": self.coding,
            "Aptitude": self.aptitude,
            "Communication": self.communication,
            "Problem Solving": self.problem_solving
        }
        top_skill = max(scores, key=scores.get)
        if top_skill == "Coding":
            return "Software Development / Data Science"
        elif top_skill == "Aptitude":
            return "Analytics / Research"
        elif top_skill == "Communication":
            return "HR / Business Development"
        elif top_skill == "Problem Solving":
            return "Product Management / Consulting"
    
    def eligible_companies(self, company_criteria):
        return [company for company, cgpa_req in company_criteria.items() if self.cgpa >= cgpa_req]
    
    def model_features(self):
        return pd.DataFrame([{
            "cgpa": self.cgpa,
            "coding_skill": self.coding,
            "communication_skill": self.communication,
            "aptitude_skill": self.aptitude,
            "problem_solving": self.problem_solving,
            "projects_count": self.projects,
            "internship_count": self.internships,
            "certification_count": self.certifications,
            "readiness_score": self.readiness_score
        }])
