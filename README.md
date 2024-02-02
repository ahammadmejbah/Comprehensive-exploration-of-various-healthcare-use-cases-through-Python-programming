<div align="center">
      <h2> <img src="http://bytesofintelligences.com/wp-content/uploads/2023/03/Exploring-AIs-Secrets-1.png" width="300px"><br/> <p>Comprehensive Exploration of Various <span style="color: #007BFF;">Health-Care</span> Use Cases Through <span style="color: red;">Python Programming</span></p> </h2>
     </div>

<body>
<p align="center">
  <a href="mailto:ahammadmejbah@gmail.com"><img src="https://img.shields.io/badge/Email-ahammadmejbah%40gmail.com-blue?style=flat-square&logo=gmail"></a>
  <a href="https://github.com/BytesOfIntelligences"><img src="https://img.shields.io/badge/GitHub-%40BytesOfIntelligences-lightgrey?style=flat-square&logo=github"></a>
  <a href="https://linkedin.com/in/ahammadmejbah"><img src="https://img.shields.io/badge/LinkedIn-Mejbah%20Ahammad-blue?style=flat-square&logo=linkedin"></a>
  <a href="https://bytesofintelligences.com/"><img src="https://img.shields.io/badge/Website-Bytes%20of%20Intelligence-lightgrey?style=flat-square&logo=google-chrome"></a>
  <a href="https://www.youtube.com/@BytesOfIntelligences"><img src="https://img.shields.io/badge/YouTube-BytesofIntelligence-red?style=flat-square&logo=youtube"></a>
  <a href="https://www.researchgate.net/profile/Mejbah-Ahammad-2"><img src="https://img.shields.io/badge/ResearchGate-Mejbah%20Ahammad-blue?style=flat-square&logo=researchgate"></a>
  <br>
  <img src="https://img.shields.io/badge/Phone-%2B8801874603631-green?style=flat-square&logo=whatsapp">
  <a href="https://www.hackerrank.com/profile/ahammadmejbah"><img src="https://img.shields.io/badge/Hackerrank-ahammadmejbah-green?style=flat-square&logo=hackerrank"></a>
</p>

#
  
  It covers topics like patient appointment scheduling and notification, medication adherence tracking, emergency room wait time prediction, patient health risk assessment, and health monitoring systems. Additionally, it delves into healthcare facility resource optimization, scheduling appointments with doctors, automated health risk assessments, and specific disease risk predictions including diabetes, heart disease, and Alzheimer's. Each section provides detailed examples and codes to demonstrate the applications of these systems in improving healthcare efficiency and patient care.
  
# 1. Title: Use Case Title: Patient Appointment Scheduling and Notification System

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to generate random appointments for a week for multiple patients
def generate_appointments(num_patients=5, num_days=7):
    patient_ids = [f'Patient_{i:03d}' for i in range(1, num_patients + 1)]
    dates = [datetime.now().date() + timedelta(days=i) for i in range(num_days)]
    times = ['09:00 AM', '11:00 AM', '02:00 PM', '04:00 PM']
    
    appointments = []
    for patient_id in patient_ids:
        for date in dates:
            time = np.random.choice(times)
            appointments.append([patient_id, date, time])
    
    df_appointments = pd.DataFrame(appointments, columns=['Patient ID',
    'Date', 'Time'])
    return df_appointments

# Function to identify scheduling conflicts within the generated appointments
def detect_conflicts(appointments_df):
    conflicts = appointments_df[appointments_df.duplicated(subset=['Date',
    'Time'], keep=False)]
    return conflicts

# Function to generate reminders for appointments scheduled for the next day
def generate_reminders(appointments_df):
    reminders = appointments_df[appointments_df['Date'] 
    == datetime.now().date() + timedelta(days=1)]
    reminder_messages = []
    for _, row in reminders.iterrows():
        message = f"Reminder: You have an appointment 
        scheduled on {row['Date']} at {row['Time']}. Please arrive 10 minutes early."
        reminder_messages.append((row['Patient ID'], message))
    return reminder_messages

# Example usage of the system
appointments_df = generate_appointments()
conflicts = detect_conflicts(appointments_df)
reminders = generate_reminders(appointments_df)

print("Generated Appointments:")
print(appointments_df)
print("\nScheduling Conflicts Detected (if any):")
print(conflicts)
print("\nAppointment Reminders for the Next Day:")
for patient_id, message in reminders:
    print(f"{patient_id}: {message}")

```

# 2. Title: Use Case Title: Medication Adherence Tracking System

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to generate patients' medication schedules
def generate_medication_schedules(num_patients=10):
    patient_ids = [f'Patient_{i:03d}' for i in range(1, num_patients + 1)]
    medications = ['Med_A', 'Med_B', 'Med_C']
    schedule = []

    for patient_id in patient_ids:
        med = np.random.choice(medications)
        times_per_day = np.random.randint(1, 4)  # Patients take medication 1 to 3 times per day
        for _ in range(times_per_day):
            time = f"{np.random.randint(8, 20):02d}:00"  # Random hour between 08:00 and 19:00
            schedule.append([patient_id, med, time])

    df_schedule = pd.DataFrame(schedule, columns=['Patient ID', 'Medication', 'Scheduled Time'])
    return df_schedule

# Function to simulate medication intake and detect adherence issues
def track_medication_adherence(medication_schedule, days=7, adherence_threshold=0.8):
    adherence_records = []
    for _, row in medication_schedule.iterrows():
        for day in range(days):
            date = (datetime.now() - timedelta(days=day)).date()
            taken = np.random.choice([True, False], p=[0.9, 0.1])  # 90% chance of taking medication
            adherence_records.append([row['Patient ID'], row['Medication'], date, row['Scheduled Time'], taken])

    df_adherence = pd.DataFrame(adherence_records, 
    columns=['Patient ID', 'Medication', 'Date', 'Scheduled Time', 'Taken'])

    # Calculate adherence rate
    adherence_summary = df_adherence.groupby('Patient ID')['Taken'].mean().reset_index()
    adherence_summary['Adherent'] = adherence_summary['Taken'] >= adherence_threshold

    # Identify patients below adherence threshold
    non_adherent_patients = adherence_summary[~adherence_summary['Adherent']]
    return df_adherence, non_adherent_patients

# Example usage of the system
medication_schedule = generate_medication_schedules()
adherence_records, non_adherent_patients = track_medication_adherence(medication_schedule)

print("Medication Adherence Records for the Past Week:")
print(adherence_records)
print("\nPatients with Poor Medication Adherence:")
print(non_adherent_patients)

```
# 3. Title: Emergency Room (ER) Wait Time Prediction


```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to generate simulated ER visit data
def generate_er_visits(num_visits=20):
    np.random.seed(42)  # For reproducible results
    patient_ids = [f'Patient_{i:03d}' for i in range(1, num_visits + 1)]
    arrival_times = [datetime.now() - timedelta(
    minutes=np.random.randint(1, 240)) for _ in range(num_visits)]
    condition_severity = np.random.choice(['Low',
    'Medium', 'High'], size=num_visits, p=[0.5, 0.3, 0.2])
    treatment_durations = np.random.randint(15, 
    121, size=num_visits)  # Treatment duration between 15 and 120 minutes
    
    er_visits = pd.DataFrame({
        'Patient ID': patient_ids,
        'Arrival Time': arrival_times,
        'Condition Severity': condition_severity,
        'Treatment Duration': treatment_durations
    })
    er_visits.sort_values(by='Arrival Time', inplace=True)
    return er_visits

# Function to predict wait times for new patients
def predict_wait_time(er_visits, new_patient_severity):
    # Severity weights (higher severity patients are prioritized)
    severity_weights = {'Low': 1, 'Medium': 2, 'High': 3}
    
    # Current time for the simulation
    current_time = datetime.now()
    
    # Calculate ongoing treatments and wait times
    er_visits['End Time'] = er_visits['Arrival Time'] 
    + pd.to_timedelta(er_visits['Treatment Duration'], unit='min')
    ongoing_treatments = er_visits[er_visits['End Time'] > current_time]
    
    # Adjusted for severity; higher severity may skip ahead
    wait_time = 0
    for _, row in ongoing_treatments.iterrows():
        if severity_weights[row['Condition Severity']] < severity_weights[new_patient_severity]:
            continue
        remaining_treatment_time = (row['End Time'] - current_time).total_seconds() / 60.0
        wait_time += remaining_treatment_time
    
    return max(0, wait_time)  # Ensure wait time is not negative

# Example usage
er_visits = generate_er_visits()
print("Current ER Visits:")
print(er_visits)

# Predict wait time for a new patient with 'Medium' severity condition
new_patient_severity = 'Medium'
predicted_wait_time = predict_wait_time(er_visits, new_patient_severity)
print(f"\nPredicted wait time for a
new patient with {new_patient_severity} condition severity: {predicted_wait_time:.2f} minutes")

```

# 4. Title: Patient Health Risk Assessment

```python
import random

# Function to generate random patient data
def generate_patient_data():
    age = random.randint(20, 90)
    blood_pressure = random.randint(90, 200)
    cholesterol = random.randint(100, 300)
    bmi = round(random.uniform(18.5, 40.0), 1)
    return age, blood_pressure, cholesterol, bmi

# Function to assess health risk
def assess_patient_risk(age, blood_pressure, cholesterol, bmi):
    if age < 40 and blood_pressure < 130 and cholesterol < 200 and bmi < 25:
        return "Low Risk"
    elif 40 <= age <= 60 and 130 <= blood_pressure <= 160 
    and 200 <= cholesterol <= 240 and 25 <= bmi <= 30:
        return "Moderate Risk"
    elif age > 60 or blood_pressure > 160 or cholesterol > 240 or bmi > 30:
        return "High Risk"
    else:
        return "Uncategorized Risk"

# Example usage
patient_age, patient_bp, patient_cholesterol, patient_bmi = generate_patient_data()
risk_category = assess_patient_risk(patient_age, 
patient_bp, patient_cholesterol, patient_bmi)
print(f"Patient Details: Age: {patient_age}, 
BP: {patient_bp}, Cholesterol: {patient_cholesterol},
BMI: {patient_bmi}")
print(f"Risk Assessment: {risk_category}")

```

# 5. Title: Health Monitoring System

```python
import random

# Generate random patient data
def generate_patient_data(num_patients):
    patients = []
    for i in range(num_patients):
        patient_id = i + 1
        heart_rate = random.randint(60, 120)
        systolic_pressure = random.randint(90, 150)
        diastolic_pressure = random.randint(60, 100)
        patients.append({"patient_id": patient_id,
        "heart_rate": heart_rate, "systolic_pressure": systolic_pressure,
        "diastolic_pressure": diastolic_pressure})
    return patients

# Function to monitor patient vitals and send alerts
def monitor_vitals(patients):
    alerts = []
    for patient in patients:
        if patient['heart_rate'] > 100:
            alerts.append(f"Alert: High heart rate for Patient {patient['patient_id']}.
            Heart rate: {patient['heart_rate']}")
        if patient['systolic_pressure'] > 120 or patient['diastolic_pressure'] > 80:
            alerts.append(f"Alert: High blood pressure for Patient {patient['patient_id']}.
            Systolic: {patient['systolic_pressure']}, 
            Diastolic: {patient['diastolic_pressure']}")
    return alerts

# Main function to demonstrate the use case
def main():
    num_patients = 5
    patients = generate_patient_data(num_patients)
    
    print("Patient Data:")
    for patient in patients:
        print(patient)
    
    alerts = monitor_vitals(patients)
    
    print("\nAlerts:")
    for alert in alerts:
        print(alert)

if __name__ == "__main__":
    main()

```


# 6. Title: Healthcare Facility Resource Optimization

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Constants for simulation
CONDITION_SEVERITY = ['Low', 'Medium', 'High']
RESOURCES_NEEDED = {'Low': (1, 1, 1), 'Medium': (2, 2, 2), 'High': (3, 3, 3)}  # (beds, staff, equipment)

# Function to generate simulated daily patient admissions
def generate_patient_admissions(num_patients=50):
    patient_ids = [f'Patient_{i:04d}' for i in range(num_patients)]
    condition_severity = np.random.choice(CONDITION_SEVERITY, size=num_patients, p=[0.4, 0.4, 0.2])
    treatment_duration = np.random.randint(1, 14, size=num_patients)  # Treatment duration in days
    
    patient_admissions = pd.DataFrame({
        'Patient ID': patient_ids,
        'Condition Severity': condition_severity,
        'Treatment Duration (Days)': treatment_duration
    })
    return patient_admissions

# Function to predict resource allocation needs
def predict_resource_needs(patient_admissions):
    resources_needed = {'Beds': 0, 'Staff': 0, 'Equipment': 0}
    for _, row in patient_admissions.iterrows():
        severity = row['Condition Severity']
        beds, staff, equipment = RESOURCES_NEEDED[severity]
        resources_needed['Beds'] += beds
        resources_needed['Staff'] += staff
        resources_needed['Equipment'] += equipment
    
    return resources_needed

# Function to identify resource allocation issues
def identify_resource_issues(predicted_needs, available_resources):
    issues = {}
    for resource, needed in predicted_needs.items():
        if needed > available_resources[resource]:
            issues[resource] = needed - available_resources[resource]
    return issues

# Example usage
patient_admissions = generate_patient_admissions()
predicted_needs = predict_resource_needs(patient_admissions)

# Assuming available resources in the facility
available_resources = {'Beds': 100, 'Staff': 50, 'Equipment': 40}
resource_issues = identify_resource_issues(predicted_needs, available_resources)

print("Predicted Resource Needs Based on Patient Admissions:")
print(predicted_needs)
print("\nIdentified Resource Allocation Issues (if any):")
if resource_issues:
    for resource, shortage in resource_issues.items():
        print(f"Shortage of {shortage} units for {resource}")
else:
    print("No resource allocation issues identified. All needs are within available limits.")

```

# 7. Title: Scheduling appointments for patients with doctors

```python
import random
from datetime import datetime, timedelta

# Generate random patient and doctor data
def generate_healthcare_data(num_patients, num_doctors):
    patients = []
    for i in range(num_patients):
        patient_id = i + 1
        patient_name = f"Patient-{patient_id}"
        patients.append({"patient_id": patient_id, "patient_name": patient_name})

    doctors = []
    for i in range(num_doctors):
        doctor_id = i + 1
        doctor_name = f"Doctor-{doctor_id}"
        available_slots = [datetime(2024, 2, 1,
        random.randint(9, 17), 0) for _ in range(5)] 
        # 5 available slots per doctor
        doctors.append({"doctor_id": doctor_id,
        "doctor_name": doctor_name, "available_slots": available_slots})

    return patients, doctors

# Function to schedule appointments
def schedule_appointments(patients, doctors):
    appointments = []
    for patient in patients:
        chosen_doctor = random.choice(doctors)
        if chosen_doctor["available_slots"]:
            appointment_time = chosen_doctor["available_slots"].pop(0)
            appointments.append({"patient_id": patient["patient_id"],
            "patient_name": patient["patient_name"],
             "doctor_id": chosen_doctor["doctor_id"], "doctor_name": chosen_doctor["doctor_name"],
                                 "appointment_time": appointment_time})
    return appointments

# Main function to demonstrate the use case
def main():
    num_patients = 3
    num_doctors = 2

    patients, doctors = generate_healthcare_data(num_patients, num_doctors)

    print("Patient Data:")
    for patient in patients:
        print(patient)

    print("\nDoctor Data:")
    for doctor in doctors:
        print(doctor)

    appointments = schedule_appointments(patients, doctors)

    print("\nAppointments:")
    for appointment in appointments:
        print(appointment)

if __name__ == "__main__":
    main()

```


# 8. Title: Automated Health Risk Assessment System

```python
import pandas as pd
import numpy as np

# Function to generate simulated patient profiles
def generate_patient_profiles(num_patients=100):
    np.random.seed(42)  # Ensures reproducibility
    patient_ids = [f'Patient_{i+1:03d}' for i in range(num_patients)]
    ages = np.random.randint(18, 80, size=num_patients)
    smoking_status = np.random.choice(['Non-smoker', 'Former smoker', 'Current smoker'], size=num_patients)
    diet_quality = np.random.choice(['Poor', 'Average', 'Excellent'], size=num_patients)
    cholesterol_levels = np.random.randint(150, 250, size=num_patients)  # Total cholesterol mg/dL
    blood_pressure = np.random.randint(90, 180, size=num_patients)  # Systolic blood pressure
    
    patient_data = pd.DataFrame({
        'Patient ID': patient_ids,
        'Age': ages,
        'Smoking Status': smoking_status,
        'Diet Quality': diet_quality,
        'Cholesterol Levels (mg/dL)': cholesterol_levels,
        'Blood Pressure (mmHg)': blood_pressure
    })
    return patient_data

# Function to calculate health risk score for each patient
def calculate_health_risk_scores(patient_profiles):
    risk_scores = []
    for _, row in patient_profiles.iterrows():
        score = 0
        # Age factor
        if row['Age'] > 50: score += 2
        # Smoking factor
        if row['Smoking Status'] == 'Current smoker': score += 3
        elif row['Smoking Status'] == 'Former smoker': score += 1
        # Diet factor
        if row['Diet Quality'] == 'Poor': score += 2
        elif row['Diet Quality'] == 'Average': score += 1
        # Cholesterol factor
        if row['Cholesterol Levels (mg/dL)'] > 200: score += 2
        # Blood Pressure factor
        if row['Blood Pressure (mmHg)'] > 140: score += 2
        risk_scores.append(score)
    
    patient_profiles['Risk Score'] = risk_scores
    return patient_profiles

# Function to categorize patients into risk groups
def categorize_risk_groups(patient_profiles):
    conditions = [
        (patient_profiles['Risk Score'] <= 3),
        (patient_profiles['Risk Score'] > 3) & (patient_profiles['Risk Score'] <= 6),
        (patient_profiles['Risk Score'] > 6)
    ]
    choices = ['Low', 'Medium', 'High']
    patient_profiles['Risk Group'] = np.select(conditions, choices, default='Unknown')
    return patient_profiles

# Example usage
patient_profiles = generate_patient_profiles()
patient_profiles_with_scores = calculate_health_risk_scores(patient_profiles)
categorized_patients = categorize_risk_groups(patient_profiles_with_scores)

print("Sample of Patient Risk Categorization:")
print(categorized_patients[['Patient ID', 'Risk Score', 'Risk Group']].head())

```

# 9. Title: Patient Appointment Scheduling System

```python
import random

# Function to generate random patient data
def generate_patient_data(num_patients):
    patient_data = []
    for _ in range(num_patients):
        patient_id = random.randint(1000, 9999)
        symptom_severity = random.randint(1, 10)
        waiting_time = random.randint(0, 30)  # up to 30 days
        age = random.randint(20, 90)
        patient_data.append((patient_id, symptom_severity,
        waiting_time, age))
    return patient_data

# Function to schedule appointments
def schedule_appointment(patient_data):
    # Sorting logic: primarily by severity, then waiting time, and finally age
    return sorted(patient_data, key=lambda x: (-x[1],
    -x[2], -x[3]))

# Example usage
patients = generate_patient_data(10)
scheduled_patients = schedule_appointment(patients)
print("Scheduled Appointments:")
for patient in scheduled_patients:
    print(f"Patient ID: {patient[0]}, Severity: {patient[1]},
    Waiting Time: {patient[2]} days, Age: {patient[3]}")

```

# 10. Title: Dynamic Patient Queue Management System for Clinics

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to generate a day's schedule of patient appointments
def generate_daily_schedule(num_appointments=20):
    np.random.seed(0)  # For reproducibility
    start_time = datetime.now().replace(hour=9, minute=0,
    second=0, microsecond=0)
    appointment_types = ['Consultation', 'Follow-up', 'Emergency']
    durations = {'Consultation': 30, 'Follow-up': 20, 'Emergency': 45} 
    # Duration in minutes
    
    schedule_data = []
    for i in range(num_appointments):
        appt_type = np.random.choice(appointment_types, p=[0.5, 0.4, 0.1])
        duration = durations[appt_type]
        patient_id = f'Patient_{i+1:03d}'
        # Assume a fixed interval between appointments for simplicity,
        ignoring type for initial schedule
        appt_time = start_time + timedelta(minutes=i * 30)
        schedule_data.append([patient_id, appt_time, appt_type, duration])
    
    schedule_df = pd.DataFrame(schedule_data, columns=['Patient ID',
    
    'Appointment Time', 'Type', 'Duration'])
    return schedule_df

# Function to adjust the schedule based on real-time events
def adjust_schedule(schedule_df):
    # Simulate a no-show and an emergency
    no_show_patient = np.random.choice(schedule_df['Patient ID'])
    emergency_patient_id = 'Emergency_New'
    emergency_appt_time = datetime.now().replace(hour=10, minute=30)
    
    # Remove no-show patient
    adjusted_schedule = schedule_df[schedule_df['Patient ID'] != no_show_patient]
    
    # Add emergency patient
    emergency_entry = pd.DataFrame([[emergency_patient_id,
    emergency_appt_time, 'Emergency', 45]], 
    columns=['Patient ID', 'Appointment Time', 'Type', 'Duration'])
    adjusted_schedule = pd.concat([adjusted_schedule, emergency_entry], 
    ignore_index=True).sort_values(by='Appointment Time')
    
    # Recalculate appointment times
    for i in range(1, len(adjusted_schedule)):
        adjusted_schedule.iloc[i, 1] = 
        max(adjusted_schedule.iloc[i]['Appointment Time'], 
        adjusted_schedule.iloc[i-1]['Appointment Time'] 
        + timedelta(minutes=adjusted_schedule.iloc[i-1]['Duration']))
    return adjusted_schedule

# Example usage
daily_schedule = generate_daily_schedule()
print("Original Schedule:")
print(daily_schedule)

adjusted_schedule = adjust_schedule(daily_schedule)
print("\nAdjusted Schedule After Real-time Events:")
print(adjusted_schedule)

```

# 11. Title: HealthCare Disease Classification


```python


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generate random health data for 100 patients
np.random.seed(42)  # for reproducibility

data = {
    'Age': np.random.randint(18, 70, 100),
    'BMI': np.random.uniform(18.0, 40.0, 100),
    'BloodPressure': np.random.randint(80, 180, 100),
    'Cholesterol': np.random.choice(['Normal', 'High', 'Very High'], 100),
    'Disease': np.random.choice([0, 1], 100)  # 0: No Disease, 1: Disease
}

health_data = pd.DataFrame(data)

print(health_data.head())

def health_disease_classifier(data):
    # Separate features (X) and target variable (y)
    X = data.drop('Disease', axis=1)
    y = data['Disease']

    # Convert categorical variables to dummy/indicator variables
    X = pd.get_dummies(X, columns=['Cholesterol'], drop_first=True)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, random_state=42)

    # Initialize the Random Forest classifier
    clf = RandomForestClassifier(random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

# Use the function with the generated health data
health_disease_classifier(health_data)


```

# 12. Title: Develop a machine learning model that predicts the probability of diabetes based on symptoms and patient data.


```python
  import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Step 1: Generate synthetic patient data
np.random.seed(42)  # Ensure reproducibility
n_samples = 1000
data = {
    'Age': np.random.randint(20, 80, n_samples),
    'BMI': np.random.uniform(18.5, 40, n_samples),
    'Glucose': np.random.randint(70, 200, n_samples),
    'BloodPressure': np.random.randint(80, 120, n_samples),
    'Diabetes': np.random.choice([0, 1], n_samples)
}
df = pd.DataFrame(data)

# Step 2: Data Preprocessing
features = df[['Age', 'BMI', 'Glucose', 'BloodPressure']]
target = df['Diabetes']
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 3: Train a Logistic Regression Model
X_train, X_test, y_train, y_test = train_test_split(features_scaled,
  target, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Predict Diabetes Probabilities
predictions_proba = model.predict_proba(X_test)[:, 1]  
  # Get probabilities of the positive class
predictions = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, predictions))

# Example: Predicting for a new patient
new_patient = scaler.transform([[50, 25, 150, 85]])  # Example patient data
new_patient_proba = model.predict_proba(new_patient)[0, 1]
print(f"New patient diabetes probability: {new_patient_proba:.4f}")

```  


# 13. Title: Heart Disease Prediction

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Generate random heart disease data for 300 individuals
np.random.seed(456)  # for reproducibility

data = {
    'Age': np.random.randint(30, 75, 300),
    'Gender': np.random.choice(['Male', 'Female'], 300),
    'Cholesterol': np.random.choice(['Normal', 'High', 'Very High'], 300),
    'MaxHeartRate': np.random.randint(100, 200, 300),
    'ExerciseAngina': np.random.choice([0, 1], 300),  
    # 0: No angina, 1: Exercise-induced angina
    'HeartDisease': np.random.choice([0, 1], 300)  
    # 0: No Heart Disease, 1: Heart Disease
}

heart_disease_data = pd.DataFrame(data)

print(heart_disease_data.head())



def heart_disease_predictor(data):
    # Encode categorical variables
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    data['Cholesterol'] = le.fit_transform(data['Cholesterol'])

    # Separate features (X) and target variable (y)
    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, random_state=42)

    # Initialize the Logistic Regression model
    model = LogisticRegression(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(confusion)

# Use the function with the generated heart disease data
heart_disease_predictor(heart_disease_data)


```

# 14. Title: Predictive Modeling for Disease Diagnosis

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

# Function to generate random patient data
def generate_patient_data(num_patients):
    data = []
    for _ in range(num_patients):
        age = random.randint(20, 90)
        bmi = round(random.uniform(18.5, 40.0), 1)
        glucose = random.randint(70, 200)
        bp = random.randint(90, 180)
        disease = random.randint(0, 1)
        data.append([age, bmi, glucose, bp, disease])
    return pd.DataFrame(data, columns=["Age", "BMI", "Glucose",
    "Blood_Pressure", "Disease"])

# Logistic Regression Model
def logistic_regression_model(data):
    # Splitting the dataset
    X = data[["Age", "BMI", "Glucose", "Blood_Pressure"]]
    y = data["Disease"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, random_state=42)

    # Creating and training the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predicting and evaluating the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

# Example usage
patient_data = generate_patient_data(100)
model, model_accuracy = logistic_regression_model(patient_data)
print(f"Model Accuracy: {model_accuracy:.2f}")

# Predicting for a new patient
new_patient = pd.DataFrame([[45, 26.5, 150, 130]], columns=["Age", "BMI", "Glucose", "Blood_Pressure"])
prediction = model.predict(new_patient)
print(f"New Patient Prediction (1 for Disease, 0 for No Disease): {prediction[0]}")


```


# 15. Title: Cardiovascular Disease Risk Prediction

```python

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Generate synthetic data
np.random.seed(42)  # For reproducibility
n_samples = 1000
data = {
    'Age': np.random.randint(18, 80, size=n_samples),
    'Gender': np.random.choice(['Male', 'Female'], size=n_samples),
    'Systolic_BP': np.random.randint(110, 180, size=n_samples),
    'Diastolic_BP': np.random.randint(70, 120, size=n_samples),
    'Cholesterol': np.random.choice(['Normal', 'Above Normal',
    'Well Above Normal'], size=n_samples),
    'Diabetes': np.random.choice([0, 1], size=n_samples),
    'Smoking': np.random.choice([0, 1], size=n_samples),
    'Physical_Activity': np.random.choice(['Low', 'Moderate',
    'High'], size=n_samples),
    'CVD_Risk': np.random.choice([0, 1], size=n_samples) 
    # This is a simplified risk factor
}

df = pd.DataFrame(data)

# Preprocessing
label_encoders = {}
for column in ['Gender', 'Cholesterol', 'Physical_Activity']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Splitting the dataset
X = df.drop('CVD_Risk', axis=1)
y = df['CVD_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

```

# 17. Title: Type 2 Diabetes Mellitus Risk Prediction


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(0)  # Ensure reproducibility
n = 500  # Number of samples

data = {
    'Age': np.random.randint(20, 70, n),
    'BMI': np.random.uniform(18.5, 40, n),
    'Glucose_Level': np.random.uniform(70, 200, n),
    'Insulin': np.random.uniform(2.6, 24.9, n),
    'Family_History': np.random.choice(['Yes', 'No'], n),
    'Physical_Activity': np.random.choice(['Sedentary', 'Low', 'Moderate', 'High'], n),
    'Blood_Pressure': np.random.randint(70, 180, n),
    'Target': np.random.choice([0, 1], n)
}

df = pd.DataFrame(data)

# Encode categorical data
df['Family_History'] = df['Family_History'].map({'Yes': 1, 'No': 0})
df['Physical_Activity'] = df['Physical_Activity'].map({'Sedentary': 0,
'Low': 1, 'Moderate': 2, 'High': 3})

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Target', axis=1))
X = scaled_features
y = df['Target'].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3, random_state=0)

# Train a Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)

# Predict and evaluate
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model accuracy: {accuracy:.2f}')

```


# 18. Title: Alzheimer's Disease Progression Prediction


```python

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Function to generate synthetic data
def generate_data(n_samples=1000):
    np.random.seed(42)  # For reproducibility
    data = {
        'Age': np.random.randint(50, 90, n_samples),
        'Gender': np.random.randint(0, 2, n_samples),
        'Education': np.random.randint(8, 21, n_samples),
        'Socioeconomic_Status': np.random.choice(['Low',
        'Medium', 'High'], n_samples),
        'MMSE_Score': np.random.randint(0, 31, n_samples),
        'ADAS11_Score': np.random.uniform(5, 35, n_samples),
        'Genetic_Markers': np.random.randint(0, 2, n_samples),
        'Progression_Rate': np.random.uniform(0, 1, n_samples)  # Continuous target
    }
    df = pd.DataFrame(data)
    return df

# Function to preprocess data and run prediction model
def predict_progression(df):
    # One-hot encode categorical data
    encoder = OneHotEncoder(sparse=False)
    socioeconomic_status_encoded = encoder.fit_transform(df[['Socioeconomic_Status']])
    socioeconomic_status_df = pd.DataFrame(socioeconomic_status_encoded,
    columns=encoder.get_feature_names_out(['Socioeconomic_Status']))
    df = pd.concat([df.drop('Socioeconomic_Status', axis=1),
    socioeconomic_status_df], axis=1)
    
    # Split the dataset
    X = df.drop('Progression_Rate', axis=1)
    y = df['Progression_Rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, random_state=42)
    
    # Train a Random Forest Regressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = regressor.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse:.4f}')
    return regressor

# Generate synthetic data
df = generate_data()

# Preprocess data and predict Alzheimer's Disease progression
model = predict_progression(df)

```

# 19. Title: Chronic Kidney Disease Risk Prediction


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Function to generate synthetic data
def generate_ckd_data(n=500):
    np.random.seed(99)  # Ensuring reproducibility
    data = {
        'Age': np.random.randint(20, 80, n),
        'Blood_Pressure': np.random.randint(70, 180, n),
        'Blood_Glucose': np.random.uniform(70, 200, n),
        'BMI': np.random.uniform(18, 35, n),
        'Tobacco_Use': np.random.choice([0, 1], n),
        'History_of_Hypertension': np.random.choice([0, 1], n),
        'Serum_Creatinine': np.random.uniform(0.5, 1.5, n),
        'Target': np.random.choice([0, 1], n)
    }
    return pd.DataFrame(data)

# Function to preprocess, train, and evaluate the model
def ckd_risk_prediction(df):
    # Normalize continuous variables
    scaler = MinMaxScaler()
    continuous_features = ['Age', 'Blood_Pressure', 
    'Blood_Glucose', 'BMI', 'Serum_Creatinine']
    df[continuous_features] = scaler.fit_transform(df[continuous_features])
    
    # Splitting dataset
    X = df.drop('Target', axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, random_state=99)
    
    # Train Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=99)
    gbc.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = gbc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.4f}')
    return gbc

# Generate synthetic CKD data
ckd_df = generate_ckd_data()

# Preprocess data, train and evaluate CKD risk prediction model
ckd_model = ckd_risk_prediction(ckd_df)


```

# 20. Title: Asthma Exacerbation Risk Prediction

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Function to generate synthetic data for asthma risk prediction
def generate_asthma_data(n=1000):
    np.random.seed(7)  # Seed for reproducibility
    data = {
        'Age': np.random.randint(10, 80, n),
        'Gender': np.random.randint(0, 2, n),
        'Pollen_Count': np.random.randint(0, 5000, n),
        'AQI': np.random.randint(0, 500, n),
        'Humidity': np.random.uniform(30, 100, n),
        'Temperature': np.random.uniform(-5, 40, n),
        'Previous_Asthma_Attacks': np.random.randint(0, 10, n),
        'Medication_Adherence': np.random.randint(50, 100, n),
        'Target': np.random.choice([0, 1], n)
    }
    return pd.DataFrame(data)

# Function to preprocess data and run the asthma risk prediction model
def asthma_risk_prediction(df):
    # Scale the features
    scaler = StandardScaler()
    features = df.drop('Target', axis=1)
    scaled_features = scaler.fit_transform(features)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(scaled_features,
    df['Target'], test_size=0.2, random_state=7)
    
    # SVM model for classification
    svm_model = SVC(kernel='rbf', C=1.0, random_state=7)
    svm_model.fit(X_train, y_train)
    
    # Predictions and evaluation
    predictions = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")
    return svm_model

# Generate synthetic data
asthma_data = generate_asthma_data()

# Preprocess, train, and evaluate the model
asthma_model = asthma_risk_prediction(asthma_data)


```


# 21. Title: Osteoporosis Risk Prediction

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class OsteoporosisRiskPredictor:
    def __init__(self, n_samples=500):
        self.n_samples = n_samples
        self.data = None
        self.model = RandomForestClassifier(random_state=42)
        self.features = None
        self.target = None

    def generate_data(self):
        np.random.seed(42)
        self.data = pd.DataFrame({
            'Age': np.random.randint(20, 90, self.n_samples),
            'Gender': np.random.choice([0, 1], self.n_samples),
            'Calcium_Intake': np.random.randint(200, 2500, self.n_samples),
            'Vitamin_D_Intake': np.random.randint(200, 4000, self.n_samples),
            'Physical_Activity_Level': np.random.choice([0, 1, 2], self.n_samples),
            'Smoking_Status': np.random.choice([0, 1], self.n_samples),
            'BMI': np.random.uniform(18.5, 40, self.n_samples),
            'Family_History': np.random.choice([0, 1], self.n_samples),
            'Target': np.random.choice([0, 1], self.n_samples),
        })

    def preprocess_data(self):
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.data.drop('Target', axis=1))
        self.target = self.data['Target'].values

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.features,
        self.target, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'Model Accuracy: {accuracy:.4f}')

# Using the class to predict osteoporosis risk
predictor = OsteoporosisRiskPredictor()
predictor.generate_data()
predictor.preprocess_data()
X_train, X_test, y_train, y_test = predictor.split_data()
predictor.train_model(X_train, y_train)
predictor.evaluate_model(X_test, y_test)

```

# 22. Title: Diabetes Risk Prediction

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Function to simulate data
def simulate_data(n):
    np.random.seed(0)
    data = {
        'Age': np.random.randint(20, 70, n),
        'BMI': np.random.uniform(18, 40, n),
        'BloodPressure': np.random.randint(70, 180, n),
        'GlucoseLevel': np.random.randint(70, 200, n),
        'InsulinLevel': np.random.uniform(15, 300, n),
        'FamilyHistory': np.random.randint(0, 2, n),
        'Outcome': np.random.randint(0, 2, n)
    }
    return pd.DataFrame(data)

# Function to preprocess data
def preprocess_data(df):
    features = df.drop('Outcome', axis=1)
    labels = df['Outcome']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return train_test_split(features_scaled, labels, test_size=0.2, random_state=0)

# Function to train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    return model

# Function for prediction
def predict_diabetes(model, scaler, input_data):
    scaled_data = scaler.transform([input_data])
    prediction = model.predict(scaled_data)
    return "Diabetes Risk: High" if prediction == 1 else "Diabetes Risk: Low"

# Main execution
df = simulate_data(1000)
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
model = train_model(X_train, y_train)
print(predict_diabetes(model, scaler, [45, 28.5, 130, 150, 85, 1]))  # Example prediction

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

```


# 23. Title: Sleep Disorder Risk Prediction

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(n=1000):
    np.random.seed(123)
    data = {
        'Age': np.random.randint(18, 65, n),
        'BMI': np.random.uniform(18.5, 40, n),
        'Stress_Level': np.random.randint(0, 11, n),
        'Caffeine_Consumption': np.random.randint(0, 5, n),
        'Screen_Time': np.random.uniform(1, 5, n),
        'Sleep_Duration': np.random.uniform(4, 10, n),
        'Physical_Activity': np.random.choice(['low', 'moderate', 'high'], n),
        'Environmental_Noise': np.random.choice(['low', 'moderate', 'high'], n),
        'Target': np.random.choice([0, 1], n)
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    # Convert categorical data to numeric
    df = pd.get_dummies(df, columns=['Physical_Activity', 'Environmental_Noise'])
    # Scale data
    scaler = StandardScaler()
    features = df.drop('Target', axis=1)
    features_scaled = scaler.fit_transform(features)
    target = df['Target']
    return features_scaled, target

def train_test_split_and_evaluate(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123)
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=123)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

# Main execution flow
df = generate_synthetic_data()
features, target = preprocess_data(df)
train_test_split_and_evaluate(features, target)

```

# 25. Title: Heart Disease Prediction

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class DataSimulator:
    def __init__(self, size):
        self.size = size

    def simulate(self):
        np.random.seed(0)
        data = {
            'Age': np.random.randint(30, 80, self.size),
            'Cholesterol': np.random.randint(100, 300, self.size),
            'RestingBP': np.random.randint(70, 180, self.size),
            'MaxHeartRate': np.random.randint(70, 200, self.size),
            'ChestPain': np.random.randint(0, 4, self.size),
            'HeartDisease': np.random.randint(0, 2, self.size)
        }
        return pd.DataFrame(data)

class Preprocessor:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()

    def preprocess(self):
        X = self.data.drop('HeartDisease', axis=1)
        y = self.data['HeartDisease']
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=0)

class HeartDiseasePredictor:
    def __init__(self):
        self.model = SVC()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, features):
        return self.model.predict([features])[0]

# Simulation and Preprocessing
simulator = DataSimulator(1000)
df = simulator.simulate()
preprocessor = Preprocessor(df)
X_train, X_test, y_train, y_test = preprocessor.preprocess()

# Model Training and Prediction
predictor = HeartDiseasePredictor()
predictor.train(X_train, y_train)

# Example Prediction
example_features = [50, 250, 130, 150, 2]  # Example patient data
prediction = predictor.predict(example_features)
print("Heart Disease Prediction:", "Positive" if prediction == 1 else "Negative")

# Model Evaluation
y_pred = [predictor.predict(x) for x in X_test]
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

```




















