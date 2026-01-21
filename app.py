import gradio as gr
import joblib
import pandas as pd
import numpy as np


model_pipeline = joblib.load('diabetes_pipeline.pkl')

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):

    input_df = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }])
    

    cols_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_clean:
        if input_df.loc[0, col] == 0:
            input_df.loc[0, col] = np.nan

    prediction = model_pipeline.predict(input_df)[0]
    probs = model_pipeline.predict_proba(input_df)[0]

    label = "Diabetic" if prediction == 1 else "Not Diabetic"
    confidence = probs[1] if prediction == 1 else probs[0]
    
    return f"{label} (Confidence: {confidence:.2f})"

interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies", value=1, minimum=0, maximum=20, step=1, 
                  info="Number of times pregnant (Range: 0-17)"),
        
        gr.Number(label="Glucose Level", value=120, minimum=0, maximum=300, 
                  info="Plasma glucose concentration (Range: 0-199 mg/dL)"),
        
        gr.Number(label="Blood Pressure", value=70, minimum=0, maximum=150, 
                  info="Diastolic blood pressure (Range: 0-122 mm Hg)"),
        
        gr.Number(label="Skin Thickness", value=20, minimum=0, maximum=100, 
                  info="Triceps skin fold thickness (Range: 0-99 mm)"),
        
        gr.Number(label="Insulin Level", value=80, minimum=0, maximum=900, 
                  info="2-Hour serum insulin (Range: 0-846 mu U/ml)"),
        
        gr.Number(label="BMI", value=30.0, minimum=0, maximum=70, 
                  info="Body mass index (Range: 0.0-67.1)"),
        
        gr.Number(label="Diabetes Pedigree Function", value=0.5, minimum=0, maximum=3.0, 
                  info="Diabetes likelihood based on family history (Range: 0.08-2.42)"),
        
        gr.Number(label="Age", value=33, minimum=1, maximum=120, step=1, 
                  info="Age in years (Range: 21-81)")
    ],
    outputs="text",
    title="Diabetes Prediction System",
    description="Enter patient details below. The 'Range' info shows typical values found in the training dataset."
)

if __name__ == "__main__":
    interface.launch()