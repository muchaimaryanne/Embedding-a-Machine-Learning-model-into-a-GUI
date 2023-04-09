import gradio as gr
import pandas as pd
import pickle



# Load the pipeline using pickle
with open('C:/Users/mucha/Documents/Azubi Africa Career Accelerator/Month 4/Career_Accelerator_P4-ML_apps/gradio_project/pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)



def predict(TotalCharges, tenure, MonthlyCharges, PaymentMethod_Electronic_check, InternetService_Fiber_optic, Contract_Two_year, gender_Male, OnlineSecurity_Yes, PaperlessBilling_Yes, Partner_Yes):
    input_list = [TotalCharges, tenure, MonthlyCharges, PaymentMethod_Electronic_check, InternetService_Fiber_optic, Contract_Two_year, gender_Male, OnlineSecurity_Yes, PaperlessBilling_Yes, Partner_Yes]
    
    # Check if any of the input values are strings ('Yes' or 'No')
    for i, val in enumerate(input_list):
        if isinstance(val, str):
            if val.lower() == "yes" or "male":
                input_list[i] = True
            elif val.lower() == "no" or "female":
                input_list[i] = False
            elif not isinstance(val, (int, float)):
                return "Error: Input value must be 'Yes' or 'No' or numeric"
            else:
                input_list[i] = float(val)
        # Convert 'Yes'/'No' values to boolean
        input_df = pd.DataFrame([input_list], columns=['TotalCharges', 'tenure', 'MonthlyCharges', 'PaymentMethod_Electronic_check', 'InternetService_Fiber_optic', 'Contract_Two_year', 'gender_Male', 'OnlineSecurity_Yes', 'PaperlessBilling_Yes', 'Partner_Yes'])
    
    
    prediction = pipeline['model'].predict(input_df)
    pred_proba = pipeline['model'].predict_proba(input_df)[0]
    
    # Create a dataframe to display the prediction and confidence score
    pred_df = pd.DataFrame({'class': pipeline.classes_,
                            'confidence': pred_proba})
    # Sort the dataframe by confidence score in descending order
    pred_df = pred_df.sort_values('confidence', ascending=False)
    return pred_df

# Define the Gradio interface
input_interface = [
    
        gr.inputs.Slider(minimum=0, maximum=2000, default=50, label="Total Charges"),
        gr.inputs.Slider(minimum=0, maximum=50, default=5, label="Tenure"),
        gr.inputs.Slider(minimum=0, maximum=500, default=20, label="Monthly Charges"),
        gr.inputs.Dropdown(choices=['Yes','No'], label="Electronic check Payment"),
        gr.inputs.Dropdown(choices=['Yes','No'], label="Fiber optic Internet Service"),
        gr.inputs.Dropdown(choices=['Yes','No'], label="Two Year Contract"),
        gr.inputs.Dropdown(choices=['Male','Female'], label="Gender"),
        gr.inputs.Dropdown(choices=['Yes','No'], label="Online Security"),
        gr.inputs.Dropdown(choices=['Yes','No'], label="Paperless Billing"),
        gr.inputs.Dropdown(choices=['Yes','No'], label="Partner")
]
    
label=pipeline['label']

output_table = gr.outputs.Dataframe(headers=['churn', 'confidence'],type='pandas')

# Create the Gradio app

gr.Interface(fn=predict, inputs=input_interface, outputs=output_table,title="Customer Churn Prediction").launch()
