from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime
import os
import time
import requests
import subprocess

# Define default arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 3),  # Start date of your DAG
    'retries': 1,
}

# FastAPI Endpoint and settings
API_ENDPOINT = "http://localhost:8000"

# Define the DAG
with DAG(
    dag_id='drift_orchestration',  # Unique name for the DAG
    default_args=default_args,
    description='Airflow DAG to track and orchestrate the drift generation and storage on cloud platforme;',
    schedule_interval='@daily',  # Run the DAG every day
    catchup=False,  # Only run the most recent schedule
) as dag:

    # Task 1: Verify if the user is an admin (via FastAPI authentication)
    def check_admin_user():
        url = f"{API_ENDPOINT}/token"
        # Dummy login data for example purposes
        login_data = {"username": "admin", "password": "adminpass"}
        # response = requests.post(url, data=login_data)
        response = requests.post(
            "http://127.0.0.1:8000/token/",
            data=login_data
        )
        
        # if response.status_code != 200:
        if response["status_code"] != 200:
            raise Exception("Login failed or user is not an admin.")
        else:
            print("Admin login successful.")
    
    task_1 = PythonOperator(
        task_id='admin_logging',
        python_callable=check_admin_user
    )

    # Task 2: Generate the drift report
    def generate_drift_report_task():
        url = f"{API_ENDPOINT}/monitoring"
        response = requests.get(url)
        
        if response.status_code == 200:
            print("Drift report successfully generated and served.")
        else:
            raise Exception("Error generating drift report.")
    
    task_2 = PythonOperator(
        task_id='data_drift_report_generation',
        python_callable=generate_drift_report_task
    )
    

    # Task 3: Compare current and previous drift reports (via dvc)
    def compare_drift_reports():
        report_path = "../../dataops/brain_data/drift_seg_report.html"
        
        # Fetch the last report from DVC (version control)
        subprocess.run(["dvc", "pull", report_path], check=True)

        if os.path.exists(report_path):
            print("Comparing current report with the previous version...")
            # Logic to compare files (e.g., hash comparison or file diff)
            current_hash = subprocess.check_output(["sha256sum", report_path]).split()[0]
            prev_hash = subprocess.check_output(["dvc", "status"]).splitlines()[-1]  # Simulate getting previous hash

            if current_hash == prev_hash:
                print("No drift detected. The DAG will stop here.")
                return 'skip'  # This indicates no further actions required
            else:
                print("Drift detected.")
        else:
            raise Exception(f"File {report_path} not found.")
    
    task_3 = PythonOperator(
        task_id='drift_difference',
        python_callable=compare_drift_reports
    )

    # Task 4: Track and push the new report (only if drift is detected)
    def save_and_push_report():
        report_path = "../../dataops/brain_data/drift_seg_report.html"

        # Track new drift report with DVC
        subprocess.run(["dvc", "add", report_path], check=True)
        subprocess.run(["git", "add", report_path], check=True)
        subprocess.run(["git", "commit", "-m", "New drift report detected"], check=True)
        subprocess.run(["dvc", "push"], check=True)

        print("New drift report successfully tracked and pushed to Azure.")
    
    task_4 = PythonOperator(
        task_id='drift_tracking',
        python_callable=save_and_push_report
    )

    # Define task dependencies (task_1 -> task_2 -> task_3 -> task_4)
    task_1 >> task_2 >> task_3 >> task_4

    
    
    
    
    
    
    
    
    
    # # Task 3: Drift Comparison with Conditional Branching
    # def drift_comparison():
    #     print("Check difference between this and the last drift generated")
    #     # Here you would normally check the actual drift difference.
    #     drift_detected = True  # Example result, replace with actual condition
        
    #     if drift_detected:
    #         return 'drift_tracking'  # Continue to save drift if there's a difference
    #     else:
    #         return 'end_dag'  # End the DAG if there's no difference

    # task_3 = BranchPythonOperator(
    #     task_id='drift_difference',
    #     python_callable=drift_comparison
    # )
    
    # def save_track_drift_report(ti):
    #     print("save the new file of drift by tracking it with dvc commeand and push it on AZure as usual.")
 
    # # Task 4: Save and push the drift
    # task_4 = PythonOperator(
    #     task_id='drift_tracking',
    #     python_callable=save_track_drift_report
    # )
    
    # # End Task (dummy operator to signify DAG completion)
    # end_task = DummyOperator(
    #     task_id='end_dag'
    # )
    
    # # Set task dependencies
    # task_1 >> task_2 >> task_3
    # task_3 >> [task_4, end_task]