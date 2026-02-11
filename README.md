Heart Disease Predictor - Predicts whether the patient has heart disease or not. 

Alzheimer Predictor - Predicts whether the patient has alzheimer or not. 

The 2 predictor apps above are combined into an integrated webpage app. You can access it using this link:  [https://mlops-assignment-wyy3clyzzrfmxhzjvvhbxa.streamlit.app/]

The website is hosted/deployed through Streamlit Community Cloud.

The app's target user is for **clinic staff**, not patients themselves. Clinic staff is able to use the app to check whether a patient might have heart disease or alzheimer's. Do note that the app predictors should only support screening and decision support. **Clinical judgment and further confirmatory tests are required**.
### How to use each app:
Heart Disease Predictor (Lock Yan Fong, Brendon) - Choose either single prediction or scroll down the webpage for batch prediction. *For single prediction*: Fill in details (e.g., Age, Gender, Cholesterol Total, ST_Slope and resting Diastolic Blood Pressure). Click on the Predict button to view whether the patient has heart disease or not, accompanied by the prediction confidence score. *For batch prediction*: Assuming the clinic staff has all the columns of data, upload the csv file, and the prediction will happen in real time. It shows the summary of how many heart disease and non-heart disease patients there are in the uploaded csv file records, the average prediction score of all the records given and a table preview of the first 10 rows. Below it is a download csv file button to download the full prediction csv file.

Alzheimer Predictor (Wong Kah Ann) - Choose either single prediction or scroll down the webpage for batch prediction. *For single prediction*: Fill in details (e.g., Age, DietQuality, MMSE Score, PhysicalActivity, etc) and toggle the radio buttons for Memory Complaints or Behavioural Complaints if prevalent. Click the Predict button to view whether the patient has Alzheimer's or not, along with the prediction likelihood percentage. *For batch prediction*: Assuming the clinic staff has all the columns of data, upload the csv file, and the prediction will happen in real time. Below it is a download csv file button to download the full prediction csv file.

We have also uploaded all the files from Task 1 to Task 3 to our individual folders in the GitHub repository.

Github Repo Link: [https://github.com/WongKahAnn/MLOps-Assignment]
 
ci tester///
