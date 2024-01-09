# ml_model.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'],format='mixed') 
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day_name()
    df['month'] = df['trans_date_trans_time'].dt.month
    df.sort_values(['cc_num', 'trans_date_trans_time'])
    df['hours_diff_bet_trans']=((df.groupby('cc_num')[['trans_date_trans_time']].diff())/np.timedelta64(1,'h'))
    df.loc[df['hours_diff_bet_trans'].isna(),'hours_diff_bet_trans'] = 0
    df['hours_diff_bet_trans'] = df['hours_diff_bet_trans'].astype(int) 
    df['day'] = df['trans_date_trans_time'].dt.weekday
    for col in ['city','job','merchant', 'category']:
        df[col] = WOEEncoder().fit_transform(df[col],df['is_fraud']) 
    # Perform any necessary preprocessing on your input data
    # For example, encoding categorical variables
    
    # Add more preprocessing steps as needed
    return df

def train_model(df):
    x = df.drop(columns='is_fraud', axis=1)
    y = df['is_fraud']
    smt = SMOTE() 
    X1, Y1 = smt.fit_resample(x, y)
    columns = x.columns.tolist() + ['is_fraud']
    df2= pd.DataFrame(np.column_stack([X1,Y1]), columns=columns)
    # Split the data into features (X) and target variable (y)
    

    # Split the data into training and testing sets
    
    X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.2, stratify=Y1, random_state=5) 

    # Create a RandomForestClassifier (replace with your model of choice)
    DecisionTree=DecisionTreeClassifier()
    DecisionTree.fit(X_train,Y_train)
    y_pred_dt = DecisionTree.predict(X_test)
    accuracy_dt = accuracy_score(Y_test, y_pred_dt)



    # Print the accuracy for demonstration purposes
    print(f"Model Accuracy: {accuracy_score(Y_test, y_pred_dt)}")

    return DecisionTree

def predict(DecisionTree, input_data):
    # Perform any necessary preprocessing on the input data
    input_df = pd.DataFrame([input_data])
    input_df = preprocess_data(input_df)

    # Make predictions using the trained model
    prediction = DecisionTree.predict(input_df)[0]

    # Return the prediction result
    return 'Fraudulent' if prediction == 1 else 'Not Fraudulent'
