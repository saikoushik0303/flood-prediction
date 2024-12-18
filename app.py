from flask import Flask,render_template,request
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn .model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import pickle
import joblib


import mysql.connector
mydb = mysql.connector.connect(host='localhost',user='root',password='1234',port='3306',database='flood')
cur = mydb.cursor()


app = Flask(__name__)
app.config['upload folder']= r'upload'
@app.route('/')
def home():
    return render_template('index.html')

def cleaning(file):
    data = file[['Mar-May','Jun-Sep','10days_june','increased Rainfall','flood']]
    return data
def spliting(file):
    global X,y
    X = file.drop(['flood'],axis = 1)
    y = file.flood
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 9)
    return x_train,x_test,y_train,y_test

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == "POST":
        email = request.form['email']
        psw = request.form['password']
        sql = "SELECT * FROM prd WHERE Email=%s and Password=%s"
        val = (email, psw)
        cur = mydb.cursor()
        cur.execute(sql, val)
        results = cur.fetchall()
        mydb.commit()
        if len(results) >= 1:
            return render_template('uhome.html', msg='login succesful')
        else:
            return render_template('login.html', msg='Invalid Credentias')

    return render_template('login.html')



@app.route('/registration',methods=['GET','POST'])
def registration():

    if request.method == "POST":
        print('a')
        name = request.form['name']
        print(name)
        email = request.form['email']
        pws = request.form['psw']
        print(pws)
        cpws = request.form['cpsw']
        if pws == cpws:
            sql = "select * from prd"
            print('abcccccccccc')
            cur = mydb.cursor()
            cur.execute(sql)
            all_emails = cur.fetchall()
            mydb.commit()
            all_emails = [i[2] for i in all_emails]
            if email in all_emails:
                return render_template('registration.html', msg='success')
            else:
                sql = "INSERT INTO prd(name,email,password) values(%s,%s,%s)"
                values = (name, email, pws)
                cur.execute(sql, values)
                mydb.commit()
                cur.close()
                return render_template('registration.html', msg='success')
        else:
            return render_template('registration.html', msg='password not matched')

    return render_template('registration.html')

@app.route('/uhome')
def uhome():
    return render_template('uhome.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == "POST":
        file = request.files['file']
        global df
        print(file)
        filetype = os.path.splitext(file.filename)[1]
        print(filetype)
        if filetype == '.csv':
            UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload')
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            print("File saved at:", path)
            
            df = pd.read_csv(path)
            if 'Unnamed: 0' in df.columns:
                df.drop(['Unnamed: 0'], axis=1, inplace=True)
                
            return render_template('view.html', col_name=df.columns, row_val=list(df.values.tolist()))
        else:
            return render_template('upload.html', msg='Invalid file format. Please upload a CSV file.')
    return render_template('upload.html')
@app.route('/model', methods=['POST', "GET"])
def model():
    if request.method == 'POST':
        clean_data = cleaning(df)
        x_train, x_test, y_train, y_test = spliting(clean_data)
        model = int(request.form['model'])
        
        # Ensure the 'models' directory exists
        os.makedirs('models', exist_ok=True)
        
        if model == 1:  # KNN
            knn = KNeighborsClassifier()
            score = cross_val_score(knn, X, y, cv=5)
            print(score)
            print(score.mean())
            kn = knn.fit(x_train, y_train)
            file = 'models/knn.h5'
            pickle.dump(kn, open(file, 'wb'))
            pre = kn.predict(x_test)
            scores = accuracy_score(y_test, pre)
            print(scores)
            return render_template('model.html', msg='success', score=scores, Selected='KNN')
        
        if model == 2:  # Decision Tree Classifier
            dt = DecisionTreeClassifier()
            score = cross_val_score(dt, X, y, cv=5)
            print(score)
            print(score.mean())
            d = dt.fit(x_train, y_train)
            file = 'models/dt.h5'
            pickle.dump(d, open(file, 'wb'))
            pre = d.predict(x_test)
            scores = accuracy_score(y_test, pre)
            print(scores)
            return render_template('model.html', msg='success', score=scores, Selected='Decision Tree Classifier')
        
        if model == 3:  # Logistic Regression
            lr = LogisticRegression()
            score = cross_val_score(lr, X, y, cv=5)
            print(score)
            print(score.mean())
            l = lr.fit(x_train, y_train)
            file = 'models/lr.h5'
            pickle.dump(l, open(file, 'wb'))
            pre = l.predict(x_test)
            scores = accuracy_score(y_test, pre)
            print(scores)
            return render_template('model.html', msg='success', score=scores, Selected='Logistic Regression')
        
        if model == 4:  # XGBoost
            xg = xgb.XGBClassifier()
            score = cross_val_score(xg, X, y, cv=5)
            print(score)
            print(score.mean())
            x = xg.fit(x_train, y_train)
            file = 'models/xgb.h5'
            pickle.dump(x, open(file, 'wb'))  # Save as .h5
            joblib.dump(x, 'models/xgb.joblib')  # Save as .joblib
            
            pre = x.predict(x_test)
            scores = accuracy_score(y_test, pre)
            print(scores)
            return render_template('model.html', msg='success', score=scores, Selected='XGBoost')

    return render_template('model.html')
@app.route('/prediction',methods = ["POST","GET"])

def prediction():
    if request.method == "POST":
        a = float(request.form['f1'])
        b = float(request.form['f2'])
        c = float(request.form['f3'])
        d = float(request.form['f4'])
        values = [[float(a),float(b),float(c),float(d)]]
        filenam = r'models/xgb.h5'
        model = pickle.load(open(filenam,'rb'))
        model = joblib.load('models/xgb.joblib')

        ex = pd.DataFrame(values,columns=X.columns)
        pred = model.predict(ex)
        print(pred)
        return render_template('prediction.html',res = pred)
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)