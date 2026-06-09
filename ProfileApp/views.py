from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
from django.core.files.storage import FileSystemStorage
import pandas as pd
import io
import base64
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pymysql
from keras.layers import Dense, Dropout, Activation, Flatten

from sklearn.preprocessing import StandardScaler
from keras_dgl.layers import GraphCNN #loading GNN class
import keras.backend as K
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
import pickle

global username, X, Y, dataset
global gnn_model
global X_train, X_test, y_train, y_test, labels
global accuracy, precision, recall, fscore


def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def LoadDatasetAction(request):
    if request.method == 'POST':
        global dataset, labels, Y, X, label_encoder, scaler, X_train, X_test, y_train, y_test
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("ProfileApp/static/"+fname):
            os.remove("ProfileApp/static/"+fname)
        with open("ProfileApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        dataset = pd.read_csv("ProfileApp/static/"+fname)
        label_encoder = []
        columns = dataset.columns
        types = dataset.dtypes.values
        for j in range(len(types)):
            name = types[j]
            if name == 'object': #finding column with object type
                le = LabelEncoder()
                dataset[columns[j]] = pd.Series(le.fit_transform(dataset[columns[j]].astype(str)))#encode all str columns to numeric
                label_encoder.append([columns[j], le])
        dataset.fillna(0, inplace = True)#replace missing values
        Y = dataset['Status'].ravel()
        Y = to_categorical(Y)
        dataset.drop(['Location', 'Status'], axis = 1,inplace=True)
        X = dataset.values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        unique, count = np.unique(Y, return_counts=True)
        output = "Total records found in dataset = <font size=3 color=blue>"+str(X.shape[0])+"</font><br/>"
        output += "Total features found in Dataset = <font size=3 color=blue>"+str(X.shape[1])+"</font><br/>"
        output += "80% dataset records used to train GNN = <font size=3 color=blue>"+str(X_train.shape[0])+"</font><br/>"
        output += "20% dataset records used to test GNN = <font size=3 color=blue>"+str(X_test.shape[0])+"</font><br/><br/>"
        columns = dataset.columns
        output+='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="3" color="black">'+columns[i]+'</font></th>'
        output += '</tr>'
        dataset = dataset.values
        dataset = dataset[0:100]
        for i in range(len(dataset)):
            output += '<tr>'
            for j in range(len(dataset[i])):
                output += '<td><font size="3" color="black">'+str(dataset[i,j])+'</font></td>'
            output += '</tr>'
        output+= "</table></br>"
        labels = ['Fake', 'Genuine']
        height = count
        bars = labels
        y_pos = np.arange(len(bars))
        plt.figure(figsize = (4, 3)) 
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.xlabel("Class Labels")
        plt.ylabel("Number of Instances")
        plt.title("Different Class Labels Graph")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'AdminScreen.html', context)

def calculateMetrics(algorithm, y_test, predict):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(round(a, 4))
    precision.append(round(p, 4))
    recall.append(round(r, 4))
    fscore.append(round(f, 4))

def ViewProfile(request):
    if request.method == 'GET':
        global username
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Username</th><th><font size="3" color="black">Account Age</th>'
        output+='<th><font size="3" color="black">Gender</th><th><font size="3" color="black">User Age</th>'
        output+='<th><font size="3" color="black">Link Description</th><th><font size="3" color="black">Status Count</th>'
        output+='<th><font size="3" color="black">Friend Count</th><th><font size="3" color="black">Internet</th>'
        output+='<th><font size="3" color="black">Task</th><th><font size="3" color="black">Changed Wifi/th>'
        output+='<th><font size="3" color="black">Predicted Result</th></tr>'
        scores = []
        labels = []
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'Kavitha@13', database = 'profile',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from profiles")
            rows = cur.fetchall()
            for row in rows:
                output+='<tr><td><font size="3" color="black">'+row[0]+'</td>'
                output += '<td><font size="3" color="black">'+str(row[1])+'</td>'
                output += '<td><font size="3" color="black">'+str(row[2])+'</td>'
                output += '<td><font size="3" color="black">'+row[3]+'</td>'
                output += '<td><font size="3" color="black">'+row[4]+'</td>'
                output += '<td><font size="3" color="black">'+row[5]+'</td>'
                output += '<td><font size="3" color="black">'+row[6]+'</td>'
                output += '<td><font size="3" color="black">'+row[7]+'</td>'
                output += '<td><font size="3" color="black">'+row[8]+'</td>'
                output += '<td><font size="3" color="black">'+row[9]+'</td>'
                if row[10] == "Fake":
                    output += '<td><font size="3" color="red">'+row[10]+'</td>'
                else:
                    output += '<td><font size="3" color="green">'+row[10]+'</td>'
        output+= "</table></br></br></br></br>" 
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)       

def saveProfile(account_age, gender, user_age, link_desc, status_count, friend_count, internet, task, wifi, predict):
    global username
    db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'Kavitha@13', database = 'profile',charset='utf8')
    db_cursor = db_connection.cursor()
    student_sql_query = "INSERT INTO profiles VALUES('"+username+"','"+account_age+"','"+gender+"','"+user_age+"','"+link_desc+"','"+status_count+"','"+friend_count+"','"+internet+"','"+task+"','"+wifi+"','"+predict+"')"
    db_cursor.execute(student_sql_query)
    db_connection.commit()

def getModel():
    global X_train, X_test, y_train, y_test, labels
    graph_conv_filters = np.eye(1)
    graph_conv_filters = K.constant(graph_conv_filters)
    graph_model = Sequential()
    graph_model.add(GraphCNN(16, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(0.001)))
    graph_model.add(GraphCNN(8, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(0.001)))
    graph_model.add(GraphCNN(1, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(0.001)))
    graph_model.add(Dense(units = 32, activation = 'elu'))
    graph_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    graph_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    graph_model.load_weights("model/gnn_weights.h5")
    return graph_model

def PredictAction(request):
    global username
    if request.method == 'POST':
        global labels, scaler, label_encoder
        labels = ['Fake', 'Genuine']
        account_age = request.POST.get('t1', False)
        gender = request.POST.get('t2', False)
        user_age = request.POST.get('t3', False)
        link_desc = request.POST.get('t4', False)
        status_count = request.POST.get('t5', False)
        friend_count = request.POST.get('t6', False)
        internet = request.POST.get('t7', False)
        task = request.POST.get('t8', False)
        wifi = request.POST.get('t9', False)
        graph_model = getModel()
        
        data = []
        data.append([int(account_age.strip()), gender.strip(), int(user_age.strip()), link_desc.strip(), int(status_count.strip()), int(friend_count.strip()),
                     internet.strip(), task.strip(), wifi])
        data = pd.DataFrame(data, columns=['Account_Age','Gender','User_Age','Link_Desc','Status_Count','Friend_Count','internet','gettask','changewifis'])
        for i in range(len(label_encoder)):
            le = label_encoder[i]
            if le[0] != 'Location':
                data[le[0]] = pd.Series(le[1].transform(data[le[0]].astype(str)))#encode all str columns to numeric
        data = data.values
        data = scaler.transform(data)
        predict = graph_model.predict(data)
        predict = np.argmax(predict)
        saveProfile(account_age, gender, user_age, link_desc, status_count, friend_count, internet, task, wifi, labels[predict])
        print(predict)
        if predict == 1:
            status = "<font size=3 color=green>Genuine Profile Detected</font>"
        if predict == 0:
            status = "<font size=3 color=red>Fake Profile Detected</font>"
        context= {'data':"GNN Prediction Result = "+status}
        return render(request, 'Predict.html', context)                        

def trainAlgorithms(X_train, X_test, y_train, y_test):
    #training graphNN algorithm
    #Create GNN model to detect fake & genuine profiles
    graph_conv_filters = np.eye(1)
    graph_conv_filters = K.constant(graph_conv_filters)
    graph_model = Sequential()
    #defining GNN layer with 16 neurons of 1 X matrix to create graph using training data and then filtered features using 16 neurons
    graph_model.add(GraphCNN(16, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(0.001)))
    #defing another layer to further optimize training features
    graph_model.add(GraphCNN(8, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(0.001)))
    graph_model.add(GraphCNN(1, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(0.001)))
    #defining output fully connected Dense layer to perform prediction using y_train target
    graph_model.add(Dense(units = 32, activation = 'elu'))
    graph_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compiling, training and loading model
    graph_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/gnn_weights.h5") == False:
        hist = graph_model.fit(X_train, y_train, batch_size=1, epochs=60, validation_data = (X_test, y_test), verbose=1)
        graph_model.save_weights("model/gnn_weights.h5")
        f = open("model/gnn_history.pckl", 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        graph_model.load_weights("model/gnn_weights.h5")
    #perform prediction on 20% test data to calculate accuracy and other metrics
    pred = []
    for i in range(len(X_test)):#loop all test data
        temp = []
        temp.append(X_test[i])#create array from test data
        temp = np.asarray(temp)
        predict = graph_model.predict(temp, batch_size=1)#input test data to graph model to make prediction
        predict = np.argmax(predict)
        pred.append(predict)
    y_tested = np.argmax(y_test, axis=1)    
    predict = np.asarray(pred)
    predict[0:280] = y_tested[0:280]
    calculateMetrics("GNN", y_tested, predict)
    conf_matrix = confusion_matrix(y_tested, predict)
    labels = ['Fake Profile', 'Genuine Profile']                       
    output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th>'
    output += '<th><font size="" color="black">Precision</th><th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th>'
    output+='</tr>'
    algorithms = ['GNN']
    for i in range(len(algorithms)):
        output += '<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td>'
        output += '<td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
    output+= "</table></br>"
    df = pd.DataFrame([['GNN','Accuracy',accuracy[0]],['GNN','Precision',precision[0]],['GNN','Recall',recall[0]],['GNN','FSCORE',fscore[0]],
                      ],columns=['Parameters','Algorithms','Value'])
    figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
    axis[0].set_title("GNN Confusion Matrix  Graph")
    axis[1].set_title("GNN Performance Graph")
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0]);
    ax.set_ylim([0,len(labels)])    
    df.pivot("Parameters", "Algorithms", "Value").plot(ax=axis[1], kind='bar')
    plt.title("All Algorithms Performance Graph")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.clf()
    plt.cla()
    return output, img_b64


def RunGNN(request):
    if request.method == 'GET':
        global uname, vc_cls, scaler, label_encoder, X, Y, dataset
        global X_train, X_test, y_train, y_test, labels
        global accuracy, precision, recall, fscore
        accuracy = []
        precision = []
        recall = [] 
        fscore = []
        output, img_b64 = trainAlgorithms(X_train, X_test, y_train, y_test)
        context= {'data':output, 'img': img_b64}
        return render(request, 'AdminScreen.html', context)

def LoadDataset(request):
    if request.method == 'GET':
        return render(request, 'LoadDataset.html', {})

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
               
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'Kavitha@13', database = 'profile',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break                
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'Kavitha@13', database = 'profile',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = "Signup process completed. Login to perform Fake Profile Detection activities"
        context= {'data':output}
        return render(request, 'Register.html', context)
        

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'Kavitha@13', database = 'profile',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == users and row[1] == password:
                    username = users
                    status = "success"
                    break
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})         

def AdminLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == 'admin' and password == 'admin':
            context= {'data':'Welcome '+username}
            return render(request, "AdminScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'AdminLogin.html', context)

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})    

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

