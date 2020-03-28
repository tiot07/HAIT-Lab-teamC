import io
import csv
import pprint
import datetime
import os
import shutil
from flask import Flask, render_template, request, jsonify
from wtforms import Form, StringField, SubmitField, validators, ValidationError
import numpy as np
from sklearn.externals import joblib
from flask_table import Table, Col    

app = Flask(__name__)

allmemberlist = []
memberlist = []
attendlist = []
attendance = {}
attendTime = {}
exitTime = {}
attendImg = {}

class AtdItem(object):
    def __init__(self, name, description1):
        self.name = name
        self.description1 = description1

        
class NotAtdItem(object):
    def __init__(self, name):
        self.name = name
        

class LogItem(object):
    def __init__(self, name, description1, description2):
        self.name = name
        self.description1 = description1
        self.description2 = description2


class AtdTable(Table):
    name = Col('名前')
    description1 = Col('出席時刻')


class NotAtdTable(Table):
    name = Col('名前')
    

class LogTable(Table):
    name = Col('名前')
    description1 = Col('時刻')
    description2 = Col('入退状況')


class addUser(Form):
    username = StringField("登録者の名前",
                           [validators.InputRequired("この項目は入力必須です")])

    # html側で表示するsubmitボタンの表示
    submit = SubmitField("決定")


def read_csv():
    with open('face_log.csv') as f:
        reader = csv.DictReader(f, fieldnames=['name', 'time', 'state', 'path'])
        logdictlist = [row for row in reader]

    loglength = len(logdictlist)
    for i in range(loglength):
        logdictlist[i]['time'] = datetime.datetime.strptime(logdictlist[i]['time'], '%Y%m%d%H%M%S')

    return logdictlist


def update_log(logdictlist):
    d_len = len(logdictlist)

    for i in range(d_len):
        if logdictlist[i]['state'] == 'in':
            attendance[logdictlist[i]['name']] = True
            attendTime[logdictlist[i]['name']] = logdictlist[i]['time']
            attendImg[logdictlist[i]['name']] = logdictlist[i]['path']

        if logdictlist[i]['state'] == 'out':
            attendance[logdictlist[i]['name']] = False
            exitTime[logdictlist[i]['name']] = logdictlist[i]['time']


@app.route('/')
def index1():
    return render_template('index1.html')

@app.route('/conf/')
def index2():
    atditems = []
    natditems = []
    length = len(memberlist)
    
    if length == 0:
        return render_template('index2_n.html')

    logdictlist = read_csv()

    update_log(logdictlist)

    for i in range(length):
        name = memberlist[i]
        if attendance[name]:
            atditems.append(AtdItem(name, attendTime[name]))
            if name not in attendlist:
                attendlist.append(name)
        else:
            natditems.append(NotAtdItem(name))
            if name in attendlist:
                attendlist.remove(name)
            
    atdtable = AtdTable(atditems)
    natdtable = NotAtdTable(natditems)
    return render_template('index2.html', table1=atdtable, table2=natdtable, list=attendlist, dict=attendImg)

@app.route('/log/')
def index2_log():
    logitems = []

    logdictlist = read_csv()

    loglength = len(logdictlist)
    print(loglength)

    for i in range(loglength):
        logitems.append(LogItem(logdictlist[i]['name'], logdictlist[i]['time'], logdictlist[i]['state'].upper()))

    logtable = LogTable(logitems)
    return render_template('index2_log.html', logtable=logtable)

@app.route('/create/', methods = ['GET', 'POST'])
def index3():
    if request.method == 'POST':
        member_index = request.form.getlist('checkbox')
        member_index_i = [int(s) for s in member_index]

        if len(allmemberlist) in member_index_i:
            member_index_i = range(len(allmemberlist))
        ilen = len(member_index_i)
        print(member_index_i)

        for i in range(ilen):
            memberlist.append(allmemberlist[member_index_i[i]])
            attendance[allmemberlist[member_index_i[i]]]=False
            attendTime[allmemberlist[member_index_i[i]]]='-'
            exitTime[allmemberlist[member_index_i[i]]]='-'
        
        print(memberlist)
        
        return render_template('index3.html', message = '出席簿を作成しました、トップページに戻ってください', member=allmemberlist)
    elif request.method == 'GET':
        
        return render_template('index3.html', message = '今回の出席予定者を全て選択して下さい', member=allmemberlist)

@app.route('/camera/')
def index4():
    return render_template('index4.html')

@app.route('/init/')
def index5():
    memberlist.clear()
    attendance.clear()
    attendTime.clear()
    exitTime.clear()
    shutil.rmtree('static/images')
    os.mkdir('static/images')
    return render_template('index5.html')

if __name__ == "__main__":
    app.run(port=8000, debug=True)

