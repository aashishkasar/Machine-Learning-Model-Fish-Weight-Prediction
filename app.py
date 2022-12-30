from flask import Flask,request,render_template,url_for
import joblib
import sklearn

app=Flask(__name__)

model=joblib.load(r"D:\Data Sets\ML Models\fish1.joblib")

@app.route("/")
def f():
    return render_template("fish.html")


@app.route("/result",methods=["GET","POST"])
def result():
        var1=int(request.form["species"])
        var2=float(request.form["len1"])
        var3=float(request.form["len2"])
        var4=float(request.form["len3"])
        var5=float(request.form["height"])
        var6=float(request.form["width"])
        predict=model.predict([[var1,var2,var3,var4,var5,var6]])
        return render_template('output.html',predict=predict)


if __name__=="__main__":
            app.run(debug=True)


