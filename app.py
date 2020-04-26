from flask import Flask, render_template, request
import LSTM

app = Flask(__name__)
app.static_folder = 'static'
res = ["dsadasd", 0]
rnews = ""


@app.route('/')
def index():
    global res
    # state="Fake"
    return render_template("index.html", res=res, to_show=False)


@app.route('/send', methods=['GET', 'POST'])
def send():
    global res
    if request.method == 'POST':
        onews = str(request.form['news'])
        ftr = open('y.txt', 'w')
        ftr.write(onews)
        ftr.close()
        res = LSTM.process_inp_news()
        print(res)
        return render_template('index.html', res=res, rnews=onews, to_show=True)
    return render_template("index.html", to_show=False, res=res)


if __name__ == "__main__":
    app.run()
