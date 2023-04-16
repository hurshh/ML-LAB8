import os

from flask import *
import pickle
import pandas as pd

from werkzeug.utils import secure_filename

filename = 'spam-sms-mnb-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl', 'rb'))


UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')

ALLOWED_EXTENSIONS = {'csv'}



app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'sesh1'


@app.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        f = request.files.get('file')

        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)

        f.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))

        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)

        return render_template('index2.html')
    return render_template("index.html")


@app.route('/show_data')
def showData():
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path, encoding='unicode_escape')

    ans = []

    for ind in uploaded_df.index:
        data = [uploaded_df['MESSAGES'][ind]]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        if(my_prediction == [0]):
            ans.append("HAM")
        else:
            ans.append("SPAM")


    uploaded_df['Result'] = ans
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_csv_data.html',
                           data_var=uploaded_df_html)


if __name__ == '__main__':
    app.run(debug=True)
