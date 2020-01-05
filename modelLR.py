from flask import Flask, request, jsonify
import traceback
import pandas as pd
from joblib import load
from titanicModel import createFeatures
import sys

# Your API definition
app = Flask(__name__)
@app.route('/predict', methods=['POST']) # Your API endpoint URL would consist /predict

def predict():

	if lr:
		try:
			#load request
			json_ = request.json
			print(json_)
			query_df = pd.DataFrame(json_)

			query = createFeatures(query_df, model_columns)
			prediction = list(lr.predict(query))

			return jsonify({'prediction': str(prediction)})

		except:
			return jsonify({'trace': traceback.format_exc()})

	else:
		print('Model was not found')
		return 'No model'


if __name__ == '__main__':
	try:
		# This is for a command-line argument
		port = int(sys.argv[1])
	except:
		# If you don't provide any port then the port will be set to 12345
		port = 12345

	# Load "model.pkl"
	lr = load('models/model.pkl')
	print('Model loaded')

	# Load model's column names
	model_columns = load('models/model_columns.pkl')
	print('Model columns loaded')

	app.run(port=port, debug=True)