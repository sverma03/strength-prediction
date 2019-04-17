# import Flask class from the flask module
from flask import Flask, render_template, request

import pickle
import pandas

# Create Flask object to run
app = Flask(__name__)

# Load the persisted ML model.
print ("Loading model...")
concreteStrengthPredictorFile = open('model/concrete_lgbm_final_model_ver_1_0.sav', 'rb')
concreteStrengthPredictorModel = pickle.load(concreteStrengthPredictorFile)
concreteStrengthPredictorFile.close()
	
@app.route('/')
@app.route('/index')
def home():
    return "Hi, Welcome to Flask!!"

	
# Render Concrete mixture input page
@app.route('/input')
def input():
    return render_template('input.html')

	
# This function will be called when the input page is submitted
@app.route('/predict', methods=["POST"])
def predict():

    # Enter into this snippet of the code only if the method is POST.
	if request.method == "POST":
	
		# Get values from browser
		input_dict = request.form.to_dict()
		
		# Extract the values alone. Convert them to float as they will be in String format
		input_dict_values = map(float, list(input_dict.values()))
		
		# Form the dictionary object once again with the existing keys and formatted values
		input_dict = dict(zip(list(input_dict.keys()), input_dict_values))
		
		# Alternately, you can extract each field, cast to float, and pass it as a value
		#input_dict = dict{'cement': float(request.form['cement']),
		#				  'blast': float(request.form['blast']),
		#				  'flyash': float(request.form['flyash']),
		#				  'water': float(request.form['water'])],
		#				  'superplasticizer': float(request.form['superplasticizer']),
		#				  'coarse_aggregate': float(request.form['coarse_aggregate']),
		#				  'fine_aggregate': float(request.form['fine_aggregate']),
		#				  'age': float(request.form['age'])
		#				 }
		
		# Construct the dataframe out of the dictionary object
		concrete_df = pandas.DataFrame(input_dict, index=[0])
		print ("Input values: \n", concrete_df)

		# Pass the dataframe object to loaded ML model and do prediction
		strength_predicted = str(round(concreteStrengthPredictorModel.predict(concrete_df)[0], 2))
		print ("Predicted Concrete Strength: ", strength_predicted)

		return render_template('results.html', strength_predicted=strength_predicted)


	
# ----------------------------- FOR LOCAL SERVER ONLY -----------------------------
# Load the persisted ML model. 
# NOTE: The model will be loaded only once at the start of the server
def load_model():
	global concreteStrengthPredictorModel
	print ("Loading model...")
	concreteStrengthPredictorFile = open('model/concrete_lgbm_final_model_ver_1_0.sav', 'rb')
	concreteStrengthPredictorModel = pickle.load(concreteStrengthPredictorFile)
	concreteStrengthPredictorFile.close()
	
# FOR LOCAL SERVER ONLY	
if __name__ == "__main__":
	print("**Starting Server...")
	
	# Call function that loads Model
	load_model()
	
	# Run Server
	app.run()