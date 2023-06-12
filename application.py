from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            DER_mass_MMC = float(request.form.get('DER_mass_MMC')),
            DER_mass_transverse_met_lep = float(request.form.get('DER_mass_transverse_met_lep')),
            DER_mass_vis = float(request.form.get('DER_mass_vis')),
            DER_pt_h = float(request.form.get('DER_pt_h')),
            DER_deltaeta_jet_jet = float(request.form.get('DER_deltaeta_jet_jet')),
            DER_mass_jet_jet = float(request.form.get('DER_mass_jet_jet')),
            DER_prodeta_jet_jet = float(request.form.get('DER_prodeta_jet_jet')),
            DER_deltar_tau_lep = float(request.form.get('DER_deltar_tau_lep')),
            DER_pt_tot = float(request.form.get('DER_pt_tot')),
            DER_sum_pt = float(request.form.get('DER_sum_pt')),
            DER_pt_ratio_lep_tau = float(request.form.get('DER_pt_ratio_lep_tau')),
            DER_met_phi_centrality = float(request.form.get('DER_met_phi_centrality')),
            DER_lep_eta_centrality = float(request.form.get('DER_lep_eta_centrality')),
            PRI_tau_pt = float(request.form.get('PRI_tau_pt')),
            PRI_tau_eta = float(request.form.get('PRI_tau_eta')),
            PRI_tau_phi = float(request.form.get('PRI_tau_phi')),
            PRI_lep_pt = float(request.form.get('PRI_lep_pt')),
            PRI_lep_eta = float(request.form.get('PRI_lep_eta')),
            PRI_lep_phi = float(request.form.get('PRI_lep_phi')),
            PRI_met = float(request.form.get('PRI_met')),
            PRI_met_phi = float(request.form.get('PRI_met_phi')),
            PRI_met_sumet = float(request.form.get('PRI_met_sumet')),
            PRI_jet_num = int (request.form.get('PRI_jet_num')),
            PRI_jet_leading_pt = float(request.form.get('PRI_jet_leading_pt')),
            PRI_jet_leading_eta = float(request.form.get('PRI_jet_leading_eta')),
            PRI_jet_leading_phi = float(request.form.get('PRI_jet_leading_phi')),
            PRI_jet_subleading_pt = float(request.form.get('PRI_jet_subleading_pt')),
            PRI_jet_subleading_eta = float(request.form.get('PRI_jet_subleading_eta')),
            PRI_jet_subleading_phi = float(request.form.get('PRI_jet_subleading_phi')),
            PRI_jet_all_pt = float(request.form.get('PRI_jet_all_pt')),
            Weight = float(request.form.get('Weight'))

        )
        pred_df = data.get_data_as_data_frame()
        print("Before Prediction", pred_df)

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        # if results == 0:
        #     print ('b')
        # else:
        #     print ('s')
        print("After Prediction", results[0])
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)