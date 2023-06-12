import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=("artifacts/model.pkl")
            preprocessor_path=('artifacts/preprocessor.pkl')
            # print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            # print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 DER_mass_MMC: float,
                 DER_mass_transverse_met_lep: float,
                 DER_mass_vis: float,
                 DER_pt_h: float,
                 DER_deltaeta_jet_jet: float,
                 DER_mass_jet_jet: float,
                 DER_prodeta_jet_jet: float, 
                 DER_deltar_tau_lep: float,
                 DER_pt_tot: float,
                 DER_sum_pt: float,
                 DER_pt_ratio_lep_tau: float,
                 DER_met_phi_centrality: float,
                 DER_lep_eta_centrality: float,
                 PRI_tau_pt: float,
                 PRI_tau_eta: float,
                 PRI_tau_phi: float,
                 PRI_lep_pt: float,
                 PRI_lep_eta: float,
                 PRI_lep_phi: float,
                 PRI_met: float,
                 PRI_met_phi: float,
                 PRI_met_sumet: float,
                 PRI_jet_num: int,
                 PRI_jet_leading_pt:float,
                 PRI_jet_leading_eta: float,
                 PRI_jet_leading_phi:float,
                 PRI_jet_subleading_pt: float,
                 PRI_jet_subleading_eta: float,
                 PRI_jet_subleading_phi: float,
                 PRI_jet_all_pt: float,
                 Weight: float):
                
                self.DER_mass_MMC = DER_mass_MMC
                self.DER_mass_transverse_met_lep = DER_mass_transverse_met_lep
                self.DER_mass_vis = DER_mass_vis
                self.DER_pt_h = DER_pt_h
                self.DER_deltaeta_jet_jet = DER_deltaeta_jet_jet
                self.DER_mass_jet_jet = DER_mass_jet_jet
                self.DER_prodeta_jet_jet = DER_prodeta_jet_jet
                self.DER_deltar_tau_lep = DER_deltar_tau_lep
                self.DER_pt_tot = DER_pt_tot
                self.DER_sum_pt = DER_sum_pt
                self.DER_pt_ratio_lep_tau = DER_pt_ratio_lep_tau
                self.DER_met_phi_centrality = DER_met_phi_centrality
                self.DER_lep_eta_centrality = DER_lep_eta_centrality
                self.PRI_tau_pt = PRI_tau_pt
                self.PRI_tau_eta = PRI_tau_eta
                self.PRI_tau_phi = PRI_tau_phi
                self.PRI_lep_pt = PRI_lep_pt
                self.PRI_lep_eta = PRI_lep_eta
                self.PRI_lep_phi = PRI_lep_phi
                self.PRI_met = PRI_met
                self.PRI_met_phi = PRI_met_phi
                self.PRI_met_sumet = PRI_met_sumet
                self.PRI_jet_num = PRI_jet_num
                self.PRI_jet_leading_pt = PRI_jet_leading_pt
                self.PRI_jet_leading_eta = PRI_jet_leading_eta
                self.PRI_jet_leading_phi = PRI_jet_leading_phi
                self.PRI_jet_subleading_pt = PRI_jet_subleading_pt
                self.PRI_jet_subleading_eta = PRI_jet_subleading_eta
                self.PRI_jet_subleading_phi = PRI_jet_subleading_phi
                self.PRI_jet_all_pt = PRI_jet_all_pt
                self.Weight = Weight


    def get_data_as_data_frame(self):
    
        try:
            custom_data_input_dict = {
                "DER_mass_MMC": [self.DER_mass_MMC],
                "DER_mass_transverse_met_lep": [self.DER_mass_transverse_met_lep],
                "DER_mass_vis": [self.DER_mass_vis],
                "DER_pt_h": [self.DER_pt_h],
                "DER_deltaeta_jet_jet": [self.DER_deltaeta_jet_jet],
                "DER_mass_jet_jet": [self.DER_mass_jet_jet],
                "DER_prodeta_jet_jet": [self.DER_prodeta_jet_jet],
                "DER_deltar_tau_lep": [self.DER_deltar_tau_lep],
                "DER_pt_tot": [self.DER_pt_tot],
                "DER_sum_pt": [self.DER_sum_pt],
                "DER_pt_ratio_lep_tau": [self.DER_pt_ratio_lep_tau],
                "DER_met_phi_centrality": [self.DER_met_phi_centrality],
                "DER_lep_eta_centrality": [self.DER_lep_eta_centrality],
                "PRI_tau_pt": [self.PRI_tau_pt],
                "PRI_tau_eta": [self.PRI_tau_eta],
                "PRI_tau_phi": [self.PRI_tau_phi],
                "PRI_lep_pt": [self.PRI_lep_pt],
                "PRI_lep_eta": [self.PRI_lep_eta],
                "PRI_lep_phi": [self.PRI_lep_phi],
                "PRI_met": [self.PRI_met],
                "PRI_met_phi": [self.PRI_met_phi],
                "PRI_met_sumet": [self.PRI_met_sumet],
                "PRI_jet_num": [self.PRI_jet_num],
                "PRI_jet_leading_pt": [self.PRI_jet_leading_pt],
                "PRI_jet_leading_eta": [self.PRI_jet_leading_eta],
                "PRI_jet_leading_phi": [self.PRI_jet_leading_phi],
                "PRI_jet_subleading_pt": [self.PRI_jet_subleading_pt],
                "PRI_jet_subleading_eta": [self.PRI_jet_subleading_eta],
                "PRI_jet_subleading_phi": [self.PRI_jet_subleading_phi],
                "PRI_jet_all_pt": [self.PRI_jet_all_pt],
                "Weight": [self.Weight],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)