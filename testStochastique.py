#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 12:22:18 2025

@author: irich2025
"""
import numpy as np
import matplotlib.pyplot as plt
import      torch
import      argparse
from        lib             import init
from        lib.runners     import vaeRunner, latentRunner
from        utils.figs_time import vis_temporal_Prediction, predFieldFigure, errorFieldFigure, Ec, Ec_tot
from lib.init import pathsBib
from lib.pp_space import get_Ek_t
from lib.pp_time import make_physical_prediction


nn="easy"
#nn=self"
#nn="lstm"

re=40
#re=100

#m="test"
m="run" 
#m="train"
#m="infer"

t="pre" #ou "val" ou "final
#pod=True

device = ('cuda' if torch.cuda.is_available() else "cpu")

datafile = init.init_env(re)

bvae   = vaeRunner(device,datafile)
if m == 'train':
    bvae.train()
elif m == 'test':
    bvae.infer(t)
elif m == 'run':
    bvae.run()

lruner = latentRunner(nn,device)
lruner.train()

#######################################
#Générer un plot en détaillant les paramètres

"""lruner.infer(args.t) 
#vis_temporal_Prediction(model_type=args.nn, predictor=lruner, vae=bvae)
model_type=args.nn 
predictor=lruner 
vae=bvae

figPath     = pathsBib.fig_path + model_type + '/'
case_name   = predictor.filename
datPath     = pathsBib.res_path + case_name + '.npz'
d           = np.load(datPath)
g           = d['g']
p           = d['p']

VAErec, pred = make_physical_prediction(vae=vae,pred_latent=p,true_latent=g,device=vae.device)
    
stepPlot     = int(predictor.config.in_dim + 1) # Here we test the prediction purely based on the predicted variables 

predFieldFigure(vae.test_d,VAErec,pred,
                vae.std, vae.mean,
                stepPlot  = stepPlot,
                model_name= model_type,
                save_file = figPath + "recSnapShot_" + case_name + '.jpg')"""
 
#################################################################################
#Comparaison de 2 réalisations 
ns=4 #nb snapshots équirépartis à affich
model_type=nn 
predictor=lruner 
vae=bvae
stepPlot    = int(predictor.config.in_dim + 1) # Here we test the prediction purely based on the predicted variables 

figPath     = pathsBib.fig_path + model_type + '/'
case_name   = predictor.filename
datPath     = pathsBib.res_path + case_name + '.npz'


# Première réalisation
predictor.infer(t)
d1           = np.load(datPath)
g1           = d1['g']
p1           = d1['p']
VAErec_1, pred_1 = make_physical_prediction(vae=vae, pred_latent=p1, true_latent=g1, device=vae.device) 
#pred_1 = pred_1 * vae.std + vae.mean# Remise à l’échelle



# Deuxième réalisation
predictor.infer(t)
d2           = np.load(datPath)
g2           = d2['g']
p2           = d2['p']
VAErec_2, pred_2 = make_physical_prediction(vae=vae, pred_latent=p2, true_latent=g2, device=vae.device)
#pred__2 = pred_2 * vae.std + vae.mean# Remise à l’échelle

predFieldFigure(vae.test_d,VAErec_1,pred_1,
                vae.std, vae.mean,
                stepPlot  = stepPlot,
                model_name= model_type,
                save_file = figPath + "recSnapShot_1" + case_name + '.jpg')  

predFieldFigure(vae.test_d,VAErec_2,pred_2,
                vae.std, vae.mean,
                stepPlot  = stepPlot,
                model_name= model_type,
                save_file = figPath + "recSnapShot_2" + case_name + '.jpg')  

errorFieldFigure(pred_1, pred_2, vae.std, vae.mean,
                 stepPlot=stepPlot,
                 model_name=model_type,
                 save_file=figPath + "diff_pred1_pred2_" + case_name + ".jpg")

#true data = data test
errorFieldFigure(pred_1, vae.test_d, vae.std, vae.mean,
                 stepPlot=stepPlot,
                 model_name=model_type,
                 save_file=figPath + "diff_pred1_true_" + case_name + ".jpg")

errorFieldFigure(pred_2, vae.test_d, vae.std, vae.mean,
                 stepPlot=stepPlot,
                 model_name=model_type,
                 save_file=figPath + "diff_pred2_true_" + case_name + ".jpg")

#Plot Ec
Ec1 = Ec_tot(pred_1)
Ec2 = Ec_tot(pred_2)
plt.figure()
plt.plot(Ec1, label="Champ 1", linestyle='--')
plt.plot(Ec2, label="Champ", linestyle = '--')
plt.title("Énergie cinétique totale pour 2 réalisations")
plt.xlabel("Temps (snapshot index)")
plt.ylabel("Énergie Ec")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()