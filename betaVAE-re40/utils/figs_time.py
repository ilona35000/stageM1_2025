
"""
The Visualisation of time-series prediction results

@yuningw
"""

from lib.runners import latentRunner, vaeRunner
from  pathlib import Path
from lib.init import pathsBib
import numpy as np 
from lib.pp_time import make_physical_prediction
import matplotlib.pyplot as plt 
import utils.plt_rc_setup
from lib.pp_time import PDF
import os

#-----------------------------------------------------------
class colorplate:
    """
    Color in HTML format for plotting 
    """

    red     = "r"
    blue    = "b" 
    yellow  = "y" 
    cyan    = "c" 
    black   = "k" 
    
cc = colorplate()

#-----------------------------------------------------------
def vis_temporal_Prediction(
                            vae         :vaeRunner,
                            predictor   :latentRunner,
                            model_type,
                            if_loss     = True, 
                            if_evo      = True,
                            if_window   = True,
                            if_pmap_s   = False,
                            if_pmap_all = False,
                            if_snapshot = True
                            ):
    """
    Visualisation of the temporal-dynamics prediction results 

    Args:

        vae         : (lib.runners.vaeRunner) The module for employed VAE     
    
        predictor   : (lib.runners.latentRunner) The module for latent-space temporal-dynamics predictions

        model_type  : (str) The type of model used (easy/self/lstm)
        
        if_loss     : (bool) If plot the loss evolution 
        
        if_evo      : (bool) If plot the temporal evolution of latent mode
        
        if_window   : (bool) If plot the l2-norm error horizon
        
        if_pmap_s   : (bool) If plot the single Poincare Map
        
        if_pmap_all : (bool) If plot all the Poincare Maps

        if_snapshot : (bool) If plot the flow field reconsturction 

    """
    
    figPath     = pathsBib.fig_path + model_type + '/'
    case_name   = predictor.filename
    datPath     = pathsBib.res_path + case_name + '.npz'
    
    print("#"*30)
    print(f"Start visualisation:\nSave Fig to:{figPath}\nLoad data from:{datPath}")

    try:
        d           = np.load(datPath) #data
        g           = d['g'] #test_data
        p           = d['p'] #Preds
        e           = d['e'] #window_error
        pmap_g      = d['pmap_g'] #Intersec_test = détections passages à 0
        pmap_p      = d['pmap_p'] #Intersec_pred = détections passages à 0
        #save dans à la fin de runners

    except:
        print(f"ERROR: FAILD loading data")

    Path(figPath).mkdir(exist_ok=True)

    #if if_loss:
    plot_loss(predictor.history, save_file= figPath + "loss_" + case_name + '.jpg' )
    print(f'INFO: Loss Evolution Saved!')

    if if_evo and (g.any() !=None) and (p.any() != None):
        plot_signal(g,p,            save_file=figPath + "signal_" + case_name + '.jpg' )
        print(f"INFO: Prediction Evolution Saved!")

    if if_window and (e.any() != None):
        plot_pred_horizon_error(e, colorplate.blue, save_file=figPath + "horizon_" + case_name + '.jpg' )
        print(f"INFO: l2-norm error Horizion Saved!")

    ## Poincare Map 
    planeNo = 0
    postive_dir = True
    lim_val = 2.5  # Limitation of x and y bound when compute joint pdf
    grid_val = 50
    i = 1; j =1 

    if if_pmap_s and (pmap_g.any() != None) and (pmap_p.any() != None):
        plotSinglePoincare( planeNo, i, j, 
                        pmap_p,pmap_g,
                        lim_val, grid_val,
                        save_file = figPath + f'Pmap_{i}_{j}_' + case_name + '.jpg')
        print(f"INFO: Single Poincare Map of {i}, {j} Saved!")
    
    if if_pmap_all and (pmap_g.any() != None) and (pmap_p.any() != None):
        plotCompletePoincare(predictor.config.latent_dim,planeNo, 
                        pmap_p, pmap_g,
                        lim_val, grid_val,
                        save_file = None, 
                        dpi       = 200)
        print(f"INFO: Complete Poincare Map Saved!")

    if if_snapshot and (p.any() != None) and (g.any() != None):
        VAErec, pred = make_physical_prediction(vae=vae,pred_latent=p,true_latent=g,device=vae.device)
        
        stepPlot     = int(predictor.config.in_dim + 1) # Here we test the prediction purely based on the predicted variables 

        predFieldFigure(vae.test_d,VAErec,pred,
                        vae.std,vae.mean,
                        stepPlot  = stepPlot,
                        model_name= model_type,
                        save_file = figPath + "recSnapShot_" + case_name + '.jpg')
        print(f"INFO: Reconstruted Snapshot at {stepPlot} Saved!")
    return


#-----------------------------------------------------------
def plot_loss(history, save_file, dpi = 200):
    """
    Plot the loss evolution during training
    Args: 
        history     : A dictionary contains loss evolution in list
        save_file   : Path to save the figure, if None, then just show the plot
        dpi         : The dpi for save the file 

    Returns:
        A fig for loss 

    """

    fig, axs = plt.subplots(1,1,figsize=(12,5))
    
    axs.semilogy(history["train_loss"],lw = 2, c = cc.blue)
    if len(history["val_loss"]) != 0 :
        axs.semilogy(history["val_loss"],lw = 2, c = cc.red)
    axs.set_xlabel("Epoch")
    axs.set_ylabel("MSE Loss") 
    
    if save_file != None:
        plt.savefig(save_file, bbox_inches='tight', dpi= dpi)


#-----------------------------------------------------------
def plot_signal(test_data, Preds, save_file:None, dpi = 200):
    """
    Plot the temproal evolution of prediction and test data
    Args: 
        test_data   : A numpy array of test data 
        Preds       : A numpy array of prediction data
        
        save_file   : Path to save the figure, if None, then just show the plot
        dpi         : The dpi for save the file 

    Returns:
        A fig for temporal dynamic of ground truth and predictions on test data

    """
    import sys
    
    try: 
        test_data.shape == Preds.shape 
    except:
        print("The prediction and test data must have same shape!")
        sys.exit()
    

    Nmode = min(test_data.shape[0],test_data.shape[-1])

    fig, axs = plt.subplots(Nmode,1,figsize=(16,2.5* Nmode),sharex=True)
    
    for i, ax in enumerate(axs):
        ax.plot(test_data[:,i],c = cc.black,lw = 1.5)
        ax.plot(Preds[:, i], c='red', marker='x', linestyle='None')
        ax.set_ylabel(f"M{i+1}")
        axs[-1].set_xlabel("t",fontsize=20)

    ax.legend(["Ground truth (test)",'Prediction'])
    
    if save_file != None:
        plt.savefig(save_file, bbox_inches='tight', dpi= dpi)


#-----------------------------------------------------------

def plot_pred_horizon_error(window_err, Color, save_file, dpi=300):
    """
    Viusalize the latent-space prediction horizon error 

    Args:
    
        window_err      :   (NumpyArray) The horizon of l2-norm error of  prediction 

        Color           :   (str) The color for the line 

        save_file   : Path to save the figure, if None, then just show the plot

        dpi         : The dpi for save the file 
        
    """

    fig, axs = plt.subplots(1,1, figsize = (8,4))

    axs.plot(window_err, lw = 3, c = Color)
    axs.set_xlabel("Prediction steps",fontsize = 20)
    axs.set_ylabel(r"$\epsilon$",fontsize = 20)  

    if save_file != None:
        plt.savefig(save_file, bbox_inches='tight', dpi= dpi)

    return 


#-----------------------------------------------------------
def plotSinglePoincare( planeNo, i, j, 
                        InterSec_pred,InterSec_test,
                        lim_val, grid_val,
                        save_file:None, 
                        dpi = 200):
    """
    
    Visualisation of a single Poincare Map for test data and prediction
    
    Args: 

        planeNo     :   (int) The plane no to compute the intersection 

        i           :   (int) The Number of the mode on x-Axis

        j           :   (int) The Number of the mode on y-Axis

        lim_val     :   (float) Limitation of region on the map

        grid_val    :   (int) Number of the mesh grid 

        save_file   :   (str) Path to save the file 

        dpi         :   (int) The dpi for the image 

    """

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    _, _, pdf_test = PDF(InterSecX=InterSec_test[:, i],
                        InterSecY=InterSec_test[:, j],
                        xmin=-lim_val, xmax=lim_val,
                        ymin=-lim_val, ymax=lim_val,
                        x_grid=grid_val, y_grid=grid_val,
                        )

    xx, yy, pdf_pred = PDF(InterSecX=InterSec_pred[:, i],
                        InterSecY=InterSec_pred[:, j],
                        xmin=-lim_val, xmax=lim_val,
                        ymin=-lim_val, ymax=lim_val,
                        x_grid=grid_val, y_grid=grid_val,
                        )

    axs.contour(xx, yy, pdf_test, colors=cc.black)
    axs.contour(xx, yy, pdf_pred, colors='lightseagreen')
    axs.set_xlim(-lim_val, lim_val)
    axs.text(0.80, 0.08, '$r_{}=0$'.format(planeNo+1), fontsize=16,
                transform=axs.transAxes, bbox=dict(facecolor='white', alpha=0.4))
    axs.set_xlabel(f"$r_{i + 1}$", fontsize='large')
    axs.set_ylabel(f"$r_{j + 1}$", fontsize='large')
    axs.set_aspect('equal', "box")
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.grid(visible=True, markevery=1, color='gainsboro', zorder=1)
    if save_file != None:
        plt.savefig(save_file, bbox_inches = 'tight', dpi = dpi)


    return 


#-----------------------------------------------------------
def plotCompletePoincare(Nmodes,planeNo, 
                        InterSec_pred, InterSec_test,
                        lim_val, grid_val,
                        save_file = None, 
                        dpi       = 200):
    """

    Visualisation of whole Poincare Maps for test data and prediction
    

    Args: 

        planeNo     :   (int) The plane no to compute the intersection 

        lim_val     :   (float) Limitation of region on the map

        grid_val    :   (int) Number of the mesh grid 

        save_file   :   (str) Path to save the file 

        dpi         :   (int) The dpi for the image 
        
    
    """    
    
    fig, axs = plt.subplots(Nmodes, Nmodes,
                            figsize=(5 * Nmodes, 5 * Nmodes),
                            sharex=True, sharey=True)

    for i in range(0, Nmodes):
        for j in range(0, Nmodes):
            if i == j or j == planeNo or i == planeNo or j > i:
                axs[i, j].set_visible(False)
                continue

            _, _, pdf_test = PDF(InterSecX=InterSec_test[:, i],
                                InterSecY=InterSec_test[:, j],
                                xmin=-lim_val, xmax=lim_val,
                                ymin=-lim_val, ymax=lim_val,
                                x_grid=grid_val, y_grid=grid_val,
                                )

            xx, yy, pdf_pred = PDF(InterSecX=InterSec_pred[:, i],
                                InterSecY=InterSec_pred[:, j],
                                xmin=-lim_val, xmax=lim_val,
                                ymin=-lim_val, ymax=lim_val,
                                x_grid=grid_val, y_grid=grid_val,
                                )

            axs[i, j].contour(xx, yy, pdf_test, colors=cc.black)
            axs[i, j].contour(xx, yy, pdf_pred, colors='lightseagreen')
            axs[i, j].set_xlim(-lim_val, lim_val)
            axs[i, j].set_xlabel(f"$r_{i + 1}$", fontsize='large')
            axs[i, j].set_ylabel(f"$r_{j + 1}$", fontsize='large')
            axs[i, j].set_aspect('equal', "box")
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].grid(visible=True, markevery=1, color='gainsboro', zorder=1)
    
    if save_file != None:
        plt.savefig(save_file, bbox_inches = 'tight', dpi = dpi)

    return



#-----------------------------------------------------------

def predFieldFigure(true, VAErec, pred, std_data, mean_data, stepPlot, model_name, save_file, dpi=200):
    
    """
    
    Visualise the flow fields reconstructed by the latent-space prediction from the transformer/lstm 

    true        :       (NumpyArray) The ground truth 

    VAErec      :       (NumpyArray) The reconstruction from VAE ONLY 

    pred        :       (NumpyArray) The reconstruction from the prediction of transformer 

    std_data    :       (NumpyArray) Std of flow fields
    
    mean_data   :       (NumpyArray) Mean of flow fields

    model_name  :       (str) The name of the predictor model: easy/self/lstm

    save_file   :       (str) Path to save the file 

    dpi         :       (int) The dpi for the image 
        
    """

    fig, ax = plt.subplots(2, 3, figsize=(11, 4), sharex='col', sharey='row')

    Umax = 1.5
    Umin = 0
    Vlim = 1

    # From dataset
    true_u  = (true[stepPlot, 0, :, :] * std_data[0, 0, :, :] + mean_data[0, 0, :, :]).squeeze()
    true_v  = (true[stepPlot, 1, :, :] * std_data[0, 1, :, :] + mean_data[0, 1, :, :]).squeeze()
    
    vae_u   = (VAErec[stepPlot, 0, :, :] * std_data[0, 0, :, :] + mean_data[0, 0, :, :]).squeeze()
    vae_v   = (VAErec[stepPlot, 1, :, :] * std_data[0, 1, :, :] + mean_data[0, 1, :, :]).squeeze()
    
    pred_u  = (pred[stepPlot, 0, :, :] * std_data[0, 0, :, :] + mean_data[0, 0, :, :]).squeeze()
    pred_v  = (pred[stepPlot, 1, :, :] * std_data[0, 1, :, :] + mean_data[0, 1, :, :]).squeeze()
    
    im = ax[0, 0].imshow(true_u,
                        cmap="RdBu_r", vmin=Umin, vmax=Umax, extent=[-9, 87, -14, 14])
    ax[0, 0].set_title('True u\n($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)')
    fig.colorbar(im, ax=ax[0, 0], shrink=0.7, ticks=([0, 0.5, 1, 1.5]))

    im = ax[1, 0].imshow(true_v,
                        cmap="RdBu_r", vmin=-Vlim, vmax=Vlim, extent=[-9, 87, -14, 14])
    ax[1, 0].set_title('True v\n($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)')
    fig.colorbar(im, ax=ax[1, 0], shrink=0.7)

    # Encoded and decoded
    im = ax[0, 1].imshow(vae_u,
                        cmap="RdBu_r", vmin=Umin, vmax=Umax, extent=[-9, 87, -14, 14])
    ax[0, 1].set_title(r'$\beta$-VAE' + ' u\n($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)')
    fig.colorbar(im, ax=ax[0, 1], shrink=0.7, ticks=([0, 0.5, 1, 1.5]))

    im = ax[1, 1].imshow(vae_v,
                        cmap="RdBu_r", vmin=-Vlim, vmax=Vlim, extent=[-9, 87, -14, 14])
    ax[1, 1].set_title(r'$\beta$-VAE' + ' v\n($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)')
    fig.colorbar(im, ax=ax[1, 1], shrink=0.7)

    # Encoded, predicted and decoded
    im = ax[0, 2].imshow(pred_u,
                        cmap="RdBu_r", vmin=Umin, vmax=Umax, extent=[-9, 87, -14, 14])
    ax[0, 2].set_title(r'$\beta$-VAE + ' + model_name + ' u\n($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)')
    fig.colorbar(im, ax=ax[0, 2], shrink=0.7, ticks=([0, 0.5, 1, 1.5]))

    im = ax[1, 2].imshow(pred_v,
                        cmap="RdBu_r", vmin=-Vlim, vmax=Vlim, extent=[-9, 87, -14, 14])
    ax[1, 2].set_title(r'$\beta$-VAE + ' + model_name + ' v\n($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)')
    fig.colorbar(im, ax=ax[1, 2], shrink=0.7)

    ax[1, 0].set_xlabel('x/c')
    ax[1, 1].set_xlabel('x/c')
    ax[1, 2].set_xlabel('x/c')
    ax[0, 0].set_ylabel('y/c')
    ax[1, 0].set_ylabel('y/c')

    # fig.set_tight_layout(True)

    if save_file != None:
        plt.savefig(save_file, bbox_inches = 'tight', dpi = dpi)

    return fig, ax


#Ajout fonction
def errorFieldFigure(pred1, pred2, std, mean, stepPlot, model_name, save_file, dpi=200):
    """plot des erreurs de reconstructions, fonction adaptée de predFieldFigure
    Par convention pred2 est la true data si elle est donnée"""
    
    pred1 = pred1 * std + mean
    pred2 = pred2 * std + mean
    err = pred1 - pred2
    norm_err = np.linalg.norm(err, axis=1)  # norme vectorielle de l'erreur par pixel
    print("shape de norm_err", norm_err.shape)
    norms = np.linalg.norm(norm_err.reshape(200, -1), axis=1) / np.linalg.norm(pred2)


    fig, ax = plt.subplots(1, 3, figsize=(11, 4), sharex='col', sharey='row')

    
    im = ax[0].imshow(err[stepPlot, 0, :, :],  cmap="RdBu_r")
    ax[0].set_title('Error u\n($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)',fontsize=8)
    fig.colorbar(im, ax=ax[0], shrink=0.5).ax.tick_params(labelsize=8)


    # Encoded and decoded
    im = ax[1].imshow(err[stepPlot, 1, :, :], cmap="RdBu_r")
    ax[1].set_title('Error v\n($t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)',fontsize=8)
    fig.colorbar(im, ax=ax[1], shrink=0.5).ax.tick_params(labelsize=8)


    # Encoded, predicted and decoded
    im = ax[2].imshow(norm_err[stepPlot], cmap="RdBu_r")
    ax[2].set_title(r'Reconstruction error - ' + model_name + '( $t+$' + (str(stepPlot) if stepPlot > 1 else "") + '$t_c$)',fontsize=8)
    fig.colorbar(im, ax=ax[2], shrink=0.7).ax.tick_params(labelsize=8)

    print("norme erreur", np.linalg.norm(norm_err))
    print("erreur relative",np.linalg.norm(norm_err)/np.linalg.norm(pred1)) #a tracer évolution dans le temps
    
    ax[0].set_xlabel('x/c')
    ax[1].set_xlabel('x/c')
    ax[2].set_xlabel('x/c')
    ax[0].set_ylabel('y/c')
    ax[1].set_ylabel('y/c')
    ax[2].set_ylabel('y/c')

    # fig.set_tight_layout(True)

    if save_file != None:
        plt.savefig(save_file, bbox_inches = 'tight', dpi = dpi)
        
    folder, filename = os.path.split(save_file)
    savefile2 = os.path.join(folder, "evoltempo_normpred_" + filename)

    plt.figure()
    plt.plot(norms)
    plt.title("Evolution temporelle de la norme de l'erreur de prédiction",fontsize=8)
    plt.xlabel("Temps (snapshot index)",fontsize=8)
    plt.ylabel("norm(erreur)",fontsize=8)
    plt.savefig(savefile2)
    return fig, ax



def Ec(field):
    """Calcule énergie cinétique selon u et v séparément pour chaque snapshot"""
    Ek_u = np.sum(field[:, 0]**2, axis=(1, 2)) / 2
    Ek_v = np.sum(field[:, 1]**2, axis=(1, 2)) / 2
    return Ek_u, Ek_v

def Ec_tot(field): 
    """Calcule énergie cinétique totale du système"""
    return 0.5 * np.mean(field[:, 0, :, :]**2 + field[:, 1, :, :]**2, axis=(1, 2))


    
    
