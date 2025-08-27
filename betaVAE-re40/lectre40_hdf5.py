#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 11:07:06 2025

@author: irich2025
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os    

def explorer(h5obj, prefix=''):
    for key in h5obj:
        item = h5obj[key]
        path = f"{prefix}/{key}"
        print(f"{path} --> {type(item)}")
        if isinstance(item, h5py.Group):
            explorer(item, path)

# Ouvre le fichier
with h5py.File('Data2PlatesGap1Re40_Alpha-00_downsampled_v6.hdf5', 'r') as f:
    explorer(f)

with h5py.File('Data2PlatesGap1Re40_Alpha-00_downsampled_v6.hdf5', 'r') as f:
    UV = f['/UV'][:]
    mean = f['/mean'][:]
    std = f['/std'][:]
    print("UV shape :", UV.shape) 
    print("mean shape :", mean.shape) 
    print("std shape :", std.shape)

UV = UV * std + mean  # broadcasting automatique
#t = 0
t=1000

u = UV[t, :, :, 0]
v = UV[t, :, :, 1]

X, Y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))

plt.figure(figsize=(8, 6))
plt.quiver(X, Y, u, v, scale=50)  # ajuste 'scale' si les flèches sont trop petites/grandes
plt.title(f"Champ de vitesse à t={t}")
plt.gca().invert_yaxis()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

norm = np.sqrt(u**2 + v**2)
plt.imshow(norm, cmap='inferno')
plt.colorbar(label="||v||")
plt.title(f"Norme de la vitesse à t={t}")
plt.show()"""


frames = list(range(0, 1001, 10))
"""
# Animation UV
os.makedirs("LectData/frames_quiver", exist_ok=True)

for idx, t in enumerate(frames):
    fig, ax = plt.subplots(figsize=(8, 6))
    u = UV[t, :, :, 0]
    v = UV[t, :, :, 1]
    X, Y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
    ax.quiver(X, Y, u, v, scale=50)
    ax.set_title(f"Champ de vitesse t={t}")
    ax.invert_yaxis()
    plt.savefig(f"LectData/frames_quiver/frame_{idx:04d}.png")
    plt.close()

print("images sauvegardées dans frames_quiver/")

#Anim quiver uv sep
os.makedirs("LectData/frames_quiver_sep", exist_ok=True)

# Grille
X, Y = np.meshgrid(np.arange(UV.shape[2]), np.arange(UV.shape[1]))

for idx, t in enumerate(frames):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    u = UV[t, :, :, 0]
    v = UV[t, :, :, 1]

    # Quiver pour u seulement (flèches horizontales)
    ax1.quiver(X, Y, u, np.zeros_like(u), scale=50, color="red")
    ax1.set_title(f"Champ u, t={t}")
    ax1.invert_yaxis()

    # Quiver pour v seulement (flèches verticales)
    ax2.quiver(X, Y, np.zeros_like(v), v, scale=50, color="blue")
    ax2.set_title(f"Champ v, t={t}")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"LectData/frames_quiver_sep/frame_{idx:04d}.png")
    plt.close()

print(f"images sauvegardées dans frames_quiver_sep")


#Anim UV separes
os.makedirs("LectData/frames_uv_sep", exist_ok=True)

for idx, t in enumerate(frames):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    u = UV[t, :, :, 0]
    v = UV[t, :, :, 1]

    # Plot de u
    im1 = ax1.imshow(u, origin="lower", cmap="RdBu")
    ax1.set_title(f"Composante u, t={t}")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Plot de v
    im2 = ax2.imshow(v, origin="lower", cmap="RdBu")
    ax2.set_title(f"Composante v, t={t}")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"LectData/frames_uv_sep/frame_{idx:04d}.png")
    plt.close()

print("images sauvegardées dans frames_uv_sep/")

# Animation norme vitesse

os.makedirs("LectData/frames_norm", exist_ok=True)

vmax = np.max(np.sqrt(UV[:, :, :, 0]**2 + UV[:, :, :, 1]**2))

for idx, t in enumerate(frames):
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = np.sqrt(UV[t, :, :, 0]**2 + UV[t, :, :, 1]**2)
    im = ax.imshow(norm, cmap='inferno', vmin=0, vmax=vmax)
    ax.set_title(f"||v|| t={t}")
    plt.colorbar(im)
    plt.savefig(f"LectData/frames_norm/frame_{idx:04d}.png")
    plt.close()

print("images sauvegardées dans frames_norm")


#Vorticité w
def vorticity_2d(u, v, dx, dy):
    """
    Calcule la vorticité scalaire pour un champ 2D (u,v).
    Convention: omega = dv/dx - du/dy
    """
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    return dv_dx - du_dy
    """
    Calcule la vorticité ω = dv/dx - du/dy
    avec différences finies centrées.
    Schéma décentré utilisé aux bords.
    """
    """ny, nx = u.shape
    w = np.zeros_like(u)

    # dv/dx
    dv_dx = np.zeros_like(v)
    dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)         # centré
    dv_dx[:, 0]    = (v[:, 1] - v[:, 0]) / dx                  # avant
    dv_dx[:, -1]   = (v[:, -1] - v[:, -2]) / dx                # arrière

    # du/dy
    du_dy = np.zeros_like(u)
    du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)         # centré
    du_dy[0, :]    = (u[1, :] - u[0, :]) / dy                  # avant
    du_dy[-1, :]   = (u[-1, :] - u[-2, :]) / dy                # arrière

    return dv_dx - du_dy"""


os.makedirs("LectData/frames_vort", exist_ok=True)
dx=1.0
dy=1.0
#gestion des min, max
wmin, wmax = np.inf, -np.inf
for t in frames:
    u = UV[t, :, :, 0]
    v = UV[t, :, :, 1]
    w = vorticity_2d(u, v, dx, dy)
    wmin = min(wmin, w.min())
    wmax = max(wmax, w.max())

# Échelle symétrique autour de 0
wabs = max(abs(wmin), abs(wmax))
vmin, vmax = -wabs, wabs


for idx, t in enumerate(frames):
    fig, ax = plt.subplots(figsize=(8, 6))
    u = UV[t, :, :, 0]
    v = UV[t, :, :, 1]
    w = vorticity_2d(u, v, dx, dy)

    im = ax.imshow(w, origin='upper', vmin=vmin, vmax=vmax)
    ax.set_title(f"Vorticité ω — t={t}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("ω")

    # Option: superposer quelques vecteurs vitesse clairsemés
    # (décommente si tu veux par-dessus la vorticité)
    # step = max(1, min(u.shape)//32)
    # X, Y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
    # ax.quiver(X[::step, ::step], Y[::step, ::step],
    #           u[::step, ::step], v[::step, ::step], scale=50)

    plt.tight_layout()
    plt.savefig(f"LectData/frames_vort/vort_{idx:04d}.png", dpi=120)
    plt.close()

print("Images de vorticité sauvegardées dans LectData/frames_vort/")
