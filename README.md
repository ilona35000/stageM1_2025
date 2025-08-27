Français:

Ce code a été modifié à partir du clone du repository suivant : https://github.com/KTH-FlowAI/beta-Variational-autoencoders-and-transformers-for-reduced-order-modelling-of-fluid-flows

Lien de l'article: https://www.nature.com/articles/s41467-024-45578-4#data-availability

Vous trouverez les modifications apportées dans le rapport (fichiers lectre40_hdf5.py ou 100 et testStochastique.py)

Les données sont à télécharger à l'adresse suivante : https://zenodo.org/records/10501216
Attention! L'un des fichiers prend 31.7 GB d'espace mémoire et est donc long à télécharger.

main.py : certains paramètres peuvent être changés via le fichier job.sh, le détail des arguments possibles se trouve au début du main

Le script permettant d'exécuter le code via le calculateur Nautilus se trouve dans le fichier job.sh, depuis le terminal les commandes python suffisent.

Il y aura une incompatibilité entre les chemins du code et les chemins dans votre disque mais ça se modifie facilement.





English:

This code has been modified from the clone of the following repository:
https://github.com/KTH-FlowAI/beta-Variational-autoencoders-and-transformers-for-reduced-order-modelling-of-fluid-flows

Link to the article:
https://www.nature.com/articles/s41467-024-45578-4#data-availability

The modifications made are described in the report (files lectre40_hdf5.py, lectre40_hdf5_100.py, and testStochastique.py).

The datasets can be downloaded from the following address:
https://zenodo.org/records/10501216
Note: one of the files requires 31.7 GB of storage and therefore takes a long time to download.

main.py: some parameters can be changed via the job.sh file. A detailed list of possible arguments can be found at the beginning of main.py.

The script for running the code on the Nautilus cluster is provided in the file job.sh. From a terminal, direct python commands are sufficient.

There may be incompatibilities between the paths defined in the code and those on your local machine, but they can be easily updated.
