
Molecular_energy_prediction
├─ dataAnalyse.ipynb :              # Visualisation et analyse statistiques du jeu de donéees 
│                                       - Visualisation d'un exemple d'une molécule
│                                       - Statistiques sur le dataset
│ 
├─ histogramme.ipynb :              # Prédiction de l'energie grace à l'histogramme du type d'atomes des molécules
│                                                                             
├─ spectreMatriceCoulomb.ipynb :     # Prédiction de l'energie grace au spectre de la matricde de Coulomb des molécules
│
│                                                                             
├─ contrastive.ipynb :     # Implémentation d'une méthode contrastive pour apprendre des représentations invariantes 
│                           par rotation/translations. Puis      prédictions sur ces représentations
│
│                                      
│
├─ regression_scatering_et_ou_contrastive.ipynb:  # Prédiction de l'énergie grace coef scatering et/ou embeding contrastive learning  (regession multi lineaire + contrastive learning)                       
|
|
|
├─visualisation_scatering.ipynb : # FIchier pour visualiser les lobes (convolution densité et filtre de scatering)
│                                  
├─ 📂 dataset/                         # Dataset QM7 et dataset non labelisé pour apprentissage contrastive learning
|  ├─ QM7XDataset.py
│  |
|  ├─ UnlabeledConstrastive.py
|
|
|
│                                  
├─ 📂 models/
│  ├─   contrastive_learning.py       # Encoder pour la méthode contrastive 
│  │                              
│  │                               
│  └─   regression_multi_lin.py       # Model de régression multi-linéaire      
│                                  
│                                  
│ 
└─  README.md                       # Ce fichier :
                                    - Présentation globale du projet
                                    - Structure des fichiers
       
