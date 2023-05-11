#Voici les importations nécessaires
# ast est un module en Python qui fournit des fonctionnalités pour travailler avec les arbres de syntaxe abstraite
from ast import Global
#json est un module Python intégré qui fournit des méthodes pour travailler avec des données JSON
import json
#optuna est un package externe utilisé pour l'optimisation des hyperparamètres
import optuna
#numpy est une bibliothèque de calcul numérique en Python
import numpy as np
#ogging est un module Python intégré qui fournit des fonctionnalités de journalisation.
from logging import getLogger


#importe la classe NGCF du module NGCFRecommender du package daisy.model.
from daisy.model.NGCFRecommender import NGCF

#Ces lignes importent différents modules et fonctions utilitaires du package daisy.utilss, , , l'échantillonnage et les utilitaires de préparation de jeux de données
from daisy.utils.loader import RawDataReader, Preprocessor #utiles pour le chargement des données
from daisy.utils.splitter import TestSplitter, ValidationSplitter# la division ded données
from daisy.utils.config import init_seed, init_config, init_logger#la configuration
from daisy.utils.metrics import MAP, NDCG, Recall, Precision, HR, MRR #les métriques d'évaluation
from daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler #l'échantillonnage
#AEDataset est utilisé pour préparer les données d'apprentissage pour les modèles de complétion automatique d'items, tandis que BasicDataset est utilisé pour préparer les données d'apprentissage pour les modèles de recommandation basés sur l'interaction utilisateur-article. CandidatesDataset est utilisé pour préparer les données d'apprentissage pour les modèles de recommandation basés sur les candidats.La fonction get_dataloader est utilisée pour créer un objet DataLoader à partir des ensembles de données préparés, qui est utilisé pour itérer efficacement sur les ensembles de données lors de l'entraînement des modèles.
from daisy.utils.dataset import AEDataset, BasicDataset, CandidatesDataset, get_dataloader
#get_history_matrix est utilisé pour préparer la matrice d'historique de l'utilisateur-article à partir des données brutes d'interaction utilisateur-article.get_ur est utilisé pour obtenir un dictionnaire des articles que chaque utilisateur a interagi avec.build_candidates_set est utilisé pour construire un ensemble de candidats pour chaque utilisateur en fonction des articles auxquels il a interagi.ensure_dir est utilisé pour s'assurer qu'un répertoire existe et, sinon, le créer.get_inter_matrix est utilisé pour préparer une matrice d'interaction utilisateur-article à partir des données brutes d'interaction utilisateur-article.
from daisy.utils.utils import get_history_matrix, get_ur, build_candidates_set, ensure_dir, get_inter_matrix

model_config = {
    

    'ngcf': NGCF,
    
}

#les metrics utilisées pour calculer les performances du modèle de recommandation.
metrics_config = {
    "recall": Recall,
    "mrr": MRR,
    "ndcg": NDCG,
    "hr": HR,
    "map": MAP,
    "precision": Precision,
}
#tune_params_config est un dictionnaire de configuration clé est 'ngcf', qui fait référence au modèle de recommandation NGCF, et la valeur est une liste de noms de paramètres à optimiser lors de l'entraînement du modèle
tune_params_config = {
    
    'ngcf': ['num_ng', 'factors', 'node_dropout', 'mess_dropout', 'batch_size', 'lr', 'reg_1', 'reg_2'],
    
}
#spécifier les types de données des paramètres 
param_type_config = {
    
    'factors': 'int',#les facteurs latents
    'num_ng': 'int',#e nombre de voisins
    'lr': 'float',#le taux d'apprentissage
    'batch_size': 'int',#la taille du lot
    'reg_1': 'float',#la régularisation L1 
    'reg_2': 'float',#la régularisation L2 
    'node_dropout': 'float',#e dropout de noeud 
    'mess_dropout': 'float',#e dropout de message
    
}
# TRIAL_CNT utilisée pour suivre le nombre d'itérations ou de tentatives effectuées lors de l'optimisation des hyperparamètres à l'aide de la bibliothèque externe Optuna. À chaque tentative, cette variable est incrémentée de 1 pour garder une trace du nombre total de tentatives effectuées jusqu'à présent.
TRIAL_CNT = 0 #on l'initialise à 0

if __name__ == '__main__':#vérifier si le nom du module principal est "main". Cela signifie que ce code ne sera exécuté que si ce fichier est exécuté directement et non s'il est importé dans un autre fichier.
    ''' summarize hyper-parameter part (basic yaml + args + model yaml) '''
    config = init_config() #est appelée pour lire les fichiers de configuration et fusionner les paramètres avec les arguments en ligne de commande. La configuration globale est stockée dans le dictionnaire 'config'.

    ''' init seed for reproducibility '''
    init_seed(config['seed'], config['reproducibility'])

    ''' init logger '''
    init_logger(config) #est appelée pour initialiser un objet logger. La configuration globale est passée en argument pour configurer le comportement du logger.
    logger = getLogger()#est appelée pour récupérer l'objet logger initialisé. 
    logger.info(config)#Les informations de configuration sont enregistrées dans le fichier de log.
    config['logger'] = logger

    ''' unpack hyperparameters to tune '''
    param_dict = json.loads(config['tune_pack'])#Le dictionnaire 'param_dict' est créé à partir de la chaîne JSON 'tune_pack' stockée dans la configuration globale. 
    algo_name = config['algo_name']#La clé 'algo_name' est utilisée pour extraire les noms des paramètres à optimiser pour l'algorithme de recommandation NGCF.
    kpi_name = config['optimization_metric']#La variable 'kpi_name' est initialisée avec le nom de la métrique d'optimisation à utiliser pour l'optimisation des hyperparamètres
    tune_param_names = tune_params_config[algo_name]#Les noms des paramètres à optimiser sont stockés dans la variable 'tune_param_names' en utilisant la clé 'algo_name' pour extraire la liste de noms de paramètres correspondants à l'algorithme NGCF

    ''' open logfile to record tuning process '''
    # begin tuning here
    tune_log_path = './tune_res/'#  le chemin du répertoire de stockage des fichiers pour l'optimisation des hyperparamètres.
    ensure_dir(tune_log_path)#s'assurer que le répertoire existe. Si le répertoire n'existe pas, la fonction le crée.

 #crée l'en-tête du fichier CSV qui comprend les noms des hyperparamètres à optimiser et le nom de la métrique à optimiser. 
    f = open(tune_log_path + f"best_params_{config['loss_type']}_{config['algo_name']}_{config['dataset']}_{config['prepro']}_{config['val_method']}.csv", 'w', encoding='utf-8')
    line = ','.join(tune_param_names) + f',{kpi_name}'
    f.write(line + '\n')
    f.flush()# la fonction flush() est appelée pour s'assurer que toutes les données sont écrites dans le fichier CSV.

    ''' Test Process for Metrics Exporting '''
    reader, processor = RawDataReader(config), Preprocessor(config)#Initialisation avec la configuration donnée.
    df = reader.get_data()#Récupère les données
    df = processor.process(df)#Effectue le prétraitement des données à l'aide de l'objet Preprocessor.
    # Récupère le nombre d'utilisateurs et d'articles dans les données prétraitées.
    user_num, item_num = processor.user_num, processor.item_num

    config['user_num'] = user_num#Stocke le nombre d'utilisateurs dans la configuration.
    config['item_num'] = item_num#Stocke le nombre d'articles dans la configuration.

    ''' Train Test split '''#division en train et test.
    splitter = TestSplitter(config)#création d'une instance de la classe TestSplitter en utilisant la configuration du modèle.
    train_index, test_index = splitter.split(df)#divise les données du DataFrame en deux parties, l'ensemble de formation (train_index) et l'ensemble de test (test_index), en utilisant la méthode split() de l'objet splitter
    train_set, test_set = df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()# création de deux nouveaux DataFrames, train_set et test_set, qui correspondent aux données d'apprentissage et de test respectivement, Cela est réalisé en utilisant la méthode iloc() pour sélectionner les lignes des DataFrames d'origine correspondant aux index générés par la méthode split(), La méthode copy() est utilisée pour éviter que les modifications apportées à l'un des DataFrames n'affectent l'autre.

    ''' define optimization target function '''
    def objective(trial):
        global TRIAL_CNT
        for param in tune_param_names:# parcourt de la liste des noms des hyperparamètres à optimiser.
            if param not in param_dict.keys(): continue #Si le nom de l'hyperparamètre n'est pas présent dans le dictionnaire param_dict, la boucle passe à l'hyperparamètre suivant.
                
            if isinstance(param_dict[param], list):#Si l'hyperparamètre est une liste, une valeur est tirée de manière aléatoire à partir de la liste des valeurs possibles de l'hyperparamètre en utilisant la méthode suggest_categorical de l'objet trial.
                config[param] = trial.suggest_categorical(param, param_dict[param])
            elif isinstance(param_dict[param], dict):#Si l'hyperparamètre est un dictionnaire, la méthode suggest_int ou suggest_float est appelée en fonction du type de l'hyperparamètre (int ou float) et des bornes et de l'incrément fournis dans le dictionnaire param_dict.
                if param_type_config[param] == 'int':
                    step = param_dict[param]['step']
                    config[param] = trial.suggest_int(
                        param, param_dict[param]['min'], param_dict[param]['max'], 1 if step is None else step)
                elif param_type_config[param] == 'float':
                    config[param] = trial.suggest_float(
                        param, param_dict[param]['min'], param_dict[param]['max'], step=param_dict[param]['step'])
                else:
                    raise ValueError(f'Invalid parameter type for {param}...')
            else:#Si l'hyperparamètre n'est ni une liste ni un dictionnaire, une erreur est levée.
                raise ValueError(f'Invalid parameter settings for {param}, Current is {param_dict[param]}...')
        
        #ont divise l'ensemble d'entraînement en ensembles d'entraînement et de validation pour effectuer la validation croisé
        ''' user train set to get validation combinations and build model for each dataset '''
        splitter = ValidationSplitter(config)#initialisation de l'objet ValidationSplitter avec la configuration config.
        cnt, kpis = 1, []#ont Initialise les variables cnt et kpis à 1 et une liste vide .
        for train_index, val_index in splitter.split(train_set):#parcourt les ensembles d'entraînement et de validation créés par splitter.split(train_set).
            #récupère les lignes correspondant aux index des ensembles d'entraînement et de validation et crée des copies des deux ensembles de données.
            train, validation = train_set.iloc[train_index, :].copy(), train_set.iloc[val_index, :].copy()

            ''' get ground truth '''
            val_ur = get_ur(validation)#Récupère tous les éléments uniques de l'ensemble de validation pour chaque utilisateur sous forme de dictionnaire, avec les ID utilisateur comme clé et les ID article comme valeurs.
            train_ur = get_ur(train)#Récupère tous les éléments uniques de l'ensemble d'entraînement 
            config['train_ur'] = train_ur#Stocke le dictionnaire train_ur dans la configuration config pour une utilisation ultérieure.




            ''' build and train model '''
           
            
           
            #on vérifie si le nom de l'algorithme présent dans la configuration (variable config) 
            
            if config['algo_name'].lower() in ['ngcf']:
                # Si c'est le cas, la variable config['inter_matrix'] est initialisée avec une matrice intermédiaire obtenue en appelant la fonction get_inter_matrix(train, config).
                config['inter_matrix'] = get_inter_matrix(train, config)
            model = model_config[config['algo_name']](config)#Le modèle est sélectionné en fonction de la valeur de 'algo_name'
            sampler = BasicNegtiveSampler(train, config)#on initialise  objet(d'échantillonnage ) BasicNegtiveSampler en utilisant l'ensemble d'entraînement train et la configuration config.
            #on crée des échantillons d'entraînement en appelant la méthode sampling() de l'objet sampler.
            train_samples = sampler.sampling()
            # crée un ensemble de données d'entraînement en utilisant les échantillons d'entraînement créés précédemment.
            train_dataset = BasicDataset(train_samples)
            #crée un chargeur de données (dataloader) pour l'ensemble de données d'entraînement. Le chargeur de données est créé en utilisant la fonction get_dataloader qui prend en entrée l'ensemble de données d'entraînement, la taille des lots (batch_size), un indicateur pour mélanger aléatoirement les données (shuffle=True) et le nombre de travailleurs (num_workers=4) utilisés pour charger les données.
            train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
            model.fit(train_loader)# entraîne le modèle sur 20 époques

            
            #l'expérience de validation d'entraînement est terminée 
            logger.info(f'Finish {cnt} train-validation experiment(s)...')
            cnt += 1#La valeur de cnt est incrémentée de 1 à chaque fois que le code est exécuté

            ''' build candidates set '''
            logger.info('Start Calculating Metrics...')
            val_u, val_ucands = build_candidates_set(val_ur, train_ur, config)

            ''' get predict result '''
            logger.info('==========================')
            logger.info('Generate recommend list...')
            logger.info('==========================')
            val_dataset = CandidatesDataset(val_ucands)
            val_loader = get_dataloader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
            preds = model.rank(val_loader) #fait des prédictions pour les utilisateurs de validation (val_u) en appelant la méthode rank() 

            ''' calculating KPIs '''
            kpi = metrics_config[kpi_name](val_ur, preds, val_u)
            kpis.append(kpi)
        
        TRIAL_CNT += 1
        logger.info(f'Finish {TRIAL_CNT} trial...')

        return np.mean(kpis)

    ''' init optuna workspace '''
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2022))
    study.optimize(objective, n_trials=config['hyperopt_trail'])

    ''' record the best choices '''
    logger.info(f'Trial {study.best_trial.number} get the best {kpi_name}({study.best_trial.value}) with params: {study.best_trial.params}')
    line = ','.join([str(study.best_params[param]) if param in param_dict.keys() else str(config[param]) for param in tune_param_names]) + f',{study.best_value:.4f}\n'
    f.write(line)
    f.flush()
    f.close()

