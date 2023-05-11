import time
from logging import getLogger

from daisy.model.NGCFRecommender import NGCF
from daisy.utils.splitter import TestSplitter
from daisy.utils.metrics import calc_ranking_results
from daisy.utils.loader import RawDataReader, Preprocessor
from daisy.utils.config import init_seed, init_config, init_logger
from daisy.utils.sampler import BasicNegtiveSampler,SkipGramNegativeSampler
from daisy.utils.dataset import get_dataloader, BasicDataset, CandidatesDataset, AEDataset
from daisy.utils.utils import ensure_dir, get_ur, get_history_matrix, build_candidates_set, get_inter_matrix

model_config = {
    'ngcf': NGCF,
}
if __name__ == '__main__':
    ''' summarize hyper-parameter part (basic yaml + args + model yaml) '''
    config = init_config()

    ''' init seed for reproducibility '''
    init_seed(config['seed'], config['reproducibility'])

    ''' init logger '''
    #initialise un objet logger en appelant la fonction init_logger() et stocke l'objet logger dans une variable appelée logger.
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    config['logger'] = logger
    
    ''' Test Process for Metrics Exporting '''
    #on utilise un objet RawDataReader pour charger les données en mémoire.
    reader, processor = RawDataReader(config), Preprocessor(config)
    df = reader.get_data()
    #utilise un objet Preprocessor pour prétraiter les données
    df = processor.process(df)
    #stocker le nombre total d'utilisateurs et d'articles dans les variables user_num et item_num.
    user_num, item_num = processor.user_num, processor.item_num

    config['user_num'] = user_num
    config['item_num'] = item_num

    ''' Train Test split '''
    #on utilise un objet TestSplitter pour diviser les données en train et test  et récupère les index correspondant aux deux ensembles.
    splitter = TestSplitter(config)
    train_index, test_index = splitter.split(df)
    train_set, test_set = df.iloc[train_index, :].copy(), df.iloc[test_index, :].copy()

    ''' get ground truth '''
    #calcule les listes d'articles que chaque utilisateur a interagi dans les ensembles de test et d'entraînement en utilisant les fonctions get_ur() 
    # et stocke la liste des interactions d'entraînement dans la variable total_train_ur.
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)
    # stocke la liste des interactions d'entraînement dans la variable config['train_ur'].
    config['train_ur'] = total_train_ur

    ''' build and train model '''
    #on calcule le temps de début d'exécution du script
    s_time = time.time()

    
    if config['algo_name'].lower() in ['ngcf']:
        config['inter_matrix'] = get_inter_matrix(train_set, config)
    model = model_config[config['algo_name']](config)
    sampler = BasicNegtiveSampler(train_set, config)
    train_samples = sampler.sampling()
    train_dataset = BasicDataset(train_samples)
    train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    model.fit(train_loader)



    
    ''' build candidates set '''
    logger.info('Start Calculating Metrics...')
     #on construit l'ensemble de candidats pour chaque utilisateur  de test en appelant la fonction "build_candidates_set(test_ur, total_train_ur, config)".
     # Cette fonction utilise l'ensemble de référence d'entraînement total_train_ur pour trouver des candidats d'items qui ne sont pas encore vus par l'utilisateur de test test_ur.
    test_u, test_ucands = build_candidates_set(test_ur, total_train_ur, config)

    ''' get predict result '''
    logger.info('==========================')
    logger.info('Generate recommend list...')
    logger.info('==========================')
    # un ensemble de données de candidats est créé en utilisant les candidats construits
    test_dataset = CandidatesDataset(test_ucands)
    #un chargeur de données est créé à partir de cet ensemble de données.
    test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    #le modèle NGCF est utilisé pour classer les candidats en appelant la méthode "rank()" avec le chargeur de données de test_loader.
    #  Le résultat est stocké dans une variable "preds", qui est un tableau NumPy de forme (nombre d'utilisateurs, topk), où topk est le nombre de recommandations à fournir pour chaque utilisateur.
    preds = model.rank(test_loader) # np.array (u, topk)

    ''' calculating KPIs '''
    #Ce code calcule les performances de l'algorithme de recommandation pour les mesures de KPI (indicateurs clés de performance) telles que la précision
    logger.info('Save metric@k result to res folder...')
    #les résultats KPI seront sauvegardés dans un dossier "res" spécifique.
    result_save_path = f"./res/{config['dataset']}/{config['prepro']}/{config['test_method']}/"
    algo_prefix = f"{config['loss_type']}_{config['algo_name']}"
    common_prefix = f"with_{config['sample_ratio']}{config['sample_method']}"
    #La fonction "ensure_dir(result_save_path)" crée le répertoire s'il n'existe pas.
    ensure_dir(result_save_path)
    config['res_path'] = result_save_path
   #Enfin, les performances de l'algorithme sont calculées en utilisant la fonction "calc_ranking_results(test_ur, preds, test_u, config)", 
   # qui prend en entrée l'ensemble de test test_ur, les prédictions preds pour les candidats pour chaque utilisateur de test, l'ensemble des utilisateurs de test test_u et les paramètres de configuration config. 
    results = calc_ranking_results(test_ur, preds, test_u, config)
    #Les résultats sont stockés dans un DataFrame Pandas appelé "results", qui est ensuite sauvegardé sous forme de fichier CSV dans le chemin de sauvegarde des résultats précédemment construit.
    results.to_csv(f'{result_save_path}{algo_prefix}_{common_prefix}_kpi_results.csv', index=False)
                   
