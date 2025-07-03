'''
File para testar as epocas e gerar as figuras
'''

from utils.utils import *
from runners.runners import *

def epoch_test():
    args = parse_arguments()
    if args.config:
        config_params = load_config_from_json(args.config)
        # Atualiza os parâmetros do argparse com os valores do JSON
        for key, value in config_params.items():
            setattr(args, key, value)

        df_epochs_f1 = pd.DataFrame()
        rate = 0.1

        models = ['RGCN', 'GCN']
        epochs_gae = list(range(1,200, 10))
        epochs_classifier = list(range(1,200,10))

        for model_name in models:
            dataset = torch.load(args.dataset_path, weights_only=False)
            dataset = Data(x = dataset[0]['x'], y = dataset[0]['y'], edge_index = dataset[0]['edge_index'])
            data = organize_data(data = dataset,
                                        L = args.L,
                                        rate = rate,
                                        positive_class = args.positive_class,
                                        name = args.dataset_name,
                                        alpha = args.alpha,
                                        beta = args.beta,
                                        gamma = args.gamma)
            for epoch1 in epochs_gae:
                for epoch2 in epochs_classifier:
                    model = get_model(model_name, data, L = args.L, activation_function = torch.relu, hid_dim = args.hid_dim, out_dim = args.out_dim)                
                    optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.001) 
                    train_gae(data = data, gae_model = model, optimizer = optimizer, epochs = epoch1)
                    data.N = gae_negative_inference(data, model, len(data.P))

                    df_aux2 = pu_classification(data, model, epochs = epoch2)
                    df_aux2['model'] = model_name
                    df_aux2['dataset'] = data.name
                    df_aux2['rate'] = rate
                    df_aux2['length negatives'] = len(data.N)
                    df_aux2['length positives'] = len(data.P)
                    df_aux2['epochs_neginf'] = epoch1
                    df_aux2['epochs_pu'] = epoch2
                    df_epochs_f1 = pd.concat([df_epochs_f1, df_aux2], ignore_index=True)
                    
                    df_epochs_f1.to_csv(f'results/epochs_results_{data.name}.csv')

                    model.reset_parameters()

epoch_test()