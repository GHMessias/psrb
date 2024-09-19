'''
This file run all the experiments of the paper. You can modify the models and dataset used in this experiment looking at the github repository.

Arguments:
L: number of iterations of the rewiring model
models: models to use (the models should be in the utils.getmodel function)
'''

from utils.utils import *
from runners.runners import *

def main():

    args = parse_arguments()
    if args.config:
        config_params = load_config_from_json(args.config)
        # Atualiza os parâmetros do argparse com os valores do JSON
        for key, value in config_params.items():
            setattr(args, key, value)
    
    df_pu_classify = pd.DataFrame()
    for _ in range(args.sample):
        for rate in args.rates:
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
            
            for model_name in args.model_names:
                # Defining the model to be used
                model = get_model(model_name, data, L = args.L, activation_function = torch.relu, hid_dim = args.hid_dim, out_dim = args.out_dim)

                # Searching for reliable negatives based on the model class
                if isinstance(model, (CCRNE, LP_PUL, MCLS, PU_LP, RCSVM)):
                    model.train()
                    data.N = model.negative_inference(num_neg = len(data.P))
            
                if isinstance(model, GAE):
                    optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.001) 
                    train_gae(data = data, gae_model = model, optimizer = optimizer, epochs = 100)
                    data.N = gae_negative_inference(data, model, len(data.P))
    
            
                # Setting the parameters to PU task
                if not args.neg_inf_only:
                    df_aux2 = pu_classification(data, model)
                    df_aux2['model'] = model_name
                    df_aux2['dataset'] = data.name
                    df_aux2['rate'] = rate
                    df_aux2['length negatives'] = len(data.N)
                    df_aux2['length positives'] = len(data.P)
                    df_pu_classify = pd.concat([df_pu_classify, df_aux2], ignore_index=True)
                    
                    df_pu_classify.to_csv(f'results/pu_classify_results_{data.name}.csv')


main()