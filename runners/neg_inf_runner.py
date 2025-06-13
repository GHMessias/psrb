'''
Arquivo para rodar somente os experimentos envolvendo DropEdge, SkipEdge (para inferência de negativos somente) 
e GAT, GraphSage para a inferência dos negativos.
'''
from utils.utils import *
from runners.runners import *

def neg_inf():
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
            
            # TODO: alterar a função get_model para pegar os modelos com dropedge e skipnode.
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

            # TODO: Criar o código de avaliação dos reliable negatives

    return