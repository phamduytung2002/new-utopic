import argparse

def _parse_args():
    parser = argparse.ArgumentParser(description='Contrastive topic modeling')
    parser.add_argument('--epochs-1', default=100, type=int,
                        help='Number of training epochs for Stage 1')
    parser.add_argument('--epochs-2', default=10, type=int,
                        help='Number of training epochs for Stage 2')
    parser.add_argument('--bsz', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--dataset', default='news', type=str,
                        choices=['news', 'twitter', 'wiki', 'nips', 'stackoverflow', 'reuters', 'r52', 'imdb', 'agnews', 'yahoo'],
                        help='Name of the dataset')
    parser.add_argument('--n-cluster', default=50, type=int,
                        help='Number of clusters')
    parser.add_argument('--n-topic', type=int,
                        help='Number of topics. If not specified, use same value as --n-cluster')
    parser.add_argument('--n-word', default=2000, type=int,
                        help='Number of words in vocabulary')
    
    parser.add_argument('--base-model', type=str,
                        help='Name of base model in huggingface library.')
    
    parser.add_argument('--gpus', default=[0,1,2,3], type=int, nargs='+',
                        help='List of GPU numbers to use. Use 0 by default')
    
    parser.add_argument('--coeff-1-sim', default=1.0, type=float,
                        help='Coefficient for NN dot product similarity loss (Phase 1)')
    parser.add_argument('--coeff-1-dist', default=1.0, type=float,
                        help='Coefficient for NN SWD distribution loss (Phase 1)')
    parser.add_argument('--dirichlet-alpha-1', type=float,
                        help='Parameter for Dirichlet distribution (Phase 1). Use 1/n_topic by default.')
    
    parser.add_argument('--stage-1-ckpt', type=str,
                        help='Name of torch checkpoint file Stage 1. If this argument is given, skip Stage 1.')
    
    parser.add_argument('--coeff-2-recon', default=1.0, type=float,
                        help='Coefficient for VAE reconstruction loss (Phase 2)')
    parser.add_argument('--coeff-2-regul', default=1.0, type=float,
                        help='Coefficient for VAE KLD regularization loss (Phase 2)')
    parser.add_argument('--coeff-2-cons', default=1.0, type=float,
                        help='Coefficient for CL consistency loss (Phase 2)')
    parser.add_argument('--coeff-2-dist', default=1.0, type=float,
                        help='Coefficient for CL SWD distribution matching loss (Phase 2)')
    parser.add_argument('--dirichlet-alpha-2', type=float,
                        help='Parameter for Dirichlet distribution (Phase 2). Use same value as dirichlet-alpha-1 by default.')
    
    parser.add_argument('--stage-2-lr', default=2e-1, type=float,
                        help='Learning rate of phase 2')
    parser.add_argument('--stage-2-repeat', default=5, type=int,
                        help='Repetition count of phase 2')
    
    parser.add_argument('--result-file', type=str,
                        help='File name for result summary')
    parser.add_argument('--palmetto-dir', type=str, default='./',
                        help='Directory where palmetto JAR and the Wikipedia index are. For evaluation')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    return parser


def save_config(args, path):
    with open(path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')


def load_config(path):
    args = argparse.Namespace()
    with open(path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if value.isdigit():
                if value.find('.') != -1:
                    value = float(value)
                else:
                    value = int(value)
            setattr(args, key, value)
    print(args)
    return args

def save_config(args, path):
    with open(path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')


def load_config(path):
    args = argparse.Namespace()
    with open(path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if value.isdigit():
                if value.find('.') != -1:
                    value = float(value)
                else:
                    value = int(value)
            setattr(args, key, value)
    print(args)
    return args
