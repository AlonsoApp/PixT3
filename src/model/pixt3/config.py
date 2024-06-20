from tools.general_config import load_parser, finish_args, save_config

def write_config_to_file(args, output_path):
    save_config(args, output_path)

def read_arguments():
    parser = load_parser()
    # We keep all config in general_config.py until we have other models to use in this project
    args = parser.parse_args()

    # Custom defaults
    #args.max_seq_length = args.max_seq_length if hasattr(args, 'max_seq_length') else 512

    return finish_args(args)
