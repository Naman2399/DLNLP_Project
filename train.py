import argparse

def load_dataset(args) :
    if args.dataset_name == 'news' :
        from datasets.news_dataset import create_data_loader
        return create_data_loader(args)

    elif args.dataset_name == 'hugging_face' :
        from datasets.hugging_face_dataset import create_data_loader
        return create_data_loader(args)

    else :
        print(f'Enter a valida dataset name : {args.dataset_name}')
        exit()


def run(args)  :

    # Load Dataset
    train_loader, val_loader, test_loader, x_tokenizer, y_tokenizer = load_dataset(args)

    # Load Model

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type= str, default= 'news', help = 'Options : news, hugging_face')

    parser.add_argument('--summary_csv', type=str,
                        default='/mnt/hdd/karmpatel/naman/demo/DLNLP_Project_Data/news/news_summary.csv',
                        help="summary file contents")
    parser.add_argument('--raw_csv', type=str,
                        default='/mnt/hdd/karmpatel/naman/demo/DLNLP_Project_Data/news/news_summary_more.csv',
                        help="raw csv file details")

    parser.add_argument('--bs', type=int, default=32, help="Batch size ")

    args = parser.parse_args()

    print("-" * 20, "Arguments", "-" * 20)
    # Convert to dictionary and iterate
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    run(args)