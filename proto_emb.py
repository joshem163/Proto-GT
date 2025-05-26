
from modules import *
from models import objectview  # Model architectures (GCN, SAGE, GAT, MLP) import

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main(args):
    # Convert args to an object-like structure for easier access
    if isinstance(args, dict):
        args = objectview(args)
    # print(args)
    dataset = load_data(args.dataset, None)
    if args.dataset in ['chameleon','squirrel']:
        data = load_Sq_Cha_filterred(args.dataset)  # chameleon and squirrel filtared version
    else:
        data = dataset[0]  # First graph data object


    print(data)

    # Load dataset embeddings using proto embeddings based on the dataset choice
    if args.dataset in ['pubmed','wikics','roman-empire','amazon-ratings','questions']:
        wise_pca = Contextual(data)
        wise = wise_embeddings_eucledian(data)  # Euclidean embeddings for PubMed
        wise_fe1 = torch.tensor(wise)  # Convert embedding to PyTorch tensor
        wise_fe2 = torch.tensor(wise_pca)
        spatial = spatial_embeddings(data)
        spatial_torch = torch.tensor(spatial)
        # Concatenate both sets of embeddings
        proto_domain = torch.cat((spatial_torch,wise_fe1, wise_fe2), 1).float()
    else:
        spatial = spatial_embeddings(data)
        spatial_torch=torch.tensor(spatial)
        wise = wise_embeddings(data)
        Inc_fe = torch.tensor(wise[0])
        sel_fe = torch.tensor(wise[1])
        proto_domain = torch.cat((spatial_torch,Inc_fe, sel_fe), 1).float()

    proto_domain = proto_domain.to(device) # proto embeddings
    torch.save(proto_domain, f'proto_{args.dataset}.pt') # save the proto embeddings



if __name__ == "__main__":
    #datasets_to_run = [ 'cora','computers','photo','physics','cs', 'wikics','pubmed']  # Add more as needed
    datasets_to_run = ['cora']
    for ds in datasets_to_run:
        print(f"\nRunning on dataset: {ds}")
        args = {
            'dataset': ds,
        }

        start_time = time.time()
        main(args)
        end_time = time.time()  # End timing
        total_time = end_time - start_time
        print(f"\nTotal runtime: {total_time:.2f} seconds")


