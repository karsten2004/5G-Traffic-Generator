from model.GAN import GAN
import hydra
import pandas as pd
import torch


@hydra.main(config_path='config', config_name='config')
def inference(cfg):
    version = cfg.checkpoint.version
    epoch = cfg.checkpoint.epoch

    # Get columns for create conditions
    df = pd.read_csv(cfg.data.data_path)
    cols = df.columns[~df.columns.str.contains('Unnamed')]

    # Get datamodule for create conditions and fit scalers
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup(stage='inference')
    print('Load from ckpt')
    model = GAN.load_from_checkpoint(cfg.checkpoint.path)
    # Use the best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    print(f'Model loaded on {device}')
    model.eval()

    print('inference start')
    df = pd.DataFrame()
    # Generate data for each conditions
    for i, condition in enumerate(dm.conditions):
        z = model.sample_Z(cfg.checkpoint.batch_size, 300, 100).to(device)
        condition_tensor = model.create_condition(condition, len(z)).to(device)
        gen = model(z, condition_tensor)
        gen = gen.cpu().detach().numpy().squeeze()
        gen = dm.scalers[dm.cols[i]].inverse_transform(gen.reshape(-1, 1))
        gen = gen.reshape(-1)
        df[cols[i]] = gen
    df.to_csv(f'5GTGAN_V{version}_epoch_{epoch}.csv', index=False)


if __name__ == '__main__':
    inference()
