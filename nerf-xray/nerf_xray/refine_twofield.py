from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from nerf_xray.deformation_fields import BSplineField3d, BSplineField1d, BsplineTemporalDeformationField3d, BsplineTemporalDeformationField3dConfig
from tqdm import tqdm, trange
import tyro

def load_def_field(p: Path, old_ng: int, weight_nn_width: int):
    print(f'Loading from {p}')
    data = torch.load(Path(p))
    _data_f = {}
    _data_b = {}
    key_map_f = {}
    key_map_b = {}
    for key in data['pipeline'].keys():
        if 'deformation_field_f' in key:
            name = '.'.join(key.split('.')[2:])
            _data_f[name] = data['pipeline'][key]
            key_map_f[name] = key
        if 'deformation_field_b' in key:
            name = '.'.join(key.split('.')[2:])
            _data_b[name] = data['pipeline'][key]
            key_map_b[name] = key

    deformation_field_f = make_def_field(old_ng, weight_nn_width)
    deformation_field_f.load_state_dict(_data_f)
    deformation_field_b = make_def_field(old_ng, weight_nn_width)
    deformation_field_b.load_state_dict(_data_b)
    return deformation_field_f, deformation_field_b, key_map_f, key_map_b

def make_def_field(ng: int, weight_nn_width: int):
    config = BsplineTemporalDeformationField3dConfig(
        support_range=[(-1,1),(-1,1),(-1,1)],
        num_control_points=(ng,ng,ng),
        weight_nn_width=weight_nn_width
    )
    df2 = BsplineTemporalDeformationField3d(
        config=config
    )
    return df2

def main(
    ckpt_path: Path,
    old_resolution: int,
    new_resolution: int,
    old_nn_width: int,
    new_nn_width: int
):
    old_df_f, old_df_b, key_map_f, key_map_b = load_def_field(ckpt_path, old_resolution, old_nn_width)
    new_df_f = make_def_field(new_resolution, new_nn_width)
    new_df_b = make_def_field(new_resolution, new_nn_width)
    # send to cuda
    old_df_f = old_df_f.to('cuda')
    old_df_b = old_df_b.to('cuda')
    new_df_f = new_df_f.to('cuda')
    new_df_b = new_df_b.to('cuda')

    optimizer = torch.optim.AdamW(list(new_df_f.parameters())+list(new_df_b.parameters()), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.01, 1000)
    losses = []
    pbar = trange(1500)

    for i in pbar:
        optimizer.zero_grad()
        nq = new_resolution+1
        x = torch.linspace(-1, 1, nq)
        y = torch.linspace(-1, 1, nq)
        z = torch.linspace(-1, 1, nq)
        X,Y,Z = torch.meshgrid(x,y,z, indexing='ij')
        pos = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1).to('cuda')
        _t = (i%20)/20
        t = _t*torch.ones_like(pos[:,0])
        loss = pos.new_zeros(1)
        for old_df, new_df in zip([old_df_f, old_df_b], [new_df_f, new_df_b]):
            uA = old_df(pos, t)
            uB = new_df(pos, t)
            loss += torch.nn.functional.mse_loss(uA, uB)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        pbar.set_postfix({'loss':loss.item(), 'lr':scheduler.get_last_lr()[0]})

    fig, ax = plt.subplots(ncols=2, figsize=(10,4))
    with torch.no_grad():
        for i,t in enumerate(np.linspace(0,1,6)):
            z = torch.linspace(-1, 1, 50).to('cuda')
            pos = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=1).to('cuda')
            time = torch.ones_like(z)*t
            for k, (old_df, new_df) in enumerate(zip([old_df_f, old_df_b], [new_df_f, new_df_b])):
                u = old_df(pos, time) - pos
                ax[k].plot(z.cpu(), u[:,2].cpu(), label=f'{t:.2f}', ls='--', color=f'C{i}')
                u = new_df(pos, time) - pos
                ax[k].plot(z.cpu(), u[:,2].cpu(), label=f'{t:.2f}', color=f'C{i}')
    plt.tight_layout()
    plt.savefig(ckpt_path.with_name('def_field_refining.png'))
    plt.close()

    data = torch.load(ckpt_path)
    new_dict_f = new_df_f.state_dict()
    new_dict_b = new_df_b.state_dict()
    for key in key_map_f:
        data['pipeline'][key_map_f[key]] = new_dict_f[key].to('cuda')
    for key in key_map_b:
        data['pipeline'][key_map_b[key]] = new_dict_b[key].to('cuda')
    new_p = ckpt_path.with_name(ckpt_path.stem+'-mod.ckpt')
    torch.save(data, new_p)
    print(f'Modified checkpoint saved to: {new_p}')

if __name__=='__main__':
    tyro.cli(main)