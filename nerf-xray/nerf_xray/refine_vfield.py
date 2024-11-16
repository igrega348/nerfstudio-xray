from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from nerf_xray.deformation_fields import BsplineTemporalIntegratedVelocityField3dConfig, BsplineTemporalIntegratedVelocityField3d
from tqdm import tqdm, trange
import tyro

def load_def_field(p: Path, old_ng: int, weight_nn_width: int):
    print(f'Loading from {p}')
    data = torch.load(Path(p))
    _data = {}
    key_map = {}
    for key in data['pipeline'].keys():
        if 'deformation' in key:
            _data[key.split('deformation_field.')[1]] = data['pipeline'][key]
            key_map[key.split('deformation_field.')[1]] = key
    data = _data

    deformation_field = make_def_field(old_ng, weight_nn_width)
    deformation_field.load_state_dict(data)
    return deformation_field, key_map

def make_def_field(ng: int, weight_nn_width: int):
    df2 = BsplineTemporalIntegratedVelocityField3dConfig(
        support_range=[(-1,1),(-1,1),(-1,1)],
        num_control_points=(ng,ng,ng),
        weight_nn_width=weight_nn_width
    ).setup()
    return df2

def main(
    ckpt_path: Path,
    old_resolution: int,
    new_resolution: int,
    old_nn_width: int,
    new_nn_width: int
):
    old_df, key_map = load_def_field(ckpt_path, old_resolution, old_nn_width)
    new_df = make_def_field(new_resolution, new_nn_width)
    # send to cuda
    old_df = old_df.to('cuda')
    new_df = new_df.to('cuda')

    optimizer = torch.optim.AdamW(new_df.parameters(), lr=1e-2)
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

        t = torch.rand(1).to('cuda')

        x0, x1, x2 = pos[:,0], pos[:,1], pos[:,2]
        phi = old_df.weight_nn(t.view(-1,1)).view(*old_df.bspline_field.grid_size, 3)
        uA = old_df.disp_func(x0, x1, x2, phi_x=phi)

        phi = new_df.weight_nn(t.view(-1,1)).view(*new_df.bspline_field.grid_size, 3)
        uB = new_df.disp_func(x0, x1, x2, phi_x=phi)

        # uA = old_df(pos, t, tf)
        # uB = new_df(pos, t, tf)

        loss = torch.nn.functional.mse_loss(uA, uB)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        pbar.set_postfix({'loss':loss.item(), 'lr':scheduler.get_last_lr()[0]})

    with torch.no_grad():
        for i,t in enumerate(np.linspace(0,1,6)):
            z = torch.linspace(-1, 1, 50).to('cuda')
            pos = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=1).to('cuda')
            time = torch.ones_like(z)*t
            u = old_df(pos, time, 1.0) - pos
            plt.plot(z.cpu(), u[:,2].cpu(), label=f'{t:.2f}', ls='--', color=f'C{i}')
            u = new_df(pos, time, 1.0) - pos
            plt.plot(z.cpu(), u[:,2].cpu(), label=f'{t:.2f}', color=f'C{i}')
    plt.savefig(ckpt_path.with_name('def_field_refining.png'))
    plt.close()

    data = torch.load(ckpt_path)
    new_dict = new_df.state_dict()
    for key in key_map:
        data['pipeline'][key_map[key]] = new_dict[key].to('cuda')
    new_p = ckpt_path.with_name(ckpt_path.stem+'-mod.ckpt')
    torch.save(data, new_p)
    print(f'Modified checkpoint saved to: {new_p}')

if __name__=='__main__':
    tyro.cli(main)