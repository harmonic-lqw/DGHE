import numpy as np
import torch


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs, f"{x_shape[0]}, {t.shape}"
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out



def denoising_step(xt, t, t_next, *,
                   models,
                   logvars,
                   b,
                   sampling_type='ddim',
                   eta=0.0,
                   learn_sigma=False,
                   index=None,
                   t_edit=0,
                   hs_coeff=(1.0),
                   delta_h=None,
                   use_mask=False,
                   dt_lambda=1,
                   ignore_timestep=False,
                   image_space_noise=0,
                   dt_end = 999,
                   warigari=False,
                   conditions=None,
                   gan_direction=None,
                   plist=None,
                   w=None,
                   ):

    # Compute noise and variance
    model = models

    # et表示无编辑条件预测噪声，et_modified表示编辑条件下模型预测的噪声
    et, et_modified, delta_h, middle_h = model(xt, t, index=index, t_edit=t_edit, conditions=conditions, plist=plist, hs_coeff=hs_coeff, w=w, delta_h=delta_h, gan_direction=gan_direction, ignore_timestep=ignore_timestep, use_mask=use_mask)

    if learn_sigma:
        et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
        if index is not None:
            et_modified, _ = torch.split(et_modified, et_modified.shape[1] // 2, dim=1)
        logvar = logvar_learned
    else:
        logvar = extract(logvars, t, xt.shape)

    if type(image_space_noise) != int:
        if t[0] >= t_edit:
            index = 0
            if type(image_space_noise) == torch.nn.parameter.Parameter:
                et_modified = et + image_space_noise * hs_coeff[1]
            else:
                # print(type(image_space_noise))
                temb = models.module.get_temb(t)
                et_modified = et + image_space_noise(et, temb) * 0.01

    # Compute the next x
    bt = extract(b, t, xt.shape)
    at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)
    # 生成过程的最后一步: t_next == -1
    if t_next.sum() == -t_next.shape[0]:
        at_next = torch.ones_like(at)
    else:
        at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)

    xt_next = torch.zeros_like(xt)
    if sampling_type == 'ddpm':
        weight = bt / torch.sqrt(1 - at)

        mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)
        noise = torch.randn_like(xt)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
        xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
        xt_next = xt_next.float()

    elif sampling_type == 'ddim':
        # DDIM扩散过程
        # index标记得到的x0预测是原还是编辑；index=None -> 得到第t步原x0预测、index=1 -> 得到第t步编辑的x0
        # index表示有index+1个属性需要同时编辑，index=none表示不进行编辑
        if index is not None:
            x0_t = (xt - et_modified * (1 - at).sqrt()) / at.sqrt()
        else:
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        # Deterministic.
        # 在这里如果xt_next=t+1(原本next表示t-1),其实就是Inversion Process，就是一个加噪的过程
        # DDIM公式中的η，在编辑（1）和去噪（2）步骤时都是0，只有DDPM过程（3）才是1
        if eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        # Add noise. When eta is 1 and time step is 1000, it is equal to ddpm.
        # DDPM reverse process
        else:
            c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)

    if dt_lambda != 1 and t[0] >= dt_end:
        xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et * dt_lambda

    # Asyrp & DiffStyle
    if not warigari or index is None:
        return xt_next, x0_t, delta_h, middle_h

    # Warigari by young-hyun, Not in the paper
    else:
        # will be updated
        return xt_next, x0_t, delta_h, middle_h
    
