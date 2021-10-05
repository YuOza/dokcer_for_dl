import torch.utils.data
from torchvision.utils import save_image
from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
from dataloader import *

import os
from PIL import Image
lreq.use_implicit_lreq.set(True)


def place(canvas, image, x, y):
    im_size = image.shape[2]
    if len(image.shape) == 4:
        image = image[0]
    canvas[:, y: y + im_size, x: x + im_size] = image * 0.5 + 0.5


def save_sample(model, sample, i):
    os.makedirs('results', exist_ok=True)

    with torch.no_grad():
        model.eval()
        x_rec = model.generate(model.generator.layer_count - 1, 1, z=sample)

        def save_pic(x_rec):
            resultsample = x_rec * 0.5 + 0.5
            resultsample = resultsample.cpu()
            save_image(resultsample,
                       'sample_%i_lr.png' % i, nrow=16)

        save_pic(x_rec)


def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)
    model.cuda(0)
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_tl
    mapping_fl = model.mapping_fl
    dlatent_avg = model.dlatent_avg

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    extra_checkpoint_data = checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    def encode(x):
        Z, _ = model.encode(x, layer_count - 1, 1)
        Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
        return Z

    def decode(x):
        layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
        return model.decoder(x, layer_count - 1, 1, noise=True)

    rnd = np.random.RandomState(6)
    latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)

    for i in range(41):
        path = "./dataset_samples/STL10/"+str(i)+".png"
        image = Image.open(path)
        image = image.convert('RGB')
        image = image.resize((128,128))
        image = np.asarray(image, dtype=np.float32)
        image = np.transpose(image, (2,0,1))
        x = torch.tensor(image, device='cpu', requires_grad=True).cuda() / 127.5 - 1.
        if i == 0:
            input_image = torch.cat([x[None, ...].detach().cpu()], dim=3)
        else:
            input_image = torch.cat([input_image, x[None, ...].detach().cpu()], dim=0)

    def make(sample):
        canvas = []
        with torch.no_grad():
            sample = np.asarray(sample, dtype=np.float32)
            sample = np.transpose(sample, (2,0,1))
            x = torch.tensor(sample, device='cpu', requires_grad=True).cuda() / 127.5 - 1.
            if x.shape[0] == 4:
                x = x[:3]
            latents = encode(x[None, ...].cuda())
            f = decode(latents)
            r = torch.cat([x[None, ...].detach().cpu(), f.detach().cpu()], dim=3)
            canvas.append(r)
        return canvas
    
    output_image = input_image
    print(input_image.shape)
    with torch.no_grad():
        
        for i in range(5):
            for j in range(41):
                x = input_image[j] 
                latents = encode(x[None, ...].cuda())
                f = decode(latents)
                output_image = torch.cat([output_image,f.detach().cpu()], dim=0)
    

    file_name = 'make_figures/reco_STLvEX_all.png'
    save_image(output_image * 0.5 + 0.5,file_name , nrow=41)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-reconstruction-STL10', default_config='configs/STL10.yaml',
        world_size=gpu_count, write_log=False)
