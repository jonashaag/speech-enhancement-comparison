from pytorch_sound.models import register_model_architecture


@register_model_architecture('spectrogram_unet', 'spectrogram_unet_comp')
def spec_unet_comp():
    return {
        'spec_dim': 513,
        'hidden_dim': 384,
        'filter_len': 1024,
        'hop_len': 256,
        'block_layers': 4,
        'layers': 5,
        'kernel_size': 3,
        'is_mask': True,
        'norm': 'bn',
        'act': 'comp'
    }


@register_model_architecture('wave_unet', 'wave_unet_base')
def wave_unet_base():
    return {
        'hidden_dim': 128,
        'kernel_size': 3,
        'layers': 4,
        'block_layers': 8
    }


@register_model_architecture('wave_unet', 'wave_unet_large')
def wave_unet_large():
    return {
        'hidden_dim': 256,
        'kernel_size': 3,
        'layers': 4,
        'block_layers': 8,
        'dropout': 0.1,
    }


@register_model_architecture('refine_spectrogram_unet', 'refine_unet_base')
def refine_unet_base():
    return {
        'spec_dim': 256 + 1,
        'hidden_dim': 768,
        'filter_len': 512,
        'hop_len': 64,
        'block_layers': 4,
        'layers': 4,
        'kernel_size': 3,
        'refine_layers': 1,
        'is_mask': True,
        'norm': 'ins',
        'act': 'comp'
    }


@register_model_architecture('refine_spectrogram_unet', 'refine_unet_base2')
def refine_unet_base2():
    return {
        'spec_dim': 256 + 1,
        'hidden_dim': 512,
        'filter_len': 512,
        'hop_len': 128,
        'block_layers': 5,
        'layers': 4,
        'kernel_size': 3,
        'refine_layers': 1,
        'is_mask': True,
        'norm': 'ins',
        'act': 'comp'
    }
