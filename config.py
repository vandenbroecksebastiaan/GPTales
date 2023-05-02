config = {
    "model":{
        "base_learning_rate":0.0001,
        "target":"ldm.models.diffusion.ddpm.LatentDiffusion",
        "params":{
            "linear_start":0.00085,
            "linear_end":0.012,
            "num_timesteps_cond":1,
            "log_every_t":200,
            "timesteps":1000,
            "first_stage_key":"jpg",
            "cond_stage_key":"txt",
            "image_size":64,
            "channels":4,
            "cond_stage_trainable":False,
            "conditioning_key":"crossattn",
            "monitor":"val/loss_simple_ema",
            "scale_factor":0.18215,
            "use_ema":False,
            "unet_config":{
                "target":"ldm.modules.diffusionmodules.openaimodel.UNetModel",
                "params":{
                    "use_checkpoint":True,
                    "use_fp16":False,
                    "image_size":32,
                    "in_channels":4,
                    "out_channels":4,
                    "model_channels":320,
                    "attention_resolutions":[
                        4,
                        2,
                        1
                    ],
                    "num_res_blocks":2,
                    "channel_mult":[
                        1,
                        2,
                        4,
                        4
                    ],
                    "num_head_channels":64,
                    "use_spatial_transformer":True,
                    "use_linear_in_transformer":True,
                    "transformer_depth":1,
                    "context_dim":1024,
                    "legacy":False
                }
            },
            "first_stage_config":{
                "target":"ldm.models.autoencoder.AutoencoderKL",
                "params":{
                    "embed_dim":4,
                    "monitor":"val/rec_loss",
                    "ddconfig":{
                        "double_z":True,
                        "z_channels":4,
                        "resolution":256,
                        "in_channels":3,
                        "out_ch":3,
                        "ch":128,
                        "ch_mult":[
                            1,
                            2,
                            4,
                            4
                        ],
                        "num_res_blocks":2,
                        "attn_resolutions":[
                            
                        ],
                        "dropout":0
                    },
                    "lossconfig":{
                        "target":"torch.nn.Identity"
                    }
                }
            },
            "cond_stage_config":{
                "target":"ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder",
                "params":{
                    "freeze":True,
                    "layer":"penultimate"
                }
            }
        }
    }
}