#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            # 将inpaint_dir添加到group中
            if arg[0] in ['inpaint_dir', 'depth_dir']:
                setattr(group, arg[0], arg[1])
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self._unique_image=''
        self._new_image="image"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.remove_outliers_interval = 500  # [default 500]
        super().__init__(parser, "Optimization Parameters")

class SDSParams(ParamGroup):
    def __init__(self, parser):
        # diffusion part
        # ********** add **********
        self.save_guidance_path = 'dream_fusion.png'
        self.text_normal = 'A textured rock on the ground'
        self.text_depth = 'A textured rock on the ground'
        self.text = 'A textured rock on the ground'
        self.negative = ''
        self.image = ''
        self.guidance = ['SD']
        self.t_range = [0.02, 0.98]
        self.fp16 = False
        self.vram_O = False
        self.sd_version = '2.1'
        self.hf_key = None
        self.lambda_guidance = 1
        self.guidance_scale = 75
        super().__init__(parser, "SDS Parameters")

    ##diffusion part
    # #######************* add **********#############
    # parser.add_argument('--save_guidance_path', default='dream_fusion.png', type=str, help="save_guidance_path")
    # parser.add_argument('--text_normal', default='A stone bench on a grass ground', help="text prompt") # a cylindric stone pedestal on the grass in front of building walls and trees
    # parser.add_argument('--text_depth', default='A stone bench on a grass ground', help="text prompt") # a cylindric stone pedestal on the grass in front of building walls and trees
    # parser.add_argument('--text', default='A stone bench on a grass ground', help="text prompt") # a cylindric stone pedestal on the grass in front of building walls and trees
    # parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    # parser.add_argument('--image', default='', help="image prompt")
    #
    # parser.add_argument('--guidance', type=str, nargs='*', default=['SD'], help='guidance model')
    # parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help="stable diffusion time steps range")
    #
    # parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    # parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    # parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    # parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # parser.add_argument('--lambda_guidance', type=float, default=1, help="loss scale for SDS")
    # parser.add_argument('--guidance_scale', type=float, default=75, help="diffusion model classifier-free guidance scale")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
