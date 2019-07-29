import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn import constant_init, kaiming_init

from mmdet.ops import DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer

from collections import OrderedDict

BN_MOMENTUM = 0.1


class BasicBlock_bak(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=1,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 normalize=dict(type='BN'),
                 dcn=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, "Not implemented yet."

        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            bias=False
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            3,
            padding=1,
            bias=False
        )
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        # out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=1,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 normalize=dict(type='BN'),
                 dcn=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, "Not implemented yet."

        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=1,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 normalize=dict(type='BN'),
                 dcn=None):
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        self.inplanes = inplanes
        self.planes = planes
        self.conv_cfg = conv_cfg
        self.normalize = normalize
        self.dcn = dcn
        self.with_dcn = dcn is not None
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(normalize, planes*self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False
        )
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False
            )
        else:
            assert conv_cfg is None, "conv_cfg must be None for DCN"
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27


            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation
            )
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False
            )
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.normalize = normalize


    @property
    def norm1(self):
        return getattr(self, self.norm1_name)


    @property
    def norm2(self):
        return getattr(self, self.norm2_name)


    @property
    def norm3(self):
        return getattr(self, self.norm3_name)


    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if not self.with_dcn:
                out = self.conv2(out)
            elif self.with_modulated_dcn:
                offset_mask = self.conv2_offset(out)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.conv2(out, offset)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, normalization=None, multi_scale_output=True, conv_cfg=None, dcn=None):
        super(HighResolutionModule, self).__init__()
        self.conv_cfg = conv_cfg
        self.normalization = normalization if normalization is not None else dict(type='BN', frozen=False)
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)
        self.dcn = dcn


    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            print("[ERROR] ", error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            print("[ERROR]", error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            print("[ERROR]", error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            _, norm = build_norm_layer(self.normalization, num_channels[branch_index]*block.expansion)
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]*block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                norm,
                # nn.BatchNorm2d(num_channels[branch_index]*block.expansion, momentum=BN_MOMENTUM)
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    _, norm = build_norm_layer(self.normalization, num_inchannels[i])
                    fuse_layer.append(nn.Sequential(
                        build_conv_layer(
                            self.conv_cfg,
                            num_inchannels[j],
                            num_inchannels[i],
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False
                        ),
                        norm,
                        # nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            _, norm = build_norm_layer(self.normalization, num_outchannels_conv3x3)
                            conv3x3s.append(nn.Sequential(
                                build_conv_layer(
                                    self.conv_cfg,
                                    num_inchannels[j],
                                    num_outchannels_conv3x3,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False
                                ),
                                norm,))
                                # nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            _, norm = build_norm_layer(self.normalization, num_outchannels_conv3x3)
                            conv3x3s.append(nn.Sequential(
                                build_conv_layer(
                                    self.conv_cfg,
                                    num_inchannels[j],
                                    num_outchannels_conv3x3,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False
                                ),
                                norm,
                                # nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


@BACKBONES.register_module
class HRNet(nn.Module):
    """
    HRNet backbone.

    """

    def __init__(self,
                 conv_cfg=None,
                 normalize=dict(type='BN', frozen=False),
                 EXTRA=None,
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 with_cp=False,
                 zero_init_residual=True,
                 **kwargs):
        super(HRNet, self).__init__()
        extra = EXTRA
        assert extra is not None, "EXTRA is None."
        self.dcn = dcn
        self.norm_eval = norm_eval
        self.stage_with_dcn = stage_with_dcn
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.conv_cfg = conv_cfg
        self.normalize = normalize

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.norm1_name, norm1 = build_norm_layer(self.normalize, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.norm2_name, norm2 = build_norm_layer(self.normalize, 64, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # Stage 2, has two resolutions
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i]*block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        # Stage 3, has three resolutions
        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i]*block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        # Stage 4, has four resolutions
        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i]*block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels)


    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            _, norm = build_norm_layer(self.normalize, planes*block.expansion)
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes*block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                norm,
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)


    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_pre = len(num_channels_pre_layer)
        num_branches_cur = len(num_channels_cur_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    norm_name, norm = build_norm_layer(self.normalize, num_channels_cur_layer[i])
                    transition_layers.append(nn.Sequential(
                        build_conv_layer(
                            self.conv_cfg,
                            num_channels_pre_layer[i],
                            num_channels_cur_layer[i],
                            3,
                            1,
                            1,
                            bias=False
                        ),
                        # norm,
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i-num_branches_pre else inchannels
                    _, norm = build_norm_layer(self.normalize, outchannels)
                    conv3x3s.append(nn.Sequential(
                        build_conv_layer(
                            self.conv_cfg,
                            inchannels,
                            outchannels,
                            3,
                            2,
                            1,
                            bias=False
                        ),
                        # norm,
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)


    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used in last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     self.normalize,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels


    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            self.load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d) and hasattr(m, 'conv2_offset'):
                    constant_init(m.conv2_offset, 0)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")


    def load_checkpoint(self, model, filename, map_location=None, strict=False, logger=None):
        checkpoint = torch.load(filename, map_location=map_location)

        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(filename))

        # # add prefix of state_dict
        # if not list(state_dict.keys())[0].startswith('backbone.'):
        #     state_dict = {('backbone.'+k): v for k, v in state_dict.items()}

        # skip prefix of state_dict
        if list(state_dict.keys())[0].startswith('backbone.'):
            state_dict = {k[9:]: v for k, v in state_dict.items()}

        # load state_dict
        if hasattr(model, 'module'):
            self.load_state_dict(model.module, state_dict, strict, logger)
        else:
            self.load_state_dict(model, state_dict, strict, logger)
        return checkpoint


    def load_state_dict(self, module, state_dict, strict=False, logger=None):
        """Load state_dict to a module.

        This method is modified from :meth:`torch.nn.Module.load_state_dict`.
        Default value for ``strict`` is set to ``False`` and the message for
        param mismatch will be shown even if strict is False.

        Args:
            module (Module): Module that receives the state_dict.
            state_dict (OrderedDict): Weights.
            strict (bool): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
            logger (:obj:`logging.Logger`, optional): Logger to log the error
                message. If not specified, print function will be used.
        """
        unexpected_keys = []
        own_state = module.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(),
                                           param.size()))
        missing_keys = set(own_state.keys()) - set(state_dict.keys())

        err_msg = []
        if unexpected_keys:
            # err_msg.append('unexpected key in source state_dict: {}\n'.format(
            #     ', '.join(unexpected_keys)))
            print('unexpected example: ', list(unexpected_keys)[0])
            err_msg.append('unexpected key in source state_dict: {}\n'.format(len(unexpected_keys)))
        if missing_keys:
            # err_msg.append('missing keys in source state_dict: {}\n'.format(
            #     ', '.join(missing_keys)))
            print('missing example: ', list(missing_keys)[0])
            err_msg.append('missing keys in source state_dict: {}\n'.format(len(missing_keys)))
        err_msg = '\n'.join(err_msg)
        if err_msg:
            if strict:
                raise RuntimeError(err_msg)
            elif logger is not None:
                logger.warn(err_msg)
            else:
                print(err_msg)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # Stage 2
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # Stage 3
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        # Stage 4
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # print('y_list shape: ', y_list.shape)
        return tuple(y_list)

    def train(self, mode=True):
        super(HRNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()




