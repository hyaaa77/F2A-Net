import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ops import ConvBNReLU
from models.model import DSL
from models.lib.mix_transformer import mit_b3


class SFF(nn.Module):
    """Scale Feature Fusion."""

    def __init__(
        self,
        cur_in_channels=64,
        low_in_channels=32,
        out_channels=16,
        cur_scale=2,
        low_scale=1,
    ):
        super().__init__()

        self.cur_conv = nn.Sequential(
            nn.Conv2d(
                cur_in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.low_conv = nn.Sequential(
            nn.Conv2d(
                low_in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.cur_scale = cur_scale
        self.low_scale = low_scale

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                2 * out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x_cur, x_low):
        x_cur = self.cur_conv(x_cur)
        x_cur = F.interpolate(
            x_cur,
            scale_factor=self.cur_scale,
            mode="bicubic",
            align_corners=False,
        )

        x_low = self.low_conv(x_low)
        x_low = F.interpolate(
            x_low,
            scale_factor=self.low_scale,
            mode="bicubic",
            align_corners=False,
        )

        x = torch.cat((x_cur, x_low), dim=1)
        return self.out_conv(x)


class DSRConv(nn.Module):
    """Convolution block used by DSR."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class DSRFeatureBlock(nn.Module):
    """Residual feature aggregation block used in DSR."""

    def __init__(self, in_channels, out_channels, need_relu=True):
        super().__init__()

        self.need_relu = need_relu
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = DSRConv(in_channels, out_channels, 3, padding=1)
        self.conv1 = DSRConv(out_channels, out_channels, 3, padding=1)
        self.conv2 = DSRConv(out_channels, out_channels, 3, padding=1)
        self.conv3 = DSRConv(out_channels, out_channels, 3, padding=1)

        self.conv_cat = DSRConv(
            4 * out_channels,
            out_channels,
            3,
            padding=1,
        )
        self.conv_res = DSRConv(
            in_channels,
            out_channels,
            1,
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = x0 + self.conv1(x0)
        x2 = x1 + self.conv2(x1)
        x3 = x2 + self.conv3(x2)

        x_cat = self.conv_cat(
            torch.cat((x0, x1, x2, x3), dim=1)
        )
        out = x_cat + self.conv_res(x)

        if self.need_relu:
            out = self.relu(out)

        return out


class DSRAttention(nn.Module):
    """Feature interaction unit used by DSR."""

    def __init__(self, channels):
        super().__init__()

        self.gate = DSRFeatureBlock(
            2 * channels,
            channels,
            need_relu=True,
        )
        self.refine = DSRFeatureBlock(
            3 * channels,
            channels,
            need_relu=False,
        )
        self.out_conv = DSRConv(
            channels,
            channels,
            3,
            padding=1,
        )

    def forward(self, x, y):
        gate = self.gate(torch.cat((x, y), dim=1))
        z = x + y * gate

        refine = self.refine(
            torch.cat((x, y, z), dim=1)
        )

        return self.out_conv(z + refine)


class DSR(nn.Module):
    """Deep Structure Refinement."""

    def __init__(self, channels):
        super().__init__()

        self.refine1 = DSRAttention(channels)
        self.refine2 = DSRAttention(channels)

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )

        self.feature_conv = DSRConv(
            channels,
            channels,
            1,
        )
        self.pred_conv = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
        )

    def forward(self, x1, x2, x3):
        x2 = self.refine1(
            x2,
            self.upsample(x1),
        )

        x3 = self.refine2(
            x3,
            self.upsample(x2),
        )

        feature = self.feature_conv(x3)
        prediction = self.pred_conv(feature)

        return feature, prediction


def erosion_edge(pred, kernel_size=3):
    """Extract an erosion-based edge map."""
    eroded = -F.max_pool2d(
        -pred,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return pred - eroded


class EdgeGuidedRefineFuse(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        self.conv_fi = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )
        self.attn_fi = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.conv_fip1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )
        self.attn_fip1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.fuse_conv = nn.Conv2d(
            2 * in_channels,
            in_channels,
            kernel_size=1,
        )

    def forward(self, Fi, Fip1, Fb):
        Fip1_up = F.interpolate(
            Fip1,
            size=Fi.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        Fi_refine = Fi + Fi * Fb
        Fi_refine = self.conv_fi(Fi_refine)
        Fi_refine = Fi_refine * self.attn_fi(Fi_refine)

        Fip1_refine = Fip1_up + Fip1_up * Fb
        Fip1_refine = self.conv_fip1(Fip1_refine)
        Fip1_refine = (
            Fip1_refine * self.attn_fip1(Fip1_refine)
        )

        fused = torch.cat(
            [Fi_refine, Fip1_refine],
            dim=1,
        )

        return self.fuse_conv(fused)


class F2ANet(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()

        self.backbone = mit_b3()
        self.backbone.init_weights(
            pretrained=pretrained
        )

        # Scale feature fusion
        self.sff1 = SFF(
            cur_in_channels=512,
            low_in_channels=320,
            out_channels=320,
            cur_scale=2,
            low_scale=1,
        )
        self.sff2 = SFF(
            cur_in_channels=320,
            low_in_channels=128,
            out_channels=128,
            cur_scale=2,
            low_scale=1,
        )
        self.sff3 = SFF(
            cur_in_channels=128,
            low_in_channels=64,
            out_channels=64,
            cur_scale=2,
            low_scale=1,
        )

        # Feature refinement before DSR
        self.dsr_block1 = DSRFeatureBlock(64, 64)
        self.dsr_block2 = DSRFeatureBlock(128, 64)
        self.dsr_block3 = DSRFeatureBlock(320, 64)

        # Deep structure refinement
        self.dsr = DSR(64)

        # Deep semantic learning
        self.dsl4 = DSL(64, 64, 32)
        self.dsl3 = DSL(64, 64, 32)
        self.dsl2 = DSL(64, 64, 32)

        # Edge-guided refinement
        self.edge_refine = nn.ModuleList(
            [
                EdgeGuidedRefineFuse(in_channels=64),
                EdgeGuidedRefineFuse(in_channels=64),
                EdgeGuidedRefineFuse(in_channels=64),
            ]
        )

        self.edge_pred = nn.ModuleList(
            [
                ConvBNReLU(64, 1, 3, 1, 1),
                ConvBNReLU(64, 1, 3, 1, 1),
                ConvBNReLU(64, 1, 3, 1, 1),
            ]
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        # SFF
        s1 = self.sff3(x2, x1)
        s2 = self.sff2(x3, x2)
        s3 = self.sff1(x4, x3)

        # Feature refinement
        f1 = self.dsr_block1(s1)
        f2 = self.dsr_block2(s2)
        f3 = self.dsr_block3(s3)

        # DSR: structure feature + coarse prediction
        structure_feature, coarse_pred = self.dsr(
            f3,
            f2,
            f1,
        )

        edge = erosion_edge(
            coarse_pred,
            kernel_size=3,
        )

        p5_up = F.interpolate(
            coarse_pred,
            scale_factor=8,
            mode="bilinear",
        )

        # DSL stage 4
        F4, P4, P5 = self.dsl4(
            f3,
            structure_feature,
            coarse_pred,
        )
        P4 = P4 + P5

        F4 = F.interpolate(
            F4,
            scale_factor=4,
            mode="bilinear",
        )

        F4_edge = self.edge_refine[0](
            F4,
            structure_feature,
            edge,
        )
        F4_pred = self.edge_pred[0](F4_edge)
        edge4 = erosion_edge(
            F4_pred,
            kernel_size=3,
        )

        p4_up = F.interpolate(
            F4_pred,
            scale_factor=8,
            mode="bilinear",
        )

        # DSL stage 3
        F3, P3, P4 = self.dsl3(
            f2,
            F4,
            P4,
        )
        P3 = P3 + P4

        F3 = F.interpolate(
            F3,
            scale_factor=2,
            mode="bilinear",
        )

        F3_edge = self.edge_refine[1](
            F3,
            structure_feature,
            edge4,
        )
        F3_pred = self.edge_pred[1](F3_edge)
        edge3 = erosion_edge(
            F3_pred,
            kernel_size=3,
        )

        p3_up = F.interpolate(
            F3_pred,
            scale_factor=8,
            mode="bilinear",
        )

        # DSL stage 2
        F2, _, _ = self.dsl2(
            f1,
            F3,
            P3,
        )

        F2_edge = self.edge_refine[2](
            F2,
            structure_feature,
            edge3,
        )

        # Keep the original network behavior:
        # the original code used CBR_edge[1] at this stage.
        F2_pred = self.edge_pred[1](F2_edge)

        p2_up = F.interpolate(
            F2_pred,
            scale_factor=8,
            mode="bilinear",
        )

        return p2_up, p3_up, p4_up, p5_up
