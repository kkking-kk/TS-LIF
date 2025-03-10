from torch import nn

from spikingjelly.activation_based import surrogate, neuron

from SeqSNN.network.snn.TSLIF import *
from SeqSNN.network.snn.surrogate import atan as SG
import snntorch as snn
from snntorch import utils
tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True


class SSA(nn.Module):
    def __init__(
        self, length, tau, common_thr, dim, heads=8, qkv_bias=False, qk_scale=0.25
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.heads = heads
        self.qk_scale = qk_scale

        self.q_m = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)


        # self.q_lif = snn.Leaky(
        #     beta=0.99,
        #     spike_grad=SG.apply,
        #     init_hidden=True,
        #     output=False,
        # )
        self.q_tslif = TSLIFNode(
            surrogate_function=SG.apply,
        )

        self.k_m = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)

        # self.k_lif = snn.Leaky(
        #     beta=0.99,
        #     spike_grad=SG.apply,
        #     init_hidden=True,
        #     output=False,
        # )
        self.k_tslif = TSLIFNode(
            surrogate_function =SG.apply,
        )

        self.v_m = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)

        self.v_tslif = TSLIFNode(
            surrogate_function =SG.apply,
        )
        # self.v_lif = snn.Leaky(
        #     beta=0.99,
        #     spike_grad=SG.apply,
        #     init_hidden=True,
        #     output=False,
        # )

        # self.attn_lif = snn.Leaky(
        #     beta=0.99,
        #     spike_grad=SG.apply,
        #     init_hidden=True,
        #     output=False,
        # )

        self.attn_tslif = TSLIFNode(
            v_threshold=0.7,
            surrogate_function=SG.apply
        )

        self.last_m = nn.Linear(dim, dim)
        self.last_bn = nn.BatchNorm1d(dim)

        # self.last_lif = snn.Leaky(
        #     beta=0.99,
        #     spike_grad=SG.apply,
        #     init_hidden=True,
        #     output=False,
        # )
        self.last_tslif = TSLIFNode(
            surrogate_function=SG.apply
        )

    def forward(self, x):
        utils.reset(self.q_lif)
        utils.reset(self.k_lif)
        utils.reset(self.v_lif)
        utils.reset(self.attn_lif)
        utils.reset(self.last_lif)
        # x = x.transpose(0, 1)

        # T, B, L, D = x.shape
        B, T, L, D = x.shape
        # x_for_qkv = x.flatten(0, 1)  # TB L D
        x_for_qkv = x.flatten(0, 1)  # BT L D
        # q_m_out = self.q_m(x_for_qkv)  # TB L D
        q_m_out = self.q_m(x_for_qkv) # BT L D
        # q_m_out = (
        #     self.q_bn(q_m_out.transpose(-1, -2))
        #     .transpose(-1, -2)
        #     .reshape(T, B, L, D)
        #     .contiguous()
        # )
        q_m_out = (
            self.q_bn(q_m_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(B, T, L, D)
            .contiguous()
        )
        # q_m_out = self.q_lif(q_m_out)
        q_m_out = self.q_tslif(q_m_out)
        # q = (
        #     q_m_out.reshape(T, B, L, self.heads, D // self.heads)
        #     .permute(0, 1, 3, 2, 4)
        #     .contiguous()
        # )

        q = (
            q_m_out.reshape(B, T, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_m_out = self.k_m(x_for_qkv)

        k_m_out = (
            self.k_bn(k_m_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(B, T, L, D)
            .contiguous()
        )

        # k_m_out = self.k_lif(k_m_out)
        k_m_out = self.k_tslif(k_m_out)
        # k = (
        #     k_m_out.reshape(T, B, L, self.heads, D // self.heads)
        #     .permute(0, 1, 3, 2, 4)
        #     .contiguous()
        # )

        k = (
            k_m_out.reshape(B, T, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_m_out = self.v_m(x_for_qkv)
        v_m_out = (
            self.v_bn(v_m_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(B, T, L, D)
            .contiguous()
        )

        # v_m_out = self.v_lif(v_m_out)
        v_m_out = self.v_tslif(v_m_out)
        # v = (
        #     v_m_out.reshape(T, B, L, self.heads, D // self.heads)
        #     .permute(0, 1, 3, 2, 4)
        #     .contiguous()
        # )

        v = (
            v_m_out.reshape(B, T, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        attn = (q @ k.transpose(-2, -1)) * self.qk_scale
        x = attn @ v  # x_shape: T * B * heads * L * D//heads

        # x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = x.transpose(2, 3).reshape(B, T, L, D).contiguous()
        # x = self.attn_lif(x)
        x = self.attn_tslif(x)
        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2)
        # x = self.last_lif(x.reshape(B, T, L, D).contiguous())
        x = self.last_tslif(x.reshape(B, T, L, D).contiguous())
        return x


class MLP(nn.Module):
    def __init__(
        self,
        length,
        tau,
        common_thr,
        in_features,
        hidden_features=None,
        out_features=None,
    ):
        super().__init__()
        # self.length = length
        out_features = out_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)


        # self.lif1 = snn.Leaky(
        #     beta=0.99,
        #     spike_grad=SG.apply,
        #     init_hidden=True,
        #     output=False,
        # )

        self.mlp_tclif1 = TCLIFNode2(
            surrogate_function =SG.apply,
        )

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)



        self.mlp_tclif2 =  TCLIFNode(
            surrogate_function =SG.apply,
        )

        # self.lif2 = snn.Leaky(
        #     beta=0.99,
        #     spike_grad=SG.apply,
        #     init_hidden=True,
        #     output=False,
        # )

    def forward(self, x):
        utils.reset(self.lif1)
        utils.reset(self.lif2)
        # T, B, L, D = x.shape
        B, T, L, D = x.shape
        x = x.flatten(0, 1) # BT L D
        # x = x.flatten(0, 1)  # TB L D
        x = self.fc1(x)  # TB L H

        # x = (
        #     self.bn1(x.transpose(-1, -2))
        #     .transpose(-1, -2)
        #     .reshape(T, B, L, self.hidden_features)
        #     .contiguous()
        # )
        x = (
            self.bn1(x.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(B, T, L, self.hidden_features)
            .contiguous()
        )
        x = self.mlp_tclif1(x)
        # x = self.lif1(x)
        x = x.flatten(0, 1)  # TB L H
        x = self.fc2(x)  # TB L D
        # x = (
        #     self.bn2(x.transpose(-1, -2))
        #     .transpose(-1, -2)
        #     .reshape(T, B, L, D)
        #     .contiguous()
        # )
        x = (
            self.bn2(x.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(B, T, L, D)
            .contiguous()
        )
        x = self.mlp_tclif2(x)
        # x = self.lif2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        length,
        tau,
        common_thr,
        dim,
        d_ff,
        heads=8,
        qkv_bias=False,
        qk_scale=0.125,
    ):
        super().__init__()
        self.attn = SSA(
            length=length,
            tau=tau,
            common_thr=common_thr,
            dim=dim,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.mlp = MLP(
            length=length,
            tau=tau,
            common_thr=common_thr,
            in_features=dim,
            hidden_features=d_ff,
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
