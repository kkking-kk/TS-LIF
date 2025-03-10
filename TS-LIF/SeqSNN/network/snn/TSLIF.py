# from abc import abstractmethod
# from typing import Callable
# import torch
# from SeqSNN.network.snn import TSLIF_base
# from torch import nn
# from snntorch import surrogate
#
# class BaseNode(TCLIF_base.MemoryModule):
#     def __init__(self,
#                  v_threshold: float = 1.,
#                  v_reset: float = 0.,
#                  surrogate_function: Callable = None,
#                  detach_reset: bool = False,
#                  step_mode='s', backend='torch',
#                  store_v_seq: bool = True):
#
#         assert isinstance(v_reset, float) or v_reset is None
#         assert isinstance(v_threshold, float)
#         assert isinstance(detach_reset, bool)
#         super().__init__()
#
#         if v_reset is None:
#             self.register_memory('v', 0.)
#             self.register_memory('v_s', 0.)
#         else:
#             self.register_memory('v', v_reset)
#
#         self.v_threshold = v_threshold
#
#         self.v_reset = v_reset
#         self.detach_reset = detach_reset
#         self.surrogate_function = surrogate_function
#
#         self.step_mode = step_mode
#         self.backend = backend
#
#         self.store_v_seq = store_v_seq
#
#
#         self.alpha_s = torch.nn.Parameter(torch.randn([1, 64], dtype=torch.float))
#         self.alpha_l = torch.nn.Parameter(torch.randn([1, 64], dtype=torch.float))
#
#         # self.v_threshold_s = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float))
#         # self.v_threshold_l = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float))
#     @property
#     def store_v_seq(self):
#         return self._store_v_seq
#
#     @store_v_seq.setter
#     def store_v_seq(self, value: bool):
#         self._store_v_seq = value
#         if value:
#             if not hasattr(self, 'v_seq'):
#                 self.register_memory('v_seq', None)
#
#     @staticmethod
#     @torch.jit.script
#     def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
#         v = (1. - spike) * v + spike * v_reset
#
#         return v
#
#     @staticmethod
#     @torch.jit.script
#     def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
#         v = v - spike * v_threshold
#         return v
#
#
#     @abstractmethod
#     def neuronal_charge(self, x: torch.Tensor):
#         raise NotImplementedError
#
#     def neuronal_fire(self):
#         # print('22', self.surrogate_function)
#         return self.surrogate_function(self.v - self.v_threshold, 2.0)
#
#     def sl_neuronal_fire(self):
#         s_s = self.surrogate_function(self.v - self.v_threshold, 2.0)
#         s_l = self.surrogate_function(self.v_s - self.v_threshold,  2.0)
#         return s_s, s_l
#
#     def extra_repr(self):
#         return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'
#
#     def single_step_forward(self, x: torch.Tensor):
#         self.v_float_to_tensor(x)
#         self.neuronal_charge(x)
#         spike = self.neuronal_fire()
#         # s_s, s_l = self.sl_neuronal_fire()
#         # test = s_s.sum() - s_l.sum()
#         # print('diff', test)
#         # spike = self.alpha_s * s_s + self.alpha_l * s_l
#         # spike = self.kappa_s * s_s +self.kappa_l * s_l
#         self.neuronal_reset(spike)
#         # self.neuronal_reset(s_s, s_l)
#         return spike
#
#     def multi_step_forward(self, x_seq: torch.Tensor):
#
#         #### time series ###
#         T = x_seq.shape[-1]
#         # print(x_seq.shape)
#         # print(T)
#         # dsa
#         y_seq = []
#         if self.store_v_seq:
#             v_seq = []
#         for t in range(2):
#             y = self.single_step_forward(x_seq[:, t, :, :])
#             # print(y.shape)
#             # dsa
#             y_seq.append(y)
#             if self.store_v_seq:
#                 v_seq.append(self.v)
#         if self.store_v_seq:
#             self.v_seq = torch.stack(v_seq)
#             # print('v_seq', self.v_seq.shape)
#
#         # T = x_seq.shape[0]
#         # y_seq = []
#         # if self.store_v_seq:
#         #     v_seq = []
#         # for t in range(T):
#         #     y = self.single_step_forward(x_seq[t])
#         #     y_seq.append(y)
#         #     if self.store_v_seq:
#         #         v_seq.append(self.v)
#         #
#         # if self.store_v_seq:
#         #     self.v_seq = torch.stack(v_seq)
#         outputs = torch.stack(y_seq, dim=0)
#         outputs = outputs.permute(1, 0, 2, 3)
#         # print(outputs.shape)
#         # dsa
#         return outputs
#         # return torch.stack(y_seq, dim=0)
#
#     def v_float_to_tensor(self, x: torch.Tensor):
#         if isinstance(self.v, float):
#             v_init = self.v
#             self.v = torch.full_like(x.data, v_init)
#
#
# class TCLIFNode(BaseNode):
#     def __init__(self,
#                  v_threshold=1.0,
#                  v_reset=0.,
#                  surrogate_function: Callable = None,
#                  detach_reset=False,
#                  hard_reset=False,
#                  step_mode='s',
#                  k=2,
#                  decay_factor: torch.Tensor = torch.tensor([0.8, 0.2, 0.3, 0.7], dtype=torch.float),
#                  gamma: float = 0.5):
#         super(TCLIFNode, self).__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
#         self.k = k
#         for i in range(1, self.k + 1):
#             self.register_memory('v' + str(i), 0.)
#
#         self.names = self._memories
#         self.hard_reset = hard_reset
#         self.gamma = gamma
#         self.decay_factor = torch.nn.Parameter(decay_factor)
#         self.kk = torch.nn.Parameter(torch.tensor([0.8], dtype=torch.float))
#         self.yy = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float))
#
#     @property
#     def supported_backends(self):
#         if self.step_mode == 's':
#             return ('torch',)
#         elif self.step_mode == 'm':
#             return ('torch', 'cupy')
#         else:
#             raise ValueError(self.step_mode)
#
#     def neuronal_charge(self, x: torch.Tensor):
#         # self.names['v1'] = self.names['v1'] - torch.sigmoid(self.decay_factor[0][0]) * self.names['v2'] + x
#         # self.names['v2'] = self.names['v2'] + torch.sigmoid(self.decay_factor[0][1]) * self.names['v1']
#         # print(self.decay_factor[0])
#         # print(self.decay_factor[1])
#
#         self.names['v1'] = self.decay_factor[0] * self.names['v1'] + self.decay_factor[1] * x - self.yy * self.names['v2']
#         self.names['v2'] = self.decay_factor[2] * self.names['v2'] + self.decay_factor[3] * x - self.kk * self.names['v1']
#
#         # self.names['v1'] =  self.names['v1'] + (1 - torch.sigmoid(self.decay_factor[0])) * x
#         # self.names['v2'] =  self.names['v2'] + (1 - torch.sigmoid(self.decay_factor[1])) * x - self.names['v1']
#
#         self.v = self.names['v2']
#         self.v_s = self.names['v1']
#
#     def neuronal_reset(self, spike_s, spike_l):
#         # if self.detach_reset:
#         #     spike_d = spike.detach()
#         # else:
#         #     spike_d = spike
#
#
#         if not self.hard_reset:
#             # soft reset
#             # print(type(self.v_threshold))
#             # das
#             # self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.gamma)
#             self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_l , self.gamma)
#             self.names['v2'] = self.jit_soft_reset(self.names['v2'], spike_s, self.v_threshold)
#         else:
#             # hard reset
#             for i in range(2, self.k + 1):
#                 self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d,  self.v_reset)
#
#     def forward(self, x: torch.Tensor):
#         # self.v = 0.
#         # self.v1 = 0.
#         # self.v2 = 0.
#         # print('kappa_s', self.kappa_s)
#         # print('kappa_l', self.kappa_l)
#         # test= super().single_step_forward(x)
#         # s = test.sum()
#         # print('spike', s)
#         # if x.shape[-1] != 256:
#         #     self.adaptive_sl = x.shape[-1]
#         # return super().multi_step_forward(x)
#         return super().single_step_forward(x)
#     def extra_repr(self):
#         return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
#                f"hard_reset={self.hard_reset}, " \
#                f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"
#
#
#
#
# class LSLIF(nn.Module):
#     def __init__(
#             self,
#             v_threshold=1.,
#             v_reset=0.,
#             surrogate_function=surrogate.atan(alpha=2.0),
#             hard_reset=False,
#             step_mode='s',
#             k=2,
#             beta=torch.tensor([0.99, 0.1]),
#             gamma=1.,
#             num_step=1,
#             output_mems=False
#     ):
#         super().__init__(),
#         self.surrogate_function = surrogate_function  # surrogate.atan(alpha=2.0)
#         self.v_threshold = v_threshold
#         self.v_reset = v_reset
#         self.hard_reset = hard_reset
#         self.step_mode = step_mode
#         self.k = k
#         self.beta = nn.Parameter(torch.Tensor(2))
#         # nn.init.uniform_(self.beta, 0.9, 0.1)
#         self.gamma = nn.Parameter(torch.Tensor(1))
#         self.num_step = num_step
#         self.weight = nn.Parameter(torch.Tensor(2))
#         nn.init.uniform_(self.weight, 1, 1)  # Initialize weight values
#         self.output_mems = output_mems
#
#     def neuronal_fire(self):
#         self.Sd = self.surrogate_function(self.Ud - self.v_threshold, 2.0)
#         self.Ss = self.surrogate_function(self.Us - self.v_threshold, 2.0)
#
#     def neuron_charge(self, x: torch.Tensor):
#         self.Ud = torch.sigmoid(self.beta[0]) * self.Ud + (1 - torch.sigmoid(self.beta[0])) * x
#         # self.Us = torch.sigmoid(self.beta[1]) * self.Us + (1 - torch.sigmoid(self.beta[1])) * x - self.gamma * self.Ud
#         self.Us = torch.sigmoid(self.beta[1]) * self.Us + (1 - torch.sigmoid(self.beta[1])) * x - self.Ud
#
#     def neuron_reset(self):
#         if not self.hard_reset:
#             # self.Ud = self.Ud - self.v_threshold * self.Sd
#             self.Ud = self.Ud - 0.5 * self.Sd
#             self.Us = self.Us - self.v_threshold * self.Ss
#
#     def forward(self, x: torch.Tensor):
#         y_seq = []
#         if self.step_mode == 's':
#             self.Ud = torch.zeros_like(x)
#             self.Us = torch.zeros_like(x)
#             for ii in range(self.num_step):
#
#                 self.neuron_charge(x)
#                 self.neuronal_fire()
#                 self.neuron_reset()
#                 # y = self.weight[0] * self.Sd + self.weight[1] * self.Ss
#                 y = self.Sd + self.Ss
#                 y_seq.append(y)
#         else:
#             self.Ud = torch.zeros_like(x[..., 0])
#             self.Us = torch.zeros_like(x[..., 0])
#             for ii in range(x.shape[-1]):
#                 self.neuron_charge(x[..., ii])
#                 self.neuronal_fire()
#                 self.neuron_reset()
#                 y = self.weight[0] * self.Sd + self.weight[1] * self.Ss
#                 y_seq.append(y)
#         bs = y_seq[0].shape[0]
#         y_seq = torch.stack(y_seq, dim=-1).squeeze().reshape(bs, -1)
#         if self.output_mems:
#             return y_seq, self.Us
#         else:
#             return y_seq
#
#
#
#
#
#
#
#
#
#
#
# class BaseNode1(TCLIF_base.MemoryModule):
#     def __init__(self,
#                  v_threshold: float = 1.,
#                  v_reset: float = 0.,
#                  surrogate_function: Callable = None,
#                  detach_reset: bool = False,
#                  step_mode='s', backend='torch',
#                  store_v_seq: bool = True):
#
#         assert isinstance(v_reset, float) or v_reset is None
#         assert isinstance(v_threshold, float)
#         assert isinstance(detach_reset, bool)
#         super().__init__()
#
#         if v_reset is None:
#             self.register_memory('v', 0.)
#             self.register_memory('v_s', 0.)
#         else:
#             self.register_memory('v', v_reset)
#
#         self.v_threshold = v_threshold
#
#         self.v_reset = v_reset
#         self.detach_reset = detach_reset
#         self.surrogate_function = surrogate_function
#
#         self.step_mode = step_mode
#         self.backend = backend
#
#         self.store_v_seq = store_v_seq
#
#         # self.ada_last = adaptive_last
#         self.alpha_s = torch.nn.Parameter(torch.randn([1, 2048], dtype=torch.float))
#         self.alpha_l = torch.nn.Parameter(torch.randn([1, 2048], dtype=torch.float))
#
#
#         # self.v_threshold_s = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float))
#         # self.v_threshold_l = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float))
#     @property
#     def store_v_seq(self):
#         return self._store_v_seq
#
#     @store_v_seq.setter
#     def store_v_seq(self, value: bool):
#         self._store_v_seq = value
#         if value:
#             if not hasattr(self, 'v_seq'):
#                 self.register_memory('v_seq', None)
#
#     @staticmethod
#     @torch.jit.script
#     def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
#         v = (1. - spike) * v + spike * v_reset
#
#         return v
#
#     @staticmethod
#     @torch.jit.script
#     def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
#         v = v - spike * v_threshold
#         return v
#
#
#     @abstractmethod
#     def neuronal_charge(self, x: torch.Tensor):
#         raise NotImplementedError
#
#     def neuronal_fire(self):
#         # print('22', self.surrogate_function)
#         return self.surrogate_function(self.v - self.v_threshold, 2.0)
#
#     def sl_neuronal_fire(self):
#         s_s = self.surrogate_function(self.v - self.v_threshold, 2.0)
#         s_l = self.surrogate_function(self.v_s - self.v_threshold,  2.0)
#         return s_s, s_l
#
#     def extra_repr(self):
#         return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'
#
#     def single_step_forward(self, x: torch.Tensor):
#         self.v_float_to_tensor(x)
#         self.neuronal_charge(x)
#         # spike = self.neuronal_fire()
#         s_s, s_l = self.sl_neuronal_fire()
#         # test = s_s.sum() - s_l.sum()
#         # print('diff', test)
#         spike = self.alpha_s * s_s + self.alpha_l * s_l
#         # spike = self.kappa_s * s_s +self.kappa_l * s_l
#        # self.neuronal_reset(spike)
#         self.neuronal_reset(s_s, s_l)
#         return spike
#
#     def multi_step_forward(self, x_seq: torch.Tensor):
#
#         #### time series ###
#         T = x_seq.shape[-1]
#         # print(x_seq.shape)
#         # print(T)
#         # dsa
#         y_seq = []
#         if self.store_v_seq:
#             v_seq = []
#         for t in range(2):
#             y = self.single_step_forward(x_seq[:, t, :, :])
#             y_seq.append(y)
#             if self.store_v_seq:
#                 v_seq.append(self.v)
#         if self.store_v_seq:
#             self.v_seq = torch.stack(v_seq)
#             # print('v_seq', self.v_seq.shape)
#         outputs = torch.stack(y_seq, dim=0)
#         outputs = outputs.permute(1, 0, 2, 3)
#         # T = x_seq.shape[0]
#         # y_seq = []
#         # if self.store_v_seq:
#         #     v_seq = []
#         # for t in range(T):
#         #     y = self.single_step_forward(x_seq[t])
#         #     y_seq.append(y)
#         #     if self.store_v_seq:
#         #         v_seq.append(self.v)
#         #
#         # if self.store_v_seq:
#         #     self.v_seq = torch.stack(v_seq)
#         # outputs = torch.stack(y_seq, dim=0)
#         # print(outputs.shape)
#         # dsa
#         return outputs
#         # return torch.stack(y_seq, dim=0)
#
#     def v_float_to_tensor(self, x: torch.Tensor):
#         if isinstance(self.v, float):
#             v_init = self.v
#             self.v = torch.full_like(x.data, v_init)
#
#
# class TCLIFNode2(BaseNode1):
#     def __init__(self,
#                  v_threshold=0.8,
#                  v_reset=0.,
#                  surrogate_function: Callable = None,
#                  detach_reset=False,
#                  hard_reset=False,
#                  step_mode='s',
#                  k=2,
#                  decay_factor: torch.Tensor = torch.tensor([0.8, 0.2, 0.3, 0.7], dtype=torch.float),
#                  gamma: float = 0.5):
#         super(TCLIFNode2, self).__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
#         self.k = k
#         for i in range(1, self.k + 1):
#             self.register_memory('v' + str(i), 0.)
#
#         self.names = self._memories
#         self.hard_reset = hard_reset
#         self.gamma = gamma
#         self.decay_factor = torch.nn.Parameter(decay_factor)
#         self.kk = torch.nn.Parameter(torch.tensor([0.8], dtype=torch.float))
#         self.yy = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float))
#
#     @property
#     def supported_backends(self):
#         if self.step_mode == 's':
#             return ('torch',)
#         elif self.step_mode == 'm':
#             return ('torch', 'cupy')
#         else:
#             raise ValueError(self.step_mode)
#
#     def neuronal_charge(self, x: torch.Tensor):
#         # self.names['v1'] = self.names['v1'] - torch.sigmoid(self.decay_factor[0][0]) * self.names['v2'] + x
#         # self.names['v2'] = self.names['v2'] + torch.sigmoid(self.decay_factor[0][1]) * self.names['v1']
#         # print(self.decay_factor[0])
#         # print(self.decay_factor[1])
#
#         self.names['v1'] = self.decay_factor[0] * self.names['v1'] + self.decay_factor[1] * x - self.yy * self.names['v2']
#         self.names['v2'] = self.decay_factor[2] * self.names['v2'] + self.decay_factor[3] * x - self.kk * self.names['v1']
#
#         # self.names['v1'] =  self.names['v1'] + (1 - torch.sigmoid(self.decay_factor[0])) * x
#         # self.names['v2'] =  self.names['v2'] + (1 - torch.sigmoid(self.decay_factor[1])) * x - self.names['v1']
#
#         self.v = self.names['v2']
#         self.v_s = self.names['v1']
#
#     def neuronal_reset(self, spike_s, spike_l):
#         # if self.detach_reset:
#         #     spike_d = spike.detach()
#         # else:
#         #     spike_d = spike
#
#
#         if not self.hard_reset:
#             # soft reset
#             # print(type(self.v_threshold))
#             # das
#             # self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.gamma)
#             self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_l , self.gamma)
#             self.names['v2'] = self.jit_soft_reset(self.names['v2'], spike_s, self.v_threshold)
#         else:
#             # hard reset
#             for i in range(2, self.k + 1):
#                 self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d,  self.v_reset)
#
#     def forward(self, x: torch.Tensor):
#         # self.v = 0.
#         # self.v1 = 0.
#         # self.v2 = 0.
#         # print('kappa_s', self.kappa_s)
#         # print('kappa_l', self.kappa_l)
#         # test= super().single_step_forward(x)
#         # s = test.sum()
#         # print('spike', s)
#         # if x.shape[-1] != 256:
#         #     self.adaptive_sl = x.shape[-1]
#         # return super().multi_step_forward(x)
#         return super().single_step_forward(x)
#     def extra_repr(self):
#         return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
#                f"hard_reset={self.hard_reset}, " \
#                f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"
#
#
#



from abc import abstractmethod
from typing import Callable
import torch
from SeqSNN.network.snn import TSLIF_base
from torch import nn
from snntorch import surrogate
#
# class BaseNode(TCLIF_base.MemoryModule):
#     def __init__(self,
#                  v_threshold: float = 1.,
#                  v_reset: float = 0.,
#                  surrogate_function: Callable = None,
#                  detach_reset: bool = False,
#                  step_mode='s', backend='torch',
#                  store_v_seq: bool = False):
#
#         assert isinstance(v_reset, float) or v_reset is None
#         assert isinstance(v_threshold, float)
#         assert isinstance(detach_reset, bool)
#         super().__init__()
#
#         if v_reset is None:
#             self.register_memory('v', 0.)
#         else:
#             self.register_memory('v', v_reset)
#
#         self.v_threshold = v_threshold
#
#         self.v_reset = v_reset
#         self.detach_reset = detach_reset
#         self.surrogate_function = surrogate_function
#
#         self.step_mode = step_mode
#         self.backend = backend
#
#         self.store_v_seq = store_v_seq
#
#
#     @property
#     def store_v_seq(self):
#         return self._store_v_seq
#
#     @store_v_seq.setter
#     def store_v_seq(self, value: bool):
#         self._store_v_seq = value
#         if value:
#             if not hasattr(self, 'v_seq'):
#                 self.register_memory('v_seq', None)
#
#     @staticmethod
#     @torch.jit.script
#     def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
#         v = (1. - spike) * v + spike * v_reset
#
#         return v
#
#     @staticmethod
#     @torch.jit.script
#     def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
#         v = v - spike * v_threshold
#         return v
#
#     @abstractmethod
#     def neuronal_charge(self, x: torch.Tensor):
#         raise NotImplementedError
#
#     def neuronal_fire(self):
#         return self.surrogate_function(self.v - self.v_threshold)
#
#     def extra_repr(self):
#         return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'
#
#     def single_step_forward(self, x: torch.Tensor):
#         self.v_float_to_tensor(x)
#         self.neuronal_charge(x)
#         spike = self.neuronal_fire()
#         self.neuronal_reset(spike)
#         return spike
#
#     def multi_step_forward(self, x_seq: torch.Tensor):
#         T = x_seq.shape[0]
#         y_seq = []
#         if self.store_v_seq:
#             v_seq = []
#         for t in range(T):
#             y = self.single_step_forward(x_seq[t])
#             y_seq.append(y)
#             if self.store_v_seq:
#                 v_seq.append(self.v)
#
#         if self.store_v_seq:
#             self.v_seq = torch.stack(v_seq)
#
#         return torch.stack(y_seq)
#
#     def v_float_to_tensor(self, x: torch.Tensor):
#         if isinstance(self.v, float):
#             v_init = self.v
#             self.v = torch.full_like(x.data, v_init)
#
#
# class TCLIFNode(BaseNode):
#     def __init__(self,
#                  v_threshold=1.,
#                  v_reset=0.,
#                  surrogate_function: Callable = None,
#                  detach_reset=False,
#                  hard_reset=False,
#                  step_mode='s',
#                  k=2,
#                  decay_factor: torch.Tensor = torch.full([1, 2], 0, dtype=torch.float),
#                  gamma: float = 0.5):
#         super(TCLIFNode, self).__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
#         self.k = k
#         for i in range(1, self.k + 1):
#             self.register_memory('v' + str(i), 0.)
#
#         self.names = self._memories
#         self.hard_reset = hard_reset
#         self.gamma = gamma
#         self.decay = decay_factor
#         self.decay_factor = torch.nn.Parameter(decay_factor)
#
#     @property
#     def supported_backends(self):
#         if self.step_mode == 's':
#             return ('torch',)
#         elif self.step_mode == 'm':
#             return ('torch', 'cupy')
#         else:
#             raise ValueError(self.step_mode)
#
#     def neuronal_charge(self, x: torch.Tensor):
#         # v1: membrane potential of dendritic compartment
#         # v2: membrane potential of somatic compartment
#         self.names['v1'] = self.names['v1'] - torch.sigmoid(self.decay_factor[0][0]) * self.names['v2'] + x
#         self.names['v2'] = self.names['v2'] + torch.sigmoid(self.decay_factor[0][1]) * self.names['v1']
#         self.v = self.names['v2']
#
#     def neuronal_reset(self, spike):
#         if self.detach_reset:
#             spike_d = spike.detach()
#         else:
#             spike_d = spike
#
#         if not self.hard_reset:
#             # soft reset
#             self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.gamma)
#             self.names['v2'] = self.jit_soft_reset(self.names['v2'], spike_d, self.v_threshold)
#         else:
#             # hard reset
#             for i in range(2, self.k + 1):
#                 self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d,  self.v_reset)
#
#     def forward(self, x: torch.Tensor):
#         return super().single_step_forward(x)
#
#     def extra_repr(self):
#         return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
#                f"hard_reset={self.hard_reset}, " \
#                f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"
#




class BaseNode(TSLIF_base.MemoryModule):
    def __init__(self,
                 v_threshold: float = 1.,
                 v_reset: float = 0.,
                 surrogate_function: Callable = None,
                 detach_reset: bool = False,
                 step_mode='s', backend='torch',
                 store_v_seq: bool = True):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('v_s', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold

        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq


        self.alpha_s = torch.nn.Parameter(torch.randn([1, 128], dtype=torch.float))
        self.alpha_l = torch.nn.Parameter(torch.randn([1, 128], dtype=torch.float))

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset

        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v


    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold, 2.0)

    def sl_neuronal_fire(self):
        s_s = self.surrogate_function(self.v - self.v_threshold, 2.0)
        s_l = self.surrogate_function(self.v_s - self.v_threshold,  2.0)
        return s_s, s_l

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        # spike = self.neuronal_fire()
        s_s, s_l = self.sl_neuronal_fire()
        spike = self.alpha_s * s_s + self.alpha_l * s_l
        # self.neuronal_reset(spike)
        self.neuronal_reset(s_s, s_l)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):

        #### time series ###
        T = x_seq.shape[-1]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[:, t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        # if self.store_v_seq:
        #     self.v_seq = torch.stack(v_seq)
        outputs = torch.stack(y_seq, dim=0).permute(1, 0)

        return outputs

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class TSLIFNode(BaseNode):
    def __init__(self,
                 v_threshold=1.0,
                 v_reset=0.,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 hard_reset=False,
                 step_mode='s',
                 k=2,
                 decay_factor: torch.Tensor = torch.tensor([0.8, 0.2, 0.3, 0.7], dtype=torch.float),
                 gamma: float = 0.5):
        super(TSLIFNode, self).__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.k = k
        for i in range(1, self.k + 1):
            self.register_memory('v' + str(i), 0.)

        self.names = self._memories
        self.hard_reset = hard_reset
        self.gamma = gamma
        self.decay_factor = torch.nn.Parameter(decay_factor)
        self.kk = torch.nn.Parameter(torch.tensor([0.8], dtype=torch.float))
        self.yy = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float))

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        # self.names['v1'] = self.names['v1'] - torch.sigmoid(self.decay_factor[0][0]) * self.names['v2'] + x
        # self.names['v2'] = self.names['v2'] + torch.sigmoid(self.decay_factor[0][1]) * self.names['v1']


        self.names['v1'] = self.decay_factor[0] * self.names['v1'] + self.decay_factor[1] * x - self.yy * self.names['v2']
        self.names['v2'] = self.decay_factor[2] * self.names['v2'] + self.decay_factor[3] * x - self.kk * self.names['v1']

        # self.names['v1'] =  self.names['v1'] + (1 - torch.sigmoid(self.decay_factor[0])) * x
        # self.names['v2'] =  self.names['v2'] + (1 - torch.sigmoid(self.decay_factor[1])) * x - self.names['v1']

        self.v = self.names['v2']
        self.v_s = self.names['v1']

    def neuronal_reset(self, spike_s, spike_l):


        if not self.hard_reset:
            # soft reset
            # self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.gamma)
            self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_l , self.gamma)
            self.names['v2'] = self.jit_soft_reset(self.names['v2'], spike_s, self.v_threshold)
        else:
            # hard reset
            for i in range(2, self.k + 1):
                self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d,  self.v_reset)

    def forward(self, x: torch.Tensor):
        # self.v = 0.
        # self.v1 = 0.
        # self.v2 = 0.
        return super().single_step_forward(x)
    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
               f"hard_reset={self.hard_reset}, " \
               f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"








class BaseNode1(TSLIF_base.MemoryModule):
    def __init__(self,
                 v_threshold: float = 1.,
                 v_reset: float = 0.,
                 surrogate_function: Callable = None,
                 detach_reset: bool = False,
                 step_mode='s', backend='torch',
                 store_v_seq: bool = True):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('v_s', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold

        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq
        # dimension should change to fit the dataset.
        self.alpha_s = torch.nn.Parameter(torch.randn([1, 168], dtype=torch.float))
        self.alpha_l = torch.nn.Parameter(torch.randn([1, 168], dtype=torch.float))


        # self.v_threshold_s = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float))
        # self.v_threshold_l = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float))
    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset

        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v


    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        # print('22', self.surrogate_function)
        return self.surrogate_function(self.v - self.v_threshold, 2.0)

    def sl_neuronal_fire(self):
        s_s = self.surrogate_function(self.v - self.v_threshold, 2.0)
        s_l = self.surrogate_function(self.v_s - self.v_threshold,  2.0)
        return s_s, s_l

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        # spike = self.neuronal_fire()
        s_s, s_l = self.sl_neuronal_fire()
        # test = s_s.sum() - s_l.sum()
        spike = self.alpha_s * s_s + self.alpha_l * s_l
       # self.neuronal_reset(spike)
        self.neuronal_reset(s_s, s_l)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):

        #### time series ###
        T = x_seq.shape[-1]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[:, t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
            # print('v_seq', self.v_seq.shape)

        #
        # if self.store_v_seq:
        #     self.v_seq = torch.stack(v_seq)
        outputs = torch.stack(y_seq, dim=0).permute(1, 0)
        # print(outputs.shape)
        # dsa
        return outputs
        # return torch.stack(y_seq, dim=0)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class TSLIFNode2(BaseNode1):
    def __init__(self,
                 v_threshold=1.0,
                 v_reset=0.,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 hard_reset=False,
                 step_mode='s',
                 k=2,
                 decay_factor: torch.Tensor = torch.tensor([0.8, 0.2, 0.3, 0.7], dtype=torch.float),
                 gamma: float = 0.5):
        super(TSLIFNode2, self).__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.k = k
        for i in range(1, self.k + 1):
            self.register_memory('v' + str(i), 0.)

        self.names = self._memories
        self.hard_reset = hard_reset
        self.gamma = gamma
        self.decay_factor = torch.nn.Parameter(decay_factor)
        self.kk = torch.nn.Parameter(torch.tensor([0.8], dtype=torch.float))
        self.yy = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float))

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        # self.names['v1'] = self.names['v1'] - torch.sigmoid(self.decay_factor[0][0]) * self.names['v2'] + x
        # self.names['v2'] = self.names['v2'] + torch.sigmoid(self.decay_factor[0][1]) * self.names['v1']


        self.names['v1'] = self.decay_factor[0] * self.names['v1'] + self.decay_factor[1] * x - self.yy * self.names['v2']
        self.names['v2'] = self.decay_factor[2] * self.names['v2'] + self.decay_factor[3] * x - self.kk * self.names['v1']

        # self.names['v1'] =  self.names['v1'] + (1 - torch.sigmoid(self.decay_factor[0])) * x
        # self.names['v2'] =  self.names['v2'] + (1 - torch.sigmoid(self.decay_factor[1])) * x - self.names['v1']

        self.v = self.names['v2']
        self.v_s = self.names['v1']

    def neuronal_reset(self, spike_s, spike_l):


        if not self.hard_reset:
            # soft reset
            # self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.gamma)
            self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_l , self.gamma)
            self.names['v2'] = self.jit_soft_reset(self.names['v2'], spike_s, self.v_threshold)
        else:
            # hard reset
            for i in range(2, self.k + 1):
                self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d,  self.v_reset)

    def forward(self, x: torch.Tensor):
        # self.v = 0.
        # self.v1 = 0.
        # self.v2 = 0.
        # test= super().single_step_forward(x)
        # s = test.sum()
        return super().single_step_forward(x)
    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
               f"hard_reset={self.hard_reset}, " \
               f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"


