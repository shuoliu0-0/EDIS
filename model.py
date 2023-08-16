# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MGraphDTA_EDIS"""
import collections
import copy
import numpy as np

import mindspore as ms
from mindspore import nn, ops
from mindspore import Tensor, Parameter

from mindspore.nn.layer.normalization import _BatchNorm
from mindspore.nn.probability.distribution import Normal

from mindspore_gl import Graph
from mindspore_gl import GNNCell
from mindspore_gl.nn import AvgPooling


class SparseDispatcher():
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = ops.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, axis=1)
        # get according batch index for each expert
        self._batch_index = ops.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._batch_index = self._batch_index.int()
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).asnumpy().tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[ops.flatten(self._batch_index, start_dim=0)]
        self._nonzero_gates = ops.gather_elements(gates_exp, 1, self._expert_index)
        self.cat = ops.Concat(axis=0)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        if isinstance(inp, list):
            inps, result = [], []
            for index in self._batch_index:
                inps.append(inp[index])
            i = 0
            for index in self._part_sizes:
                result.append(inps[i:i + index])
                i += index
            return result
        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return ops.split(inp_exp, self._part_sizes, axis=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = self.cat(expert_out).exp()

        if multiply_by_gates:
            stitched = ops.mul(stitched, self._nonzero_gates)
        zeros = ops.zeros((self._gates.shape[0], expert_out[-1].shape[1]))
        # combine samples that have been processed by the same k experts
        combined = ops.index_add(zeros, self._batch_index, stitched.float(), 0)
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return ops.split(self._nonzero_gates, self._part_sizes, dim=0)


class EDISMOE(nn.Cell):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, num_experts, hidden_size, noisy_gating, k,
                 block_num, embedding_size, filter_num, out_dim):
        super(EDIS_MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        models = []
        for _ in range(self.num_experts):
            models.append(MGraphDTA(block_num, embedding_size, filter_num, out_dim))
        self.experts = nn.CellList(models)
        self.w_gate = Parameter(ops.zeros((input_size, num_experts), dtype=ms.float32), requires_grad=True)
        self.w_noise = Parameter(ops.zeros((input_size, num_experts), dtype=ms.float32), requires_grad=True)

        self.softplus = ops.Softplus()
        self.softmax = nn.Softmax(axis=1)
        self.mea = Parameter(Tensor(np.array([0.0]), ms.float32), requires_grad=False)
        self.std = Parameter(Tensor(np.array([1.0]), ms.float32), requires_grad=False)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if ops.shape(x)[0] == 1:
            return Tensor(np.array([0]), dtype=x.dtype)
        return x.float().var(ddof=True) / (x.float().mean() ** 2 + eps)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (ops.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = ops.zeros_like(logits)
        gates = ops.scatter(zeros, 1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def construct(self, data, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)

        expert_outputs = [self.experts[i](*copy.deepcopy(data))[0] for i in range(self.num_experts)]
        out_experts = []
        for i, out in enumerate(expert_outputs):
            out_experts.append(dispatcher.dispatch(out)[i].unsqueeze(1))
        y = dispatcher.combine(out_experts)
        return y, loss

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(axis=0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = ms.ops.shape(clean_values)[0]
        m = ms.ops.shape(noisy_top_values)[1]
        top_values_flat = ops.flatten(noisy_top_values, start_dim=0)

        threshold_positions_if_in = ops.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = ops.unsqueeze(ops.gather_elements(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = ms.Tensor.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = ops.unsqueeze(ops.gather_elements(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mea, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = ms.Tensor.where(is_in, prob_if_in, prob_if_out)
        return prob


class Conv1dReLU(nn.Cell):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.SequentialCell(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      pad_mode="pad", padding=padding, has_bias=True),
            nn.ReLU()
        )

    def construct(self, x):
        return self.inc(x)


class StackCNN(nn.Cell):
    """cnn"""
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        d = collections.OrderedDict()
        d["conv_layer0"] = Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding)
        for layer_idx in range(layer_num - 1):
            d[f"conv_layer{layer_idx + 1}"] = Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size,
                                                         stride=stride, padding=padding)
        d['pool_layer'] = nn.AdaptiveMaxPool1d(1)
        self.inc = nn.SequentialCell(d)

    def construct(self, x):
        y = self.inc(x)
        y = ops.squeeze(y, axis=-1)

        return y


class TargetRepresentation(nn.Cell):
    """target representation"""
    def __init__(self, block_num, embedding_num):
        super().__init__()
        self.block_list = nn.CellList()
        for block_idx in range(block_num):
            self.block_list.append(StackCNN(block_idx + 1, embedding_num, 96, 3))

        self.cat = ops.Concat(axis=-1)
        self.linear = nn.Dense(block_num * 96, 96)

    def construct(self, x):
        feats = [block(x) for block in self.block_list]
        x = self.cat(feats)
        x = self.linear(x)
        return x


class GraphConv(GNNCell):
    """GCN"""
    def __init__(self,
                 in_feat_size: int,
                 out_size: int,
                 bias: bool = True):
        super().__init__()
        assert isinstance(in_feat_size, int) and in_feat_size > 0, "in_feat_size must be positive int"
        assert isinstance(out_size, int) and out_size > 0, "out_size must be positive int"
        self.in_feat_size = in_feat_size
        self.out_size = out_size

        in_feat_size = (in_feat_size, in_feat_size)
        self.lin_rel = nn.Dense(in_feat_size[0], out_size, has_bias=bias)
        self.lin_root = nn.Dense(in_feat_size[1], out_size, has_bias=False)

    def construct(self, x, g: Graph):
        """
        Construct function for GraphConv.
        """
        x = ops.Squeeze()(x)
        x_r = x
        g.set_vertex_attr({"x": x})
        for v in g.dst_vertex:
            v.x = g.sum([u.x for u in v.innbs])
        x = [v.x for v in g.dst_vertex]
        x = self.lin_rel(x)
        x = self.lin_root(x_r) + x
        return x


class NodeLevelBatchNorm(_BatchNorm):
    r"""
    Applies Batch Normalization over a batch of graph data.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    batch_nodes_dim: all nodes of a batch graph
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.9, affine=True,
                 use_batch_statistics=None):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, use_batch_statistics=use_batch_statistics)

    def construct(self, input_data):
        exponential_average_factor = self.momentum
        return ops.batch_norm(
            input_data, self.moving_mean, self.moving_variance, self.gamma, self.beta,
            self.training,
            exponential_average_factor, self.eps)


class GraphConvBn(nn.Cell):
    """GCB"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)
        self.relu = nn.ReLU()

    def construct(self, data):
        x, edge_index, atom_n, edge_n = data['x'], data['edge_index'], data['atom_n'], data['edge_n']
        y = self.conv(x, edge_index[0], edge_index[1], atom_n, edge_n)
        mask_tmp = ops.ExpandDims()(data['x_mask'], -1)
        y = y * mask_tmp
        y = self.norm(y)
        data['x'] = self.relu(y)
        data['x'] = data['x'] * mask_tmp

        return data


class DenseLayer(nn.Cell):
    """Dense Layer"""
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)
        self.cat = ops.Concat(axis=1)

    def bn_function(self, data):
        concated_features = self.cat(data['x'])
        data['x'] = concated_features
        data = self.conv1(data)

        return data

    def construct(self, data):
        if isinstance(data['x'], Tensor):
            data['x'] = [data['x']]
        data = self.bn_function(data)
        data = self.conv2(data)

        return data


class DenseBlock(nn.Cell):
    """Dense Block"""
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.dense_layer = nn.CellList()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.dense_layer.append(layer)
        self.cat = ops.Concat(axis=1)

    def construct(self, data):
        features = [data['x']]
        for layer in self.dense_layer:
            data = layer(data)
            features.append(data['x'])
            data['x'] = features

        data['x'] = self.cat(data['x'])

        return data


class GraphDenseNet(nn.Cell):
    """Graph Dense Network"""
    def __init__(self, out_dim, num_input_features, block_config, bn_sizes, growth_rate=32):
        super().__init__()
        d = collections.OrderedDict()
        d['conv0'] = GraphConvBn(num_input_features, 32)
        num_input_features = 32
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i])
            d[f'block{i + 1}'] = block
            num_input_features += int(num_layers * growth_rate)
            trans = GraphConvBn(num_input_features, num_input_features // 2)
            d[f'transition{i + 1}'] = trans
            num_input_features = num_input_features // 2

        self.features = nn.SequentialCell(d)
        self.atom_num = 144
        self.edge_num = 320
        self.mean_pooling = AvgPooling()
        self.classifier = nn.Dense(num_input_features, out_dim)

    def construct(self, data):
        """Graph Dense Network"""
        batch_size = ops.shape(data['target'])[0]
        data['atom_n'] = self.atom_num * batch_size
        data['edge_n'] = self.edge_num * batch_size
        graph_mask = np.ones(batch_size).tolist()
        graph_mask.append(0)
        graph_mask = np.array(graph_mask)
        graph_mask = ms.Tensor(graph_mask, ms.int32)

        data = self.features(data)
        x = self.mean_pooling(data['x'], data['edge_index'][0], data['edge_index'][1], data['atom_n'], data['edge_n'],
                              data['batch'], data['batch'], graph_mask)

        x = x[:batch_size]
        x = self.classifier(x)

        return x


class MGraphDTA(nn.Cell):
    """MGraphDTA"""
    def __init__(self, block_num, embedding_size=128, filter_num=32, out_dim=1):
        super().__init__()
        self.protein_encoder = TargetRepresentation(block_num, embedding_size)
        self.ligand_encoder = GraphDenseNet(num_input_features=22, out_dim=filter_num * 3,
                                            block_config=[8, 8, 8], bn_sizes=[2, 2, 2])

        self.cat = ops.Concat(axis=1)
        self.classifier = nn.SequentialCell(
            nn.Dense(filter_num * 3 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.9),
            nn.Dense(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.9),
            nn.Dense(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.9),
            nn.Dense(256, out_dim)
        )

    def construct(self, x, edge_attr, edge_index, target, batch, x_mask=None):
        """MGraphDTA"""
        target = target.float()
        target = ops.reshape(target, (-1, 1200, 33))
        target = ops.permute(target, (0, 2, 1))
        protein_x = self.protein_encoder(target)

        data = {'batch': batch, 'x': x, 'edge_attr': edge_attr, 'edge_index': edge_index, 'target': target,
                'x_mask': x_mask}
        ligand_x = self.ligand_encoder(data)

        feature = self.cat((protein_x, ligand_x))
        out = self.classifier(feature)
        return out, feature