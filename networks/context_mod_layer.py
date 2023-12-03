#!/usr/bin/env python3
# Copyright 2019 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :utils/context_mod_layer.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :10/18/2019
# @version        :1.0
# @python_version :3.6.8
"""
Context-modulation layer
------------------------

This module should represent a special gain-modulation layer that can modulate
neural computation based on an external context.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn

class ContextModLayer(nn.Module):
    r"""Implementation of a layer that can apply context-dependent modulation on
    the level of neuronal computation.

    The layer consists of two parameter vectors: gains :math:`\mathbf{g}`
    and shifts :math:`\mathbf{s}`, whereas gains represent a multiplicative
    modulation of input activations and shifts an additive modulation,
    respectively.

    Note, the weight vectors :math:`\mathbf{g}` and :math:`\mathbf{s}` might
    also be passed to the :meth:`forward` method, where one may pass a separate
    set of parameters for each sample in the input batch.

    Example:
        Assume that a :class:`ContextModLayer` is applied between a linear
        (fully-connected) layer
        :math:`\mathbf{y} \equiv W \mathbf{x} + \mathbf{b}` with input
        :math:`\mathbf{x}` and a nonlinear activation function
        :math:`z \equiv \sigma(y)`.

        The layer-computation in such a case will become

        .. math::

            \sigma \big( (W \mathbf{x} + \mathbf{b}) \odot \mathbf{g} + \
            \mathbf{s} \big)

    Attributes:
        weights: A list of all internal weights of this layer. If all weights
            are assumed to be generated externally, then this attribute will be
            ``None``.
        param_shapes (list): A list of list of integers. Each list represents
            the shape of a parameter tensor. Note, this attribute is
            independent of the attribute :attr:`weights`, it always comprises
            the shapes of all weight tensors as if the network would be stand-
            alone (i.e., no weights being passed to the :meth:`forward` method).

            .. note::
                The weights passed to the :meth:`forward` method might deviate
                from these shapes, as we allow passing a distinct set of
                parameters per sample in the input batch.
        param_shapes_meta (list): List of strings. Each entry represents the
            meaning of the corresponding entry in :attr:`param_shapes`. The
            following keywords are possible:

                - ``'gain'``: The corresponding shape in :attr:`param_shapes`
                  denotes the gain :math:`\mathbf{g}` parameter.
                - ``'shift'``: The corresponding shape in :attr:`param_shapes`
                  denotes the shift :math:`\mathbf{s}` parameter.
        num_ckpts (int): The number of existing weight checkpoints (i.e., how
            often the method :meth:`checkpoint_weights` was called).
        gain_offset_applied (bool): Whether constructor argument
            ``apply_gain_offset`` was activated.
        gain_softplus_applied (bool): Whether constructor argument
            ``apply_gain_softplus`` was activated.
        has_gains (bool): Is ``True`` if ``no_gains`` was not set in the
            constructor.
        has_shifts (bool): Is ``True`` if ``no_shifts`` was not set in the
            constructor.


    Args:
        num_features (int or tuple): Number of units in the layer (size of
            parameter vectors :math:`\mathbf{g}` and :math:`\mathbf{s}`).

            In case a ``tuple`` of integers is provided, the gain
            :math:`\mathbf{g}` and shift :math:`\mathbf{s}` parameters will
            become multidimensional tensors with the shape being prescribed
            by ``num_features``. Please note the `broadcasting rules`_ as
            :math:`\mathbf{g}` and :math:`\mathbf{s}` are simply multiplied
            or added to the input.

            Example:
                Consider the output of a convolutional layer with output shape
                ``[B,C,W,H]``. In case there should be a scalar gain and shift
                per feature map, ``num_features`` could be ``[C,1,1]`` or
                ``[1,C,1,1]`` (one might also pass a shape ``[B,C,1,1]`` to the
                :meth:`forward` method to apply separate shifts and gains per
                sample in the batch).

                Alternatively, one might want to provide shift and gain per
                output unit, i.e., ``num_features`` should be ``[C,W,H]``. Note,
                that due to weight sharing, all output activities within a
                feature map are computed using the same weights, which is why it
                is common practice to share shifts and gains within a feature
                map (e.g., in Spatial Batch-Normalization).
        no_weights (bool): If ``True``, the layer will have no trainable weights
            (:math:`\mathbf{g}` and :math:`\mathbf{s}`). Hence, weights are
            expected to be passed to the :meth:`forward` method.
        no_gains (bool): If ``True``, no gain parameters :math:`\mathbf{g}` will
            be modulating the input activity.

            .. note::
                Arguments ``no_gains`` and ``no_shifts`` might not be activated
                simultaneously!
        no_shifts (bool): If ``True``, no shift parameters :math:`\mathbf{s}`
            will be modulating the input activity.
        apply_gain_offset (bool, optional): If activated, this option will apply
            a constant offset of 1 to all gains, i.e., the computation becomes

            .. math::

                \sigma \big( (W \mathbf{x} + \mathbf{b}) \odot \
                (1 + \mathbf{g}) + \mathbf{s} \big)

            When could that be useful? In case the gains and shifts are
            generated by the same hypernetwork, a meaningful initialization
            might be difficult to achieve (e.g., such that gains are close to 1
            and shifts are close to 0 at the beginning). Therefore, one might
            initialize the hypernetwork such that all outputs are close to zero
            at the beginning and the constant shift ensures that meaningful
            gains are applied.
        apply_gain_softplus (bool, optional): If activated, this option will
            enforce poitive gain modulation by sending the gain weights
            :math:`\mathbf{g}` through a softplus function (scaled by :math:`s`,
            see ``softplus_scale``).

            .. math::

                \mathbf{g} = \frac{1}{s} \log(1+\exp(\mathbf{g} \cdot s))
        softplus_scale (float): If option ``apply_gain_softplus`` is ``True``,
            then this will determine the sclae of the softplus function.

    .. _broadcasting rules:
        https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-\
        semantics
    """
    def __init__(self, num_features, no_weights=False, no_gains=False,
                 no_shifts=False, apply_gain_offset=False,
                 apply_gain_softplus=False, softplus_scale=1.):
        super(ContextModLayer, self).__init__()

        assert(isinstance(num_features, (int, list, tuple)))
        if not isinstance(num_features, int):
            for nf in num_features:
                assert(isinstance(nf, int))
        else:
            num_features = [num_features]

        assert(not no_gains or not no_shifts)
        self._num_features = num_features
        self._no_weights = no_weights
        self._no_gains = no_gains
        self._no_shifts = no_shifts
        self._apply_gain_offset = apply_gain_offset
        self._apply_gain_softplus = apply_gain_softplus
        self._sps = softplus_scale

        if apply_gain_offset and apply_gain_softplus:
            raise ValueError('Options "apply_gain_offset" and ' +
                             '"apply_gain_softplus" are not compatible.')

        self._weights = None
        self._param_shapes = [num_features] * (1 if no_gains or no_shifts \
                                               else 2)
        self._param_shapes_meta = ([] if no_gains else ['gain']) + \
            ([] if no_shifts else ['shift'])
        self.register_buffer('_num_ckpts', torch.tensor(0, dtype=torch.long))

        if not no_weights:
            self._weights = nn.ParameterList()

            if not no_gains:
                self.register_parameter('gain', nn.Parameter( \
                    torch.Tensor(*num_features), requires_grad=True))
                self._weights.append(self.gain)
                if apply_gain_offset:
                    nn.init.zeros_(self.gain)
                else:
                    nn.init.ones_(self.gain)
            else:
                self.register_parameter('gain', None)

            if not no_shifts:
                self.register_parameter('shift', nn.Parameter( \
                    torch.Tensor(*num_features), requires_grad=True))
                self._weights.append(self.shift)
                nn.init.zeros_(self.shift)
            else:
                self.register_parameter('shift', None)

    @property
    def weights(self):
        """Getter for read-only attribute :attr:`weights`.

        Returns:
            A :class:`torch.nn.ParameterList` or ``None``, if no parameters are
            internally maintained.
        """
        return self._weights

    @property
    def param_shapes(self):
        """Getter for read-only attribute :attr:`param_shapes`.

        Returns:
            (list): A list of lists of integers.
        """
        return self._param_shapes

    @property
    def param_shapes_meta(self):
        """Getter for read-only attribute :attr:`param_shapes_meta`.

        Returns:
            (list): A list of strings.
        """
        return self._param_shapes_meta

    @property
    def num_ckpts(self):
        """Getter for read-only attribute :attr:`num_ckpts`.

        Returns:
            (int)
        """
        return self._num_ckpts

    @property
    def gain_offset_applied(self):
        r"""Getter for read-only attribute :attr:`gain_offset_applied`.

        Returns:
            (bool): Whether an offset for the gain :math:`\mathbf{g}` is
            applied.
        """
        return self._apply_gain_offset

    @property
    def gain_softplus_applied(self):
        r"""Getter for read-only attribute :attr:`gain_softplus_applied`.

        Returns:
            (bool): Whether a softplus function for the gain :math:`\mathbf{g}`
            is applied.
        """
        return self._apply_gain_softplus

    @property
    def has_gains(self):
        r"""Getter for read-only attribute :attr:`has_gains`.

        Returns:
            (bool): Whether gains :math:`\mathbf{g}` are part of the computation
            of this layer.
        """
        return not self._no_gains

    @property
    def has_shifts(self):
        r"""Getter for read-only attribute :attr:`has_shifts`.

        Returns:
            (bool): Whether shifts :math:`\mathbf{s}` are part of the
            computation of this layer.
        """
        return not self._no_shifts

    def forward(self, x, weights=None, ckpt_id=None, bs_dim=0):
        """Apply context-dependent gain modulation.

        Computes :math:`\mathbf{x} \odot \mathbf{g} + \mathbf{s}`, where
        :math:`\mathbf{x}` denotes the input activity ``x``.

        Args:
            x: The input activity.
            weights: Weights that should be used instead of the internally
                maintained once (determined by attribute :attr:`weights`). Note,
                if ``no_weights`` was ``True`` in the constructor, then this
                parameter is mandatory.

                Usually, the shape of the passed weights should follow the
                attribute :attr:`param_shapes`, which is a tuple of shapes
                ``[[num_features], [num_features]]`` (at least for linear
                layers, see docstring of argument ``num_features`` in the
                constructor for more details). However, one may also
                specify a seperate set of context-mod parameters per input
                sample. Assume ``x`` has shape ``[num_samples, num_features]``.
                Then ``weights`` may have the shape
                ``[[num_samples, num_features], [num_samples, num_features]]``.
            ckpt_id (int): This argument can be set in case a checkpointed set
                of weights should be used to compute the forward pass (see
                method :meth:`checkpoint_weights`).

                .. note::
                    This argument is ignored if ``weights`` is not ``None``.
            bs_dim (int): Batch size dimension in input tensor ``x``.

        Returns:
            The modulated input activity.
        """
        if self._no_weights and weights is None:
            raise ValueError('Layer was generated without weights. ' +
                             'Hence, "weights" option may not be None.')

        if weights is not None and ckpt_id is not None:
            warn('Context-mod layer received weights as well as the request ' +
                 'to load checkpointed weights. The request to load ' +
                 'checkpointed weights will be ignored.')

        # FIXME I haven't thoroughly checked whether broadcasting works
        # correctly if `bs_dim != 0`.
        batch_size = x.shape[bs_dim]

        if weights is None:
            gain, shift = self.get_weights(ckpt_id=ckpt_id)

            if self._no_gains:
                weights = [shift]
            elif self._no_shifts:
                weights = [gain]
            else:
                weights = [gain, shift]

        else:
            assert(len(weights) in [1, 2])
            nfl = len(self._num_features)
            nb = len(x.shape)
            for p in weights:
                # Note, the user might add the batch dimension when providing
                # gains and shifts, such that there are separate gain and shift
                # parameters per sample in the batch.
                assert(len(p.shape) in [nfl, nb])
                if len(p.shape) == nfl:
                    assert(np.all(np.equal(p.shape, self._num_features)))
                else:
                    # One set of parameters per sample in the batch.
                    assert(p.shape[0] == batch_size and \
                           np.all(np.equal(p.shape[1:], self._num_features)))

        gain = None
        shift = None

        if self._no_gains:
            assert(len(weights) == 1)
            shift = weights[0]
        elif self._no_shifts:
            assert(len(weights) == 1)
            gain = weights[0]
        else:
            assert(len(weights) == 2)
            gain = weights[0]
            shift = weights[1]

        if gain is not None:
            x = x.mul(self.preprocess_gain(gain))

        if shift is not None:
            x = x.add(shift)

        return x

    def preprocess_gain(self, gain):
        r"""Obtains gains :math:`\mathbf{g}` used for mudulation.
        
        Depending on the user configuration, gains might be preprocessed before
        applied for context-modulation (e.g., see attributes
        :attr:`gain_offset_applied` or :attr:`gain_softplus_applied`). This
        method transforms raw gains such that they can be applied to the network
        activation.

        Note:
            This method is called by the :meth:`forward` to transform given
            gains.

        Args:
            gain (torch.Tensor): A gain tensor.

        Returns:
            (torch.Tensor): The transformed gains.
        """
        if self._apply_gain_softplus:
            gain = 1. / self._sps * F.softplus(gain * self._sps)
        elif self._apply_gain_offset:
            gain = gain + 1.

        return gain

    def checkpoint_weights(self, device=None, no_reinit=False):
        """Checkpoint and reinit the current weights.

        Buffers for a new checkpoint will be registered and the current weights
        will be copied into them. Additionally, the current weights will be
        reinitialized (gains to 1 and shifts to 0).

        Calling this function will also increment the attribute
        :attr:`num_ckpts`.

        Note:
            This method uses the method :meth:`torch.nn.Module.register_buffer`
            rather than the method :meth:`torch.nn.Module.register_parameter` to
            create checkpoints. The reason is, that we don't want the
            checkpoints to appear as trainable weights (when calling
            :meth:`torch.nn.Module.parameters`). However, that means that
            training on checkpointed weights cannot be continued unless they are
            copied back into an actual :class:`torch.nn.Parameter` object.

        Args:
            device (optional): If not provided, the newly created checkpoint
                will be moved to the device of the current weights.
            no_reinit (bool): If ``True``, the actual :attr:`weights` will not
                be reinitialized.
        """
        assert(not self._no_weights)

        if device is None:
            if self.gain is not None:
                device = self.gain.device
            else:
                device = self.shift.device

        gname, sname = self._weight_names(self._num_ckpts)
        self._num_ckpts += 1

        if not self._no_gains:
            self.register_buffer(gname, torch.empty_like(self.gain,
                                                         device=device))
            getattr(self, gname).data = self.gain.detach().clone()
            if not no_reinit:
                if self._apply_gain_offset:
                    nn.init.zeros_(self.gain)
                else:
                    nn.init.ones_(self.gain)
        else:
            self.register_buffer(gname, None)

        if not self._no_shifts:
            self.register_buffer(sname, torch.empty_like(self.shift,
                                                         device=device))
            getattr(self, sname).data = self.shift.detach().clone()
            if not no_reinit:
                nn.init.zeros_(self.shift)
        else:
            self.register_buffer(gname, None)

    def get_weights(self, ckpt_id=None):
        """Get the current (or a set of checkpointed) weights of this context-
        mod layer.

        Args:
            ckpt_id (optional): ID of checkpoint. If not provided, the current
                set of weights is returned.
                If :code:`ckpt_id == self.num_ckpts`, then this method also
                returns the current weights, as the checkpoint has not been
                created yet.

        Returns:
            (tuple): Tuple containing:

            - **gain**: Is ``None`` if layer has no gains.
            - **shift**: Is ``None`` if layer has no shifts.
        """
        if ckpt_id is None or ckpt_id == self.num_ckpts:
            return self.gain, self.shift
        assert(ckpt_id >= 0 and ckpt_id < self.num_ckpts)

        gname, sname = self._weight_names(ckpt_id)

        gain = getattr(self, gname)
        shift = getattr(self, sname)

        return gain, shift

    def _weight_names(self, ckpt_id):
        """Get the buffer names for checkpointed gain and shift weights
        depending on the ``ckpt_id``, i.e., the ID of the checkpoint.

        Args:
            ckpt_id: ID of weight checkpoint.

        Returns:
            (tuple): Tuple containing:

            - **gain_name**
            - **shift_name**
        """
        gain_name = 'gain_ckpt_%d' % ckpt_id
        shift_name = 'shift_ckpt_%d' % ckpt_id

        return gain_name, shift_name

    def normal_init(self, std=1.):
        """Reinitialize internal weights using a normal distribution.

        Args:
            std (float): Standard deviation of init.
        """
        if self._no_weights:
            raise ValueError('Method is not applicable to layers without ' +
                             'internally maintained weights.')

        if not self._no_gains:
            if self._apply_gain_offset:
                nn.init.normal_(self.gain, std=std)
            else:
                nn.init.normal_(self.gain, mean=1., std=std)

        if not self._no_shifts:
            nn.init.normal_(self.shift, std=std)

    def uniform_init(self, width=1.):
        """Reinitialize internal weights using a uniform distribution.

        Args:
            width (float): The range of the uniform init will be determined
                as ``[mean-width, mean+width]``, where ``mean`` is 0 for shifts
                and 1 for gains.
        """
        if self._no_weights:
            raise ValueError('Method is not applicable to layers without ' +
                             'internally maintained weights.')

        if not self._no_gains:
            if self._apply_gain_offset:
                nn.init.uniform_(self.gain, a=-width, b=width)
            else:
                nn.init.uniform_(self.gain, a=1.-width, b=1.+width)

        if not self._no_shifts:
            nn.init.uniform_(self.shift, a=-width, b=width)

    def sparse_init(self, sparsity=.8):
        """Reinitialize internal weights sparsely.

        Gains will be initialized such that ``sparisity * 100`` percent of them
        will be 0, the remaining ones will be 1. Shifts are initialized to 0.

        Args:
            sparsity (float): A number between 0 and 1 determining the
                spasity level of gains.
        """
        if self._no_weights:
            raise ValueError('Method is not applicable to layers without ' +
                             'internally maintained weights.')
        assert 0 <= sparsity <= 1

        if not self._no_gains:
            num_zeros = int(self.gain.numel() * sparsity)

            inds = np.zeros(self.gain.numel(), dtype=bool)
            inds = inds.reshape(-1)
            inds[:num_zeros] = True
            np.random.shuffle(inds)
            inds = inds.reshape(*self.gain.shape)
            inds = torch.from_numpy(inds).to(self.gain.device)

            if self._apply_gain_offset:
                nn.init.zeros_(self.gain)
                self.gain.data[inds] = -1.
            else:
                nn.init.ones_(self.gain)
                self.gain.data[inds] = 0.

        if not self._no_shifts:
            nn.init.zeros_(self.shift)

if __name__ == '__main__':
    pass

