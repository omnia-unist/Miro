import torch
import torch.nn as nn
from torch.autograd import Variable
import inspect
import numpy as np
import math
from warnings import warn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from networks.torch_utils import init_params

def shapes_to_num_weights(dims):
    """The number of parameters contained in a list of tensors with the
    given shapes.

    Args:
        dims: List of tensor shapes. For instance, the attribute
            :attr:`hyper_shapes_learned`.

    Returns:
        (int)
    """
    return int(np.sum([np.prod(l) for l in dims]))
def get_default_args(func):
    """Get the default values of all keyword arguments for a given function.

    Args:
        func: A function handle.

    Returns:
        (dict): Dictionary with keyword argument names as keys and their
        default value as values.
    """
    # The code from this function has been copied from (accessed: 02/28/2020):
    #   https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    #
    # NOTE Our copyright and license does not apply for this function.
    # We use this code WITHOUT ANY WARRANTIES.
    #
    # Instead, the code in this method is licensed under CC BY-SA 3.0:
    #    https://creativecommons.org/licenses/by-sa/3.0/
    #
    # The code stems from an answer by user "mgilson":
    #    https://stackoverflow.com/users/748858/mgilson
    signature = inspect.signature(func)

    return {
        k: v.default for k, v in signature.parameters.items() \
                    if v.default is not inspect.Parameter.empty
    }
def _parse_context_mod_args(cm_kwargs):
    """Parse context-modulation arguments for a class.

    This function first loads the default values of all context-mod
    arguments passed to class :class:`mnets.mlp.MLP`. If any of these
    arguments is not occurring in the dictionary ``cm_kwargs``, then they
    will be added using the default value from class :class:`mnets.mlp.MLP`.

    Args:
        cm_kwargs (dict): A dictionary, that is modified in place (i.e.,
            missing keys are added).

    Returns:
        (list): A list of key names from ``cm_kwargs`` that are not related
        to context-modulation, i.e., unknown to this function.
    """
    from networks.mlp import MLP

    # All context-mod related arguments in `mnets.mlp.MLP.__init__`.
    cm_keys = ['use_context_mod',
                'context_mod_inputs',
                'no_last_layer_context_mod',
                'context_mod_no_weights',
                'context_mod_post_activation',
                'context_mod_gain_offset',
                'context_mod_gain_softplus']

    default_cm_kwargs = get_default_args(MLP.__init__)

    for k in cm_keys:
        assert k in default_cm_kwargs.keys()
        if k not in cm_kwargs.keys():
            cm_kwargs[k] = default_cm_kwargs[k]

    # Extract keyword arguments that do not belong to context-mod.
    unknown_kwargs = []
    for k in cm_kwargs.keys():
        if k not in default_cm_kwargs.keys():
            unknown_kwargs.append(k)

    return unknown_kwargs     
class SimpleRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, n_in=128, rnn_layers=(32,), fc_layers_pre=(),
                 fc_layers=(*[],100), activation=torch.nn.Tanh(), use_lstm=True,
                 use_bias=True, no_weights=False,
                 init_weights=None, kaiming_rnn_init=True,
                 context_mod_last_step=False,
                 context_mod_num_ts=-1,
                 context_mod_separate_layers_per_ts=False,
                 verbose=True,
                 **kwargs):
        super(SimpleRNN, self).__init__()
        nn.Module.__init__(self)
        ### IMPORTANT NOTE FOR DEVELOPERS IMPLEMENTING THIS INTERFACE ###
        ### The following member variables have to be set by all classes that
        ### implement this interface.
        ### Please always verify your implementation using the method
        ### `_is_properly_setup` at the end the constructor of any class
        ### implementing this interface.
        self._internal_params = None
        self._param_shapes = None
        # You don't have to implement this following attribute, but it might
        # be helpful, for instance for hypernetwork initialization.
        self._param_shapes_meta = None
        self._hyper_shapes_learned = None
        # You don't have to implement this following attribute, but it might
        # be helpful, for instance for hypernetwork initialization.
        self._hyper_shapes_learned_ref = None
        self._hyper_shapes_distilled = None
        self._has_bias = None
        self._has_fc_out = None
        self._mask_fc_out = None
        self._has_linear_out = None
        self._layer_weight_tensors = None
        self._layer_bias_vectors = None
        self._batchnorm_layers = None
        self._context_mod_layers = None

        ### The rest will be taken care of automatically.
        # This will be set automatically based on attribute `_param_shapes`.
        self._num_params = None
        # This will be set automatically based on attribute `_weights`.
        self._num_internal_params = None

        # Deprecated, use `_hyper_shapes_learned` instead.
        self._hyper_shapes = None
        # Deprecated, use `_param_shapes` instead.
        self._all_shapes = None
        # Deprecated, use `_internal_params` instead.
        self._weights = None
        
        
        self._bptt_depth = -1

        if activation is None or isinstance(activation, (torch.nn.ReLU, \
                torch.nn.Tanh)):
            self._a_fun = activation
        else:
            raise ValueError('Only linear, relu and tanh activations are ' + \
                             'allowed for recurrent networks.')
       
        if len(rnn_layers) == 0:
            raise ValueError('The network always needs to have at least one ' +
                             'recurrent layer.')
        if len(fc_layers) == 0:
            has_rec_out_layer = True
            #n_out = rnn_layers[-1]
        else:
            has_rec_out_layer = False
            #n_out = fc_layers[-1]
            
        self._n_in = n_in
        self._rnn_layers = list(rnn_layers)
        self._fc_layers_pre = list(fc_layers_pre)
        self._fc_layers = list(fc_layers)

        self._no_weights = no_weights

        ### Parse or set context-mod arguments ###
        rem_kwargs = _parse_context_mod_args(kwargs)
        if len(rem_kwargs) > 0:
            raise ValueError('Keyword arguments %s unknown.' % str(rem_kwargs))

        self._use_context_mod = kwargs['use_context_mod']
        self._context_mod_inputs = kwargs['context_mod_inputs']
        self._no_last_layer_context_mod = kwargs['no_last_layer_context_mod']
        self._context_mod_no_weights = kwargs['context_mod_no_weights']
        self._context_mod_post_activation = \
            kwargs['context_mod_post_activation']
        self._context_mod_gain_offset = kwargs['context_mod_gain_offset']
        self._context_mod_gain_softplus = kwargs['context_mod_gain_softplus']
        
        # Context-mod options specific to RNNs
        self._context_mod_last_step = context_mod_last_step
        # FIXME We have to specify this option even if
        # `context_mod_separate_layers_per_ts` is False (in order to set
        # sensible parameter shapes). However, the forward method can deal with
        # an arbitrary timestep length.
        self._context_mod_num_ts = context_mod_num_ts
        self._context_mod_separate_layers_per_ts = \
            context_mod_separate_layers_per_ts

        # More appropriate naming of option.
        self._context_mod_outputs = not self._no_last_layer_context_mod

        if context_mod_num_ts != -1:
            if context_mod_last_step:
                raise ValueError('Options "context_mod_last_step" and ' +
                                 '"context_mod_num_ts" are not compatible.')
            if not self._context_mod_no_weights and \
                    not context_mod_separate_layers_per_ts:
                raise ValueError('When applying context-mod per timestep ' +
                    'while maintaining weights internally, option' +
                    '"context_mod_separate_layers_per_ts" must be set.')
        ### Parse or set context-mod arguments - DONE ###
        
        self._has_bias = use_bias
        self._has_fc_out = True if not has_rec_out_layer else False
        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer.
        self._mask_fc_out = True if not has_rec_out_layer else False
        # Note, recurrent layers always use non-linearities and their activities
        # are squashed by a non-linearity (otherwise, internal states could
        # vanish/explode with increasing sequence length).
        self._has_linear_out = True if not has_rec_out_layer else False

        self._param_shapes = []
        self._param_shapes_meta = []

        self._weights = None if no_weights and self._context_mod_no_weights \
            else nn.ParameterList()
        self._hyper_shapes_learned = None \
            if not no_weights and not self._context_mod_no_weights else []
        self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned \
            is None else []

        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        self._use_lstm = use_lstm
        if use_lstm:
            self._rnn_fct = self.lstm_rnn_step
        else:
            self._rnn_fct = self.basic_rnn_step
        #################################################
        ### Define and initialize context mod weights ###
        #################################################

        # The context-mod layers consist of sequential layers ordered as:
        # - initial fully-connected layers if len(fc_layers_pre)>0
        # - recurrent layers
        # - final fully-connected layers if len(fc_layers)>0

        self._context_mod_layers = nn.ModuleList() if self._use_context_mod \
            else None

        self._cm_rnn_start_ind = 0
        self._num_fc_cm_layers = None
        if self._use_context_mod:
            cm_layer_inds = []
            cm_shapes = []

            # Gather sizes of all activation vectors within the network that
            # will be subject to context-modulation.
            if self._context_mod_inputs:
                self._cm_rnn_start_ind += 1
                cm_shapes.append([n_in])

                # We reserve layer zero for input context-mod. Otherwise, there
                # is no layer zero.
                cm_layer_inds.append(0)

            if len(fc_layers_pre) > 0:
                self._cm_rnn_start_ind += len(fc_layers_pre)

            # We use odd numbers for actual layers and even number for all
            # context-mod layers.
            rem_cm_inds = range(2, 2*(len(fc_layers_pre)+len(rnn_layers)+\
                len(fc_layers))+1, 2)

            num_rec_cm_layers = len(rnn_layers)
            if has_rec_out_layer and not self._context_mod_outputs:
                num_rec_cm_layers -= 1
            self._num_rec_cm_layers = num_rec_cm_layers

            jj = 0
            # Add initial fully-connected context-mod layers.
            num_fc_pre_cm_layers = len(fc_layers_pre)
            self._num_fc_pre_cm_layers = num_fc_pre_cm_layers
            for i in range(num_fc_pre_cm_layers):
                cm_shapes.append([fc_layers_pre[i]])
                cm_layer_inds.append(rem_cm_inds[jj])
                jj += 1

            # Add recurrent context-mod layers.
            for i in range(num_rec_cm_layers):
                if context_mod_num_ts != -1:
                    if context_mod_separate_layers_per_ts:
                        cm_rnn_shapes = [[rnn_layers[i]]] * context_mod_num_ts
                    else:
                        # Only a single context-mod layer will be added, but we
                        # directly edit the correponding `param_shape` later.
                        assert self._context_mod_no_weights
                        cm_rnn_shapes = [[rnn_layers[i]]]
                else:
                    cm_rnn_shapes = [[rnn_layers[i]]]

                cm_shapes.extend(cm_rnn_shapes)
                cm_layer_inds.extend([rem_cm_inds[jj]] * len(cm_rnn_shapes))
                jj += 1

            # Add final fully-connected context-mod layers.
            num_fc_cm_layers = len(fc_layers)
            if num_fc_cm_layers > 0 and not self._context_mod_outputs:
                num_fc_cm_layers -= 1
            self._num_fc_cm_layers = num_fc_cm_layers
            for i in range(num_fc_cm_layers):
                cm_shapes.append([fc_layers[i]])
                cm_layer_inds.append(rem_cm_inds[jj])
                jj += 1

            self._add_context_mod_layers(cm_shapes, cm_layers=cm_layer_inds)

            if context_mod_num_ts != -1 and not \
                    context_mod_separate_layers_per_ts:
                # In this case, there is only one context-mod layer for each
                # recurrent layer, but we want to have separate weights per
                # timestep.
                # Hence, we adapt the expected parameter shape, such that we
                # get a different set of weights per timestep. This will be
                # split into multiple weights that are succesively fed into the
                # same layer inside the forward method.

                for i in range(num_rec_cm_layers):
                    cmod_layer = \
                        self.context_mod_layers[self._cm_rnn_start_ind+i]
                    cm_shapes_rnn = [[context_mod_num_ts, *s] for s in \
                                      cmod_layer.param_shapes]

                    ps_ind = int(np.sum([ \
                        len(self.context_mod_layers[ii].param_shapes) \
                        for ii in range(self._cm_rnn_start_ind+i)]))
                    self._param_shapes[ps_ind:ps_ind+len(cm_shapes_rnn)] = \
                        cm_shapes_rnn
                    assert self._hyper_shapes_learned is not None
                    self._hyper_shapes_learned[ \
                        ps_ind:ps_ind+len(cm_shapes_rnn)] = cm_shapes_rnn

        ########################
        ### Internal weights ###
        ########################
        prev_dim = self._n_in


        def define_fc_layer_weights(fc_layers, prev_dim, num_prev_layers):
            """Define the weights and shapes of the fully-connected layers.

            Args:
                fc_layers (list): The list of fully-connected layer dimensions.
                prev_dim (int): The output size of the previous layer.
                num_prev_layers (int): The number of upstream layers to the 
                    current one (a layer with its corresponding
                    context-mod layer(s) count as one layer). Count should
                    start at ``1``.

            Returns:
                (int): The output size of the last fully-connected layer
                considered here.
            """
            # FIXME We should instead build an MLP instance. But then we still
            # have to adapt all attributes accordingly.
            for i, n_fc in enumerate(fc_layers):
                s_w = [n_fc, prev_dim]
                s_b = [n_fc] if self._has_bias else None

                for j, s in enumerate([s_w, s_b]):
                    if s is None:
                        continue

                    is_bias = True
                    if j % 2 == 0:
                        is_bias = False

                    if not self._no_weights:
                        self._weights.append(nn.Parameter(torch.Tensor(*s),
                                                            requires_grad=True))
                        if is_bias:
                            self._layer_bias_vectors.append(self._weights[-1])
                        else:
                            self._layer_weight_tensors.append(self._weights[-1])
                    else:
                        self._hyper_shapes_learned.append(s)
                        self._hyper_shapes_learned_ref.append( \
                            len(self.param_shapes))

                    self._param_shapes.append(s)
                    self._param_shapes_meta.append({
                        'name': 'bias' if is_bias else 'weight',
                        'index': -1 if self._no_weights else \
                            len(self._weights)-1,
                        'layer': i * 2 + num_prev_layers, # Odd numbers
                    })

                prev_dim = n_fc

            return prev_dim

        ### Initial fully-connected layers.
        prev_dim = define_fc_layer_weights(self._fc_layers_pre, prev_dim, 1)
        ### Recurrent layers.
        coeff = 4 if self._use_lstm else 1
        for i, n_rec in enumerate(self._rnn_layers):
            # Input-to-hidden
            s_w_ih = [n_rec*coeff, prev_dim]
            s_b_ih = [n_rec*coeff] if use_bias else None

            # Hidden-to-hidden
            s_w_hh = [n_rec*coeff, n_rec]
            s_b_hh = [n_rec*coeff] if use_bias else None

            # Hidden-to-output.
            # Note, for an LSTM cell, the hidden state vector is also the
            # output vector.
            if not self._use_lstm:
                s_w_ho = [n_rec, n_rec]
                s_b_ho = [n_rec] if use_bias else None
            else:
                s_w_ho = None
                s_b_ho = None

            for j, s in enumerate([s_w_ih, s_b_ih, s_w_hh, s_b_hh, s_w_ho,
                                   s_b_ho]):
                if s is None:
                    continue

                is_bias = True
                if j % 2 == 0:
                    is_bias = False

                wtype = 'ih'
                if 2 <= j < 4:
                    wtype = 'hh'
                elif j >=4:
                    wtype = 'ho'

                if not no_weights:
                    self._weights.append(nn.Parameter(torch.Tensor(*s),
                                                      requires_grad=True))
                    if is_bias:
                        self._layer_bias_vectors.append(self._weights[-1])
                    else:
                        self._layer_weight_tensors.append(self._weights[-1])
                else:
                    self._hyper_shapes_learned.append(s)
                    self._hyper_shapes_learned_ref.append( \
                        len(self.param_shapes))

                self._param_shapes.append(s)
                self._param_shapes_meta.append({
                    'name': 'bias' if is_bias else 'weight',
                    'index': -1 if no_weights else len(self._weights)-1,
                    'layer': i * 2 + 1 + 2 * len(fc_layers_pre), # Odd numbers
                    'info': wtype
                })

            prev_dim = n_rec

        ### Fully-connected layers.
        prev_dim = define_fc_layer_weights(self._fc_layers, prev_dim, \
            1 + 2 * len(fc_layers_pre) + 2 * len(rnn_layers))
        ### Initialize weights.
        if init_weights is not None:
            assert self._weights is not None
            assert len(init_weights) == len(self.weights)
            for i in range(len(init_weights)):
                assert np.all(np.equal(list(init_weights[i].shape),
                                       self.weights[i].shape))
                self.weights[i].data = init_weights[i]
        else:
            rec_start = len(fc_layers_pre)
            rec_end = rec_start + len(rnn_layers) * (2 if use_lstm else 3)
            # Note, Pytorch applies a uniform init to its recurrent layers, as
            # defined here:
            # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py#L155
            for i in range(len(self._layer_weight_tensors)):
                if i >=rec_start and i < rec_end:
                    # Recurrent layer weights.
                    if kaiming_rnn_init:
                        init_params(self._layer_weight_tensors[i],
                            self._layer_bias_vectors[i] if use_bias else None)
                    else:
                        a = 1.0 / math.sqrt(rnn_layers[(i-rec_start) // \
                            (2 if use_lstm else 3)])
                        nn.init.uniform_(self._layer_weight_tensors[i], -a, a)
                        if use_bias:
                            nn.init.uniform_(self._layer_bias_vectors[i], -a, a)
                else:
                    # FC layer weights.
                    init_params(self._layer_weight_tensors[i],
                        self._layer_bias_vectors[i] if use_bias else None)

        num_weights = shapes_to_num_weights(self._param_shapes)
        if verbose:
            if self._use_context_mod:
                cm_num_weights =  \
                    shapes_to_num_weights(cm_shapes)

            print('Creating a simple RNN with %d weights' % num_weights
                  + (' (including %d weights associated with-' % cm_num_weights
                     + 'context modulation)' if self._use_context_mod else '')
                  + '.')

        self._is_properly_setup()
        
    def _is_properly_setup(self, check_has_bias=True):
        """This method can be used by classes that implement this interface to
        check whether all required properties have been set."""
        assert(self._param_shapes is not None or self._all_shapes is not None)
        if self._param_shapes is None:
            warn('Private member "_param_shapes" should be specified in each ' +
                 'sublcass that implements this interface, since private ' +
                 'member "_all_shapes" is deprecated.', DeprecationWarning)
            self._param_shapes = self._all_shapes

        if self._hyper_shapes is not None or \
                self._hyper_shapes_learned is not None:
            if self._hyper_shapes_learned is None:
                warn('Private member "_hyper_shapes_learned" should be ' +
                     'specified in each sublcass that implements this ' +
                     'interface, since private member "_hyper_shapes" is ' +
                     'deprecated.', DeprecationWarning)
                self._hyper_shapes_learned = self._hyper_shapes
            # FIXME we should actually assert equality if
            # `_hyper_shapes_learned` was not None.
            self._hyper_shapes = self._hyper_shapes_learned

        assert self._weights is None or self._internal_params is None
        if self._weights is not None and self._internal_params is None:
            # Note, in the future we might throw a deprecation warning here,
            # once "weights" becomes deprecated.
            self._internal_params = self._weights

        assert self._internal_params is not None or \
               self._hyper_shapes_learned is not None

        if self._hyper_shapes_learned is None and \
                self.hyper_shapes_distilled is None:
            # Note, `internal_params` should only contain trainable weights and
            # not other things like running statistics. Thus, things that are
            # passed to an optimizer.
            assert len(self._internal_params) == len(self._param_shapes)

        if self._param_shapes_meta is None:
            # Note, this attribute was inserted post-hoc.
            # FIXME Warning is annoying, programmers will notice when they use
            # this functionality.
            #warn('Attribute "param_shapes_meta" has not been implemented!')
            pass
        else:
            assert(len(self._param_shapes_meta) == len(self._param_shapes))
            for dd in self._param_shapes_meta:
                assert isinstance(dd, dict)
                assert 'name' in dd.keys() and 'index' in dd.keys() and \
                    'layer' in dd.keys()
                assert dd['name'] is None or \
                       dd['name'] in ['weight', 'bias', 'bn_scale', 'bn_shift',
                                      'cm_scale', 'cm_shift', 'embedding']

                assert isinstance(dd['index'], int)
                if self._internal_params is None:
                    assert dd['index'] == -1
                else:
                    assert dd['index'] == -1 or \
                        0 <= dd['index'] < len(self._internal_params)

                assert isinstance(dd['layer'], int)
                assert dd['layer'] == -1 or dd['layer'] >= 0

        if self._hyper_shapes_learned is not None:
            if self._hyper_shapes_learned_ref is None:
                # Note, this attribute was inserted post-hoc.
                # FIXME Warning is annoying, programmers will notice when they
                # use this functionality.
                #warn('Attribute "hyper_shapes_learned_ref" has not been ' +
                #     'implemented!')
                pass
            else:
                assert isinstance(self._hyper_shapes_learned_ref, list)
                for ii in self._hyper_shapes_learned_ref:
                    assert isinstance(ii, int)
                    assert ii == -1 or 0 <= ii < len(self._param_shapes)

        assert(isinstance(self._has_fc_out, bool))
        assert(isinstance(self._mask_fc_out, bool))
        assert(isinstance(self._has_linear_out, bool))

        assert(self._layer_weight_tensors is not None)
        assert(self._layer_bias_vectors is not None)

        # Note, you should overwrite the `has_bias` attribute if you do not
        # follow this requirement.
        if check_has_bias:
            assert isinstance(self._has_bias, bool)
            if self._has_bias:
                assert len(self._layer_weight_tensors) == \
                       len(self._layer_bias_vectors)
    @property
    def bptt_depth(self):
        """Getter for attribute :attr:`bptt_depth`."""
        return self._bptt_depth

    @bptt_depth.setter
    def bptt_depth(self, value):
        """Setter for attribute :attr:`bptt_depth`."""
        self._bptt_depth = value

    @property
    def num_rec_layers(self):
        """Getter for read-only attribute :attr:`num_rec_layers`."""
        return len(self._rnn_layers)

    @property
    def use_lstm(self):
        """Getter for read-only attribute :attr:`use_lstm`."""
        return self._use_lstm
    def forward(self, x, weights=None, distilled_params=None, condition=None,
                return_hidden=False, return_hidden_int=False):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            weights (list or dict): See argument ``weights`` of method
                :meth:`mnets.mlp.MLP.forward`.
            condition (optional, int): If provided, then this argument will be
                passed as argument ``ckpt_id`` to the method
                :meth:`utils.context_mod_layer.ContextModLayer.forward`.
            return_hidden (bool, optional): If ``True``, all hidden activations
                of fully-connected and recurrent layers (where we defined
                :math:`y_t` as hidden state of vannila RNN layers as these are
                the layer outputs passed to the next layer) are returned.
                recurrent hidden activations will be returned.
            return_hidden_int (bool, optional): If ``True``, in addition to
                ``hidden``, an additional variable ``hidden_int`` is returned
                containing the internal hidden states of recurrent layers (i.e.,
                the cell states :math:`c_t` for LSTMs and the actual hidden
                state :math:`h_t` for Elman layers) are returned.

        Returns:
            (torch.Tensor or tuple): Where the tuple is containing:

            - **output** (torch.Tensor): The output of the network.
            - **hidden** (list): If ``return_hidden`` is ``True``, then the
              hidden activities of each layer are returned, which have the shape
              ``(seq_length, batch_size, n_hidden)``.
            - **hidden_int**: If ``return_hidden_int`` is ``True``, then in
              addition to ``hidden`` a tensor ``hidden_int`` is returned
              containing internal hidden states of recurrent layers.
        """
        assert distilled_params is None

        if ((not self._use_context_mod and self._no_weights) or \
                (self._no_weights or self._context_mod_no_weights)) and \
                weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        #######################
        ### Extract weights ###
        #######################
        # Extract which weights should be used.
        int_weights, cm_weights = self.split_weights(weights)

        ### Split context-mod weights per context-mod layer.
        cm_inputs_weights, cm_fc_pre_layer_weights, cm_fc_layer_weights, \
            cm_rec_layer_weights, n_cm_rec, cmod_cond = self.split_cm_weights(
                cm_weights, condition, num_ts=x.shape[0])

        ### Extract internal weights.
        fc_pre_w_weights, fc_pre_b_weights, rec_weights, fc_w_weights, \
            fc_b_weights = self.split_internal_weights(int_weights)

        ###########################
        ### Forward Computation ###
        ###########################
        ret_hidden = None
        if return_hidden:
            ret_hidden = []

        h = x

        cm_offset = 0
        if self._use_context_mod and self._context_mod_inputs:
            cm_offset += 1
            # Apply context modulation in the inputs.
            h = self._context_mod_layers[0].forward(h,
                weights=cm_inputs_weights, ckpt_id=cmod_cond, bs_dim=1)

        ### Initial fully-connected layer activities.
        ret_hidden, h = self.compute_fc_outputs(h, fc_pre_w_weights, \
            fc_pre_b_weights, len(self._fc_layers_pre), \
            cm_fc_pre_layer_weights, cm_offset, cmod_cond, False, ret_hidden)

        ### Recurrent layer activities.
        ret_hidden_int = [] # the internal hidden activations
        for d in range(len(self._rnn_layers)):
            if self._use_context_mod:
                h, h_int = self.compute_hidden_states(h, d, rec_weights[d],
                    cm_rec_layer_weights[d], cmod_cond)
            else:
                h, h_int = self.compute_hidden_states(h, d, rec_weights[d],
                                                      None, None)
            if ret_hidden is not None:
                ret_hidden.append(h)
                ret_hidden_int.append(h_int)

        ### Fully-connected layer activities.
        cm_offset = self._cm_rnn_start_ind + n_cm_rec
        ret_hidden, h = self.compute_fc_outputs(h, fc_w_weights, fc_b_weights, \
            self._num_fc_cm_layers, cm_fc_layer_weights, cm_offset, cmod_cond,
            True, ret_hidden)

        # FIXME quite ugly
        if return_hidden:
            # The last element is the output activity.
            ret_hidden.pop()
            if return_hidden_int:
                return h, ret_hidden, ret_hidden_int
            else:
                return h, ret_hidden
        else:
            return h
    def compute_fc_outputs(self, h, fc_w_weights, fc_b_weights, num_fc_cm_layers,
                cm_fc_layer_weights, cm_offset, cmod_cond, is_post_fc, 
                ret_hidden):
        """Compute the forward pass through the fully-connected layers.

        This method also appends activations to ``ret_hidden``.

        Args:
            h (torch.Tensor): The input from the previous layer.
            fc_w_weights (list): The weights for the fc layers.
            fc_b_weights (list): The biases for the fc layers.
            num_fc_cm_layers (int): The number of context-modulation
                layers associated with this set of fully-connected layers.
            cm_fc_layer_weights (list): The context-modulation weights
                associated with the current layers.
            cm_offset (int): The index to access the correct context-mod
                layers.
            cmod_cond (bool): Some condition to perform context modulation.
            is_post_fc (bool); Whether those layers are applied as last
                layers of the network. In this case, there will be no
                activation applied to the last layer outputs.
            ret_hidden (list or None): The hidden recurrent activations.

        Return:
            (Tuple): Tuple containing:

            - **ret_hidden**: The hidden recurrent activations.
            - **h**: Transformed activation ``h``.
        """
        for d in range(len(fc_w_weights)):
            use_cm = self._use_context_mod and d < num_fc_cm_layers
            # Compute output.
            h = F.linear(h, fc_w_weights[d], bias=fc_b_weights[d])

            # Context-dependent modulation (pre-activation).
            if use_cm and not self._context_mod_post_activation:
                h = self._context_mod_layers[cm_offset+d].forward(h,
                    weights=cm_fc_layer_weights[d], ckpt_id=cmod_cond, 
                    bs_dim=1)

            # Non-linearity
            # Note, non-linearity is not applied to outputs of the network.
            if self._a_fun is not None and \
                    (not is_post_fc or d < len(fc_w_weights)-1):
                h = self._a_fun(h)

            # Context-dependent modulation (post-activation).
            if use_cm and self._context_mod_post_activation:
                h = self._context_mod_layers[cm_offset+d].forward(h,
                    weights=cm_fc_layer_weights[d], ckpt_id=cmod_cond, 
                    bs_dim=1)

            if ret_hidden is not None:
                ret_hidden.append(h)

        return ret_hidden, h

    def compute_hidden_states(self, x, layer_ind, int_weights, cm_weights,
                              ckpt_id, h_0=None, c_0=None):
        """Compute the hidden states for the recurrent layer ``layer_ind`` from
        a sequence of inputs :math:`x`.

        If so specified, context modulation is applied before or after the
        nonlinearities.

        Args:
            x: The inputs :math:`x` to the layer. :math:`x` has shape
                ``[sequence_len, batch_size, n_hidden_prev]``.
            layer_ind (int): Index of the layer.
            int_weights: Internal weights associated with this recurrent layer.
            cm_weights: Context modulation weights.
            ckpt_id: Will be passed as option ``ckpt_id`` to method
                :meth:`utils.context_mod_layer.ContextModLayer.forward` if
                context-mod layers are used.
            h_0 (torch.Tensor, optional): The initial state for :math:`h`.
            c_0 (torch.Tensor, optional): The initial state for :math:`c`. Note
                that for LSTMs, if the initial state is to be defined, this
                variable is necessary also, not only :math:`h_0`, whereas for
                vanilla RNNs it is enough to provide :math:`h_0` as :math:`c_0`
                represents the output of the layer and it can be easily computed
                from `h_0`.

        Returns:
            (tuple): Tuple containing:

            - **outputs** (torch.Tensor): The sequence of visible hidden states
              given the input. It has shape 
              ``[sequence_len, batch_size, n_hidden]``.
            - **hiddens** (torch.Tensor): The sequence of hidden states given
              the input. For LSTMs, this corresponds to :math:`c`.
              It has shape ``[sequence_len, batch_size, n_hidden]``.
        """
        seq_length, batch_size, n_hidden_prev = x.shape
        n_hidden = self._rnn_layers[layer_ind]

        # Generate initial hidden states.
        # Note that for LSTMs h_0 is the hidden state and output vector whereas
        # c_0 is the internal cell state vector.
        # For a vanilla RNN h_0 is the hidden state whereas c_0 is the output
        # vector.
        if h_0 is None:
            h_0 = (torch.zeros(batch_size, n_hidden, device=x.device))
        if c_0 is None:
            c_0 = (torch.zeros(batch_size, n_hidden, device=x.device))
        assert h_0.shape[0] == c_0.shape[0] == batch_size
        assert h_0.shape[1] == c_0.shape[1] == n_hidden

        # If we want to apply context modulation in each time step, we need
        # to split the input sequence and call pytorch function at every
        # time step.
        outputs = []
        hiddens = []
        h_t = h_0
        c_t = c_0
        for t in range(seq_length):
            x_t = x[t,:,:]

            if cm_weights is not None and self._context_mod_num_ts != -1:
                curr_cm_weights = cm_weights[t]
            elif cm_weights is not None:
                assert len(cm_weights) == 1
                curr_cm_weights = cm_weights[0]
            else:
                curr_cm_weights = cm_weights

            # Compute the actual rnn step (either vanilla or LSTM, depending on
            # the flag self._use_lstm).
            is_last_step = t==(seq_length-1)
            h_t, c_t = self._rnn_fct(layer_ind, t, x_t, (h_t, c_t), int_weights,
                                     curr_cm_weights, ckpt_id, is_last_step)

            if self.bptt_depth != -1:
                if t < (seq_length - self.bptt_depth):
                    # Detach hidden/output states, such that we don't backprop
                    # through these timesteps.
                    h_t = h_t.detach()
                    c_t = c_t.detach()

            # FIXME Solution is a bit ugly. For an LSTM, the hidden state is
            # also the output whereas a normal RNN has a separate output.
            if self._use_lstm:
                outputs.append(h_t)
                hiddens.append(c_t)
            else:
                outputs.append(c_t)
                hiddens.append(h_t)

        return torch.stack(outputs), torch.stack(hiddens)


    def compute_basic_rnn_output(self, h_t, int_weights, use_cm, cm_weights,
                                 cm_idx, ckpt_id, is_last_step):
        """Compute the output of a vanilla RNN given the hidden state.

        Args:
            (...): See docstring of method :meth:`basic_rnn_step`.
            use_cm (boolean): Whether context modulation is being used.
            cm_idx (int): Index of the context-mod layer.

        Returns:
            (torch.tensor): The output.
        """
        if self.has_bias:
            weight_ho = int_weights[4]
            bias_ho = int_weights[5]
        else:
            weight_ho = int_weights[2]
            bias_ho = None

        y_t = h_t @ weight_ho.t()
        if self.has_bias:
            y_t += bias_ho

        # Context-dependent modulation (pre-activation).
        if use_cm and not self._context_mod_post_activation:
            if not self._context_mod_last_step or is_last_step:
                y_t = self._context_mod_layers[cm_idx].forward(y_t,
                    weights=cm_weights, ckpt_id=ckpt_id)

        # Compute activation.
        if self._a_fun is not None:
            y_t = self._a_fun(y_t)

        # Context-dependent modulation (post-activation).
        if use_cm and self._context_mod_post_activation:
            if not self._context_mod_last_step or is_last_step:
                y_t = self._context_mod_layers[cm_idx].forward(y_t,
                    weights=cm_weights, ckpt_id=ckpt_id)
        return y_t

    def lstm_rnn_step(self, d, t, x_t, h_t, int_weights, cm_weights, ckpt_id,
                      is_last_step):
        """ Perform an LSTM pass from inputs to hidden units.

        Apply masks to the temporal sequence for computing the loss.
        Obtained from:

            https://mlexplained.com/2019/02/15/building-an-lstm-from-scratch-\
in-pytorch-lstms-in-depth-part-1/

        and:

            https://d2l.ai/chapter_recurrent-neural-networks/lstm.html

        Args:
            d (int): Index of the layer.
            t (int): Current timestep.
            x_t: Tensor of size ``[batch_size, n_inputs]`` with inputs.
            h_t (tuple): Tuple of length 2, containing two tensors of size
                ``[batch_size, n_hidden]`` with previous hidden states ``h`` and
                ``c``.
            int_weights: See docstring of method :meth:`basic_rnn_step`.
            cm_weights: See docstring of method :meth:`basic_rnn_step`.
            ckpt_id: See docstring of method :meth:`basic_rnn_step`.
            is_last_step (bool): See docstring of method :meth:`basic_rnn_step`.

        Returns:
            (tuple): Tuple containing:

            - **h_t** (torch.Tensor): The tensor ``h_t`` of size
              ``[batch_size, n_hidden]`` with the new hidden state.
            - **c_t** (torch.Tensor): The tensor ``c_t`` of size
              ``[batch_size, n_hidden]`` with the new cell state.
        """
        use_cm = self._use_context_mod and d < self._num_rec_cm_layers
        # Determine the index of the hidden context mod layer.
        # Number of cm-layers per recurrent layer.
        n_cm_per_rec = self._context_mod_num_ts if \
            self._context_mod_num_ts != -1 and \
                self._context_mod_separate_layers_per_ts else 1
        cm_idx = self._cm_rnn_start_ind + d * n_cm_per_rec
        if self._context_mod_num_ts != -1 and \
                self._context_mod_separate_layers_per_ts:
            cm_idx += t

        c_t = h_t[1]
        h_t = h_t[0]
        HS = self._rnn_layers[d]

        if self._has_bias:
            assert len(int_weights) == 4
            weight_ih = int_weights[0]
            bias_ih = int_weights[1]
            weight_hh = int_weights[2]
            bias_hh = int_weights[3]
        else:
            assert len(int_weights) == 2
            weight_ih = int_weights[0]
            bias_ih = None
            weight_hh = int_weights[1]
            bias_hh = None

        # Compute total pre-activation input.
        gates = x_t @ weight_ih.t() + h_t @ weight_hh.t()
        if self.has_bias:
            gates += bias_ih + bias_hh

        i_t = gates[:, :HS]
        f_t = gates[:, HS:HS*2]
        g_t = gates[:, HS*2:HS*3]
        o_t = gates[:, HS*3:]

        # Compute activation.
        i_t = torch.sigmoid(i_t) # input
        f_t = torch.sigmoid(f_t) # forget
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t) # output

        # Compute c states.
        c_t = f_t * c_t + i_t * g_t

        # Note, we don't want to context-modulate the internal state c_t.
        # Otherwise, it might explode over timesteps since it wouldn't be
        # limited to [-1, 1] anymore. Instead, we only modulate the current
        # state (which is used to compute the current output h_t.
        c_t_mod = c_t

        # Context-dependent modulation (pre-activation).
        if use_cm and not self._context_mod_post_activation:
            if not self._context_mod_last_step or is_last_step:
                c_t_mod = self._context_mod_layers[cm_idx].forward(c_t_mod,
                    weights=cm_weights, ckpt_id=ckpt_id)

        # Compute h states.
        if self._a_fun is not None:
            h_t = o_t * self._a_fun(c_t_mod)
        else:
            h_t = o_t * c_t_mod

        # Context-dependent modulation (post-activation).
        # FIXME Christian: Shouldn't we apply the output gate `o_t` after
        # applying post-activation context-mod?
        if use_cm and self._context_mod_post_activation:
            if not self._context_mod_last_step or is_last_step:
                h_t = self._context_mod_layers[cm_idx].forward(h_t,
                    weights=cm_weights, ckpt_id=ckpt_id)

        return h_t, c_t
    def get_output_weight_mask(self, out_inds=None, device=None):
        """Get masks to select output weights.

        See docstring of overwritten super method
        :meth:`mnets.mnet_interface.MainNetInterface.get_output_weight_mask`.
        """
        if len(self._fc_layers) > 0:
            return MainNetInterface.get_output_weight_mask(self,
                out_inds=out_inds, device=device)

        # TODO Output layer is recurrent. Hence, we have to properly handle
        # which weights contribute solely to certain output activations.
        raise NotImplementedError()
    def get_cm_weights(self):
        """Get internal maintained weights that are associated with context-
        modulation.

        Returns:
            (list): List of weights from
            :attr:`mnets.mnet_interface.MainNetInterface.internal_params` that
            are belonging to context-mod layers.
        """
        n_cm = self._num_context_mod_shapes()

        if n_cm == 0 or self._context_mod_no_weights:
            raise ValueError('Network maintains no context-modulation weights.')

        return self.internal_params[:n_cm]

    def get_non_cm_weights(self):
        """Get internal weights that are not associated with context-modulation.

        Returns:
            (list): List of weights from
            :attr:`mnets.mnet_interface.MainNetInterface.internal_params` that
            are not belonging to context-mod layers.
        """
        n_cm = 0 if self._context_mod_no_weights else \
            self._num_context_mod_shapes()

        return self.internal_params[n_cm:]

    def get_cm_inds(self):
        """Get the indices of
        :attr:`mnets.mnet_interface.MainNetInterface.param_shapes` that are
        associated with context-modulation.

        Returns:
            (list): List of integers representing indices of
            :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.
        """
        n_cm = self._num_context_mod_shapes()

        ret = []

        for i, meta in enumerate(self.param_shapes_meta):
            if meta['name'] == 'cm_shift' or meta['name'] == 'cm_scale':
                ret.append(i)
        assert n_cm == len(ret)

        return ret

    def init_hh_weights_orthogonal(self):
        """Initialize hidden-to-hidden weights orthogonally.

        This method will overwrite the hidden-to-hidden weights of recurrent
        layers.
        """
        for meta in self.param_shapes_meta:
            if meta['name'] == 'weight' and 'info' in meta.keys() and \
                    meta['info'] == 'hh' and meta['index'] != -1:
                print('Initializing hidden-to-hidden weights of recurrent ' +
                      'layer %d orthogonally.' % meta['layer'])
                W = self.internal_params[meta['index']]
                # LSTM weight matrices are stored such that the hidden-to-hidden 
                # matrices for the 4 gates are concatenated.
                if self.use_lstm:
                    out_dim, _ = W.shape
                    assert out_dim % 4 == 0
                    fs = out_dim // 4

                    W1 = W[:fs, :]
                    W2 = W[fs:2*fs, :]
                    W3 = W[2*fs:3*fs, :]
                    W4 = W[3*fs:, :]

                    torch.nn.init.orthogonal_(W1.data)
                    torch.nn.init.orthogonal_(W2.data)
                    torch.nn.init.orthogonal_(W3.data)
                    torch.nn.init.orthogonal_(W4.data)

                    # Sanity check to show that the init on partial matrices
                    # propagates back to the original tensor.
                    assert W[0,0] == W1[0,0]
                else:
                    torch.nn.init.orthogonal_(W.data)

    def _internal_weight_shapes(self):
        """Compute the tensor shapes of all internal weights (i.e., those not
        associated with context-modulation).

        Returns:
            (list): A list of list of integers, denoting the shapes of the
            individual parameter tensors.
        """
        coeff = 4 if self._use_lstm else 1
        shapes = []

        # Initial fully-connected layers.
        prev_dim = self._n_in
        for n_fc in self._fc_layers_pre:
            shapes.append([n_fc, prev_dim])
            if self._use_bias:
                shapes.append([n_fc])

            prev_dim = n_fc

        # Recurrent layers.
        for n_rec in self._rnn_layers:
            # Input-to-hidden
            shapes.append([n_rec*coeff, prev_dim])
            if self._use_bias:
                shapes.append([n_rec*coeff])

            # Hidden-to-hidden
            shapes.append([n_rec*coeff, n_rec])
            if self._use_bias:
                shapes.append([n_rec*coeff])

            if not self._use_lstm:
                # Hidden-to-output
                shapes.append([n_rec, n_rec])
                if self._use_bias:
                    shapes.append([n_rec])

            prev_dim = n_rec

        # Fully-connected layers.
        for n_fc in self._fc_layers:
            shapes.append([n_fc, prev_dim])
            if self._use_bias:
                shapes.append([n_fc])

            prev_dim = n_fc

        return shapes

    @property
    def internal_params(self):
        """Getter for read-only attribute :attr:`internal_params`.

        Returns:
            A :class:`torch.nn.ParameterList` or ``None``, if no parameters are
            internally maintained.
        """
        return self._internal_params

    @property
    def weights(self):
        """Getter for read-only attribute :attr:`weights`.

        Note:
            Please use attribute :attr:`internal_params` instead.

        Returns:
            A :class:`torch.nn.ParameterList` or ``None``, if no parameters are
            internally maintained.
        """
        # Note, we can't deprecate "weights" just yet, as it is used all over
        # the repository (still in almost all main network implementations as
        # of April 2020).
        warn('Use attribute "internal_params" rather than "weigths", as ' +
             '"weights" might be depreacted in the future.',
             PendingDeprecationWarning)
        return self.internal_params

    @property
    def internal_params_ref(self):
        """Getter for read-only attribute :attr:`internal_params_ref`.

        Returns:
            (list): List of integers or ``None`` if no parameters are
            internally maintained.
        """
        if self.internal_params is None:
            return None

        if len(self.internal_params) == 0:
            return []

        # Note, programmers are not forced (just encouraged) to implement
        # `param_shapes_meta`.
        try:
            psm = self.param_shapes_meta
        except:
            raise NotImplementedError('Attribute "internal_params_ref" ' +
                'requires that attribute "param_shapes_meta" is implemented ' +
                'for this network.')

        ret_dict = {}

        for i, m in enumerate(psm):
            if m['index'] != -1:
                assert m['index'] not in ret_dict.keys()
                ret_dict[m['index']] = i

        assert np.all(np.isin(np.arange(len(self.internal_params)),
                              list(ret_dict.keys())))
        return np.sort(list(ret_dict.keys())).tolist()

    @property
    def param_shapes(self):
        """Getter for read-only attribute :attr:`param_shapes`.

        Returns:
            A list of lists of integers.
        """
        return self._param_shapes

    @property
    def param_shapes_meta(self):
        """Getter for read-only attribute :attr:`param_shapes_meta`.

        Returns:
            (list): A list of distionaries.
        """
        if self._param_shapes_meta is None:
            raise NotImplementedError('Attribute not implemented for this ' +
                                      'network.')

        return self._param_shapes_meta

    @property
    def hyper_shapes(self):
        """Getter for read-only attribute :attr:`hyper_shapes`.

        .. deprecated:: 1.0
            This attribute has been renamed to :attr:`hyper_shapes_learned`.

        Returns:
            A list of lists of integers.
        """
        warn('Use atrtibute "hyper_shapes_learned" instead.',
             DeprecationWarning)

        return self.hyper_shapes_learned

    @property
    def hyper_shapes_learned(self):
        """Getter for read-only attribute :attr:`hyper_shapes_learned`.

        Returns:
            A list of lists of integers.
        """
        return self._hyper_shapes_learned

    @property
    def hyper_shapes_learned_ref(self):
        """Getter for read-only attribute :attr:`hyper_shapes_learned_ref`.

        Returns:
            (list): A list of integers.
        """
        if self._hyper_shapes_learned is not None and \
                self._hyper_shapes_learned_ref is None:
            raise NotImplementedError('Attribute not implemented for this ' +
                                      'network')

        return self._hyper_shapes_learned_ref

    @property
    def hyper_shapes_distilled(self):
        """Getter for read-only attribute :attr:`hyper_shapes_distilled`.

        Returns:
            A list of lists of integers.
        """
        return self._hyper_shapes_distilled

    @property
    def has_bias(self):
        """Getter for read-only attribute :attr:`has_bias`."""
        return self._has_bias

    @property
    def has_fc_out(self):
        """Getter for read-only attribute :attr:`has_fc_out`."""
        return self._has_fc_out

    @property
    def mask_fc_out(self):
        """Getter for read-only attribute :attr:`mask_fc_out`."""
        return self._mask_fc_out

    @property
    def has_linear_out(self):
        """Getter for read-only attribute :attr:`has_linear_out`."""
        return self._has_linear_out

    @property
    def num_params(self):
        """Getter for read-only attribute :attr:`num_params`.

        Returns:
            (int): Total number of parameters in the network.
        """
        if self._num_params is None:
            self._num_params = MainNetInterface.shapes_to_num_weights( \
                self.param_shapes)
        return self._num_params

    @property
    def num_internal_params(self):
        """Getter for read-only attribute :attr:`num_internal_params`.

        Returns:
            (int): Total number of parameters currently maintained by this
            network instance.
        """
        if self._num_internal_params is None:
            if self.internal_params is None:
                self._num_internal_params = 0
            else:
                # FIXME should we distinguish between trainable and
                # non-trainable parameters (`p.requires_grad`)?
                self._num_internal_params = int(sum(p.numel() for p in \
                                                    self.internal_params))
        return self._num_internal_params

    @property
    def layer_weight_tensors(self):
        """Getter for read-only attribute :attr:`layer_weight_tensors`.

        Returns:
            A list (e.g., an instance of class :class:`torch.nn.ParameterList`).
        """
        return self._layer_weight_tensors

    @property
    def layer_bias_vectors(self):
        """Getter for read-only attribute :attr:`layer_bias_vectors`.

        Returns:
            A list (e.g., an instance of class :class:`torch.nn.ParameterList`).
        """
        return self._layer_bias_vectors

    @property
    def batchnorm_layers(self):
        """Getter for read-only attribute :attr:`batchnorm_layers`.

        Returns:
            (:class:`torch.nn.ModuleList`): A list of
            :class:`utils.batchnorm_layer.BatchNormLayer` instances, if batch
            normalization is used.
        """
        return self._batchnorm_layers

    @property
    def context_mod_layers(self):
        """Getter for read-only attribute :attr:`context_mod_layers`.

        Returns:
            (:class:`torch.nn.ModuleList`): A list of
            :class:`utils.context_mod_layer.ContextModLayer` instances, if these
            layers are in use.
        """
        return self._context_mod_layers

    def split_cm_weights(self, cm_weights, condition, num_ts=0):
        """Split context-mod weights per context-mod layer.

        Args:
            cm_weights (torch.Tensor): All context modulation weights.
            condition (optional, int): If provided, then this argument will be
                passed as argument ``ckpt_id`` to the method
                :meth:`utils.context_mod_layer.ContextModLayer.forward`.
            num_ts (int): The length of the sequences.

        Returns:
            (Tuple): Where the tuple contains:

            - **cm_inputs_weights**: The cm input weights.
            - **cm_fc_pre_layer_weights**: The cm pre-recurrent weights.
            - **cm_rec_layer_weights**: The cm recurrent weights.
            - **cm_fc_layer_weights**: The cm post-recurrent weights.
            - **n_cm_rec**: The number of recurrent cm layers.
            - **cmod_cond**: The context-mod condition.
        """

        n_cm_rec = -1
        cm_fc_pre_layer_weights = None
        cm_fc_layer_weights = None
        cm_inputs_weights = None
        cm_rec_layer_weights = None
        if cm_weights is not None:
            if self._context_mod_num_ts != -1 and \
                    self._context_mod_separate_layers_per_ts:
                assert num_ts <= self._context_mod_num_ts

            # Note, an mnet layer might contain multiple context-mod layers
            # (a recurrent layer can have a separate context-mod layer per
            # timestep).
            cm_fc_pre_layer_weights = []
            cm_rec_layer_weights = [[] for _ in range(self._num_rec_cm_layers)]
            cm_fc_layer_weights = []

            # Number of cm-layers per recurrent layer.
            n_cm_per_rec = self._context_mod_num_ts if \
                self._context_mod_num_ts != -1 and \
                    self._context_mod_separate_layers_per_ts else 1
            n_cm_rec = n_cm_per_rec * self._num_rec_cm_layers

            cm_start = 0
            for i, cm_layer in enumerate(self.context_mod_layers):
                cm_end = cm_start + len(cm_layer.param_shapes)

                if i == 0 and self._context_mod_inputs:
                    cm_inputs_weights = cm_weights[cm_start:cm_end]
                elif i < self._cm_rnn_start_ind:
                    cm_fc_pre_layer_weights.append(cm_weights[cm_start:cm_end])
                elif i >= self._cm_rnn_start_ind and \
                        i < self._cm_rnn_start_ind + n_cm_rec:
                    # Index of recurrent layer.
                    i_r = (i-self._cm_rnn_start_ind) // n_cm_per_rec
                    cm_rec_layer_weights[i_r].append( \
                        cm_weights[cm_start:cm_end])
                else:
                    cm_fc_layer_weights.append(cm_weights[cm_start:cm_end])
                cm_start = cm_end

            # We need to split the context-mod weights in the following case,
            # as they are currently just stacked on top of each other.
            if self._context_mod_num_ts != -1 and \
                    not self._context_mod_separate_layers_per_ts:
                for i, cm_w_list in enumerate(cm_rec_layer_weights):
                    assert len(cm_w_list) == 1

                    cm_rnn_weights = cm_w_list[0]
                    cm_rnn_layer = self.context_mod_layers[ \
                        self._cm_rnn_start_ind+i]

                    assert len(cm_rnn_weights) == len(cm_rnn_layer.param_shapes)
                    # The first dimension are the weights of this layer per
                    # timestep.
                    num_ts_cm = -1
                    for j, s in enumerate(cm_rnn_layer.param_shapes):
                        assert len(cm_rnn_weights[j].shape) == len(s) + 1
                        if j == 0:
                            num_ts_cm = cm_rnn_weights[j].shape[0]
                        else:
                            assert num_ts_cm == cm_rnn_weights[j].shape[0]
                    assert num_ts <= num_ts_cm

                    cm_w_chunked = [None] * len(cm_rnn_weights)
                    for j, cm_w in enumerate(cm_rnn_weights):
                        cm_w_chunked[j] = torch.chunk(cm_w, num_ts_cm, dim=0)

                    # Now we gather all these chunks to assemble the weights
                    # needed per timestep (as if
                    # `_context_mod_separate_layers_per_t` were True).
                    cm_w_list = []
                    for j in range(num_ts_cm):
                        tmp_list = []
                        for chunk in cm_w_chunked:
                            tmp_list.append(chunk[j].squeeze(dim=0))
                        cm_w_list.append(tmp_list)
                    cm_rec_layer_weights[i] = cm_w_list

            # Note, the last layer does not necessarily have context-mod
            # (depending on `self._context_mod_outputs`).
            if len(cm_rec_layer_weights) < len(self._rnn_layers):
                cm_rec_layer_weights.append(None)
            if len(cm_fc_layer_weights) < len(self._fc_layers):
                cm_fc_layer_weights.append(None)


        #######################
        ### Parse condition ###
        #######################
        cmod_cond = None
        if condition is not None:
            assert isinstance(condition, int)
            cmod_cond = condition

            # Note, the cm layer will ignore the cmod condition if weights
            # are passed.
            # FIXME Find a more elegant solution.
            cm_inputs_weights = None
            cm_fc_pre_layer_weights = [None] * len(cm_fc_pre_layer_weights)
            cm_rec_layer_weights = [[None] * len(cm_ws) for cm_ws in \
                                    cm_rec_layer_weights]
            cm_fc_layer_weights = [None] * len(cm_fc_layer_weights)

        return cm_inputs_weights, cm_fc_pre_layer_weights, cm_fc_layer_weights,\
            cm_rec_layer_weights, n_cm_rec, cmod_cond

    def split_internal_weights(self, int_weights):
        """Split internal weights per layer.

        Args:
            int_weights (torch.Tensor): All internal weights.

        Returns:
            (Tuple): Where the tuple contains:

            - **fc_pre_w_weights**: The pre-recurrent w weights.
            - **fc_pre_b_weights**: The pre-recurrent b weights.
            - **rec_weights**: The recurrent weights.
            - **fc_w_weights**:The post-recurrent w weights.
            - **fc_b_weights**: The post-recurrent b weights.
        """
        n_cm = self._num_context_mod_shapes()

        int_meta = self.param_shapes_meta[n_cm:]
        assert len(int_meta) == len(int_weights)
        fc_pre_w_weights = []
        fc_pre_b_weights = []
        rec_weights =[[] for _ in range(len(self._rnn_layers))]
        fc_w_weights = []
        fc_b_weights = []

        # Number of pre-fc weights in total.
        n_fc_pre = len(self._fc_layers_pre)
        if self.has_bias:
            n_fc_pre *= 2

        # Number of weights per recurrent layer.
        if self._use_lstm:
            n_rw = 4 if self.has_bias else 2
        else:
            n_rw = 6 if self.has_bias else 3

        for i, w in enumerate(int_weights):
            if i < n_fc_pre: # fc pre weights
                if int_meta[i]['name'] == 'weight':
                    fc_pre_w_weights.append(w)
                else:
                    assert int_meta[i]['name'] == 'bias'
                    fc_pre_b_weights.append(w)
            elif i >= n_fc_pre and \
                    i < n_rw * len(self._rnn_layers) + n_fc_pre: # recurrent w
                r_ind = (i - n_fc_pre) // n_rw
                rec_weights[r_ind].append(w)
            else: # fc weights
                if int_meta[i]['name'] == 'weight':
                    fc_w_weights.append(w)
                else:
                    assert int_meta[i]['name'] == 'bias'
                    fc_b_weights.append(w)

        if not self.has_bias:
            assert len(fc_pre_b_weights) == 0
            fc_pre_b_weights = [None] * len(fc_pre_w_weights)

            assert len(fc_b_weights) == 0
            fc_b_weights = [None] * len(fc_w_weights)

        return fc_pre_w_weights, fc_pre_b_weights, rec_weights, fc_w_weights, \
            fc_b_weights

    def split_weights(self, weights):
        """Split weights into internal and context-mod weights.

        Extract which weights should be used,  I.e., are we using internally
        maintained weights or externally given ones or are we even mixing
        between these groups.

        Args:
            weights (torch.Tensor): All weights.

        Returns:
            (Tuple): Where the tuple contains:

            - **int_weights**: The internal weights.
            - **cm_weights**: The context-mod weights.
        """
        n_cm = self._num_context_mod_shapes()

        ### FIXME Code copied from MLP its `forward` method ###

        # Make sure cm_weights are either `None` or have the correct dimensions.
        if weights is None:
            weights = self.weights

            if self._use_context_mod:
                cm_weights = weights[:n_cm]
                int_weights = weights[n_cm:]
            else:
                cm_weights = None
                int_weights = weights
        else:
            int_weights = None
            cm_weights = None

            if isinstance(weights, dict):
                assert 'internal_weights' in weights.keys() or \
                       'mod_weights' in weights.keys()
                if 'internal_weights' in weights.keys():
                    int_weights = weights['internal_weights']
                if 'mod_weights' in weights.keys():
                    cm_weights = weights['mod_weights']
            else:
                if self._use_context_mod and \
                        len(weights) == n_cm:
                    cm_weights = weights
                else:
                    assert len(weights) == len(self.param_shapes)
                    if self._use_context_mod:
                        cm_weights = weights[:n_cm]
                        int_weights = weights[n_cm:]
                    else:
                        int_weights = weights

            if self._use_context_mod and cm_weights is None:
                if self._context_mod_no_weights:
                    raise Exception('Network was generated without weights ' +
                        'for context-mod layers. Hence, they must be passed ' +
                        'via the "weights" option.')
                cm_weights = self.weights[:n_cm]
            if int_weights is None:
                if self._no_weights:
                    raise Exception('Network was generated without internal ' +
                        'weights. Hence, they must be passed via the ' +
                        '"weights" option.')
                if self._context_mod_no_weights:
                    int_weights = self.weights
                else:
                    int_weights = self.weights[n_cm:]

            # Note, context-mod weights might have different shapes, as they
            # may be parametrized on a per-sample basis.
            if self._use_context_mod:
                assert len(cm_weights) == n_cm
            int_shapes = self.param_shapes[n_cm:]
            assert len(int_weights) == len(int_shapes)
            for i, s in enumerate(int_shapes):
                assert np.all(np.equal(s, list(int_weights[i].shape)))

        ### FIXME Code copied until here ###
        return int_weights, cm_weights

    def forward(self, x, weights=None, distilled_params=None, condition=None,
                return_hidden=False, return_hidden_int=False):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            weights (list or dict): See argument ``weights`` of method
                :meth:`mnets.mlp.MLP.forward`.
            condition (optional, int): If provided, then this argument will be
                passed as argument ``ckpt_id`` to the method
                :meth:`utils.context_mod_layer.ContextModLayer.forward`.
            return_hidden (bool, optional): If ``True``, all hidden activations
                of fully-connected and recurrent layers (where we defined
                :math:`y_t` as hidden state of vannila RNN layers as these are
                the layer outputs passed to the next layer) are returned.
                recurrent hidden activations will be returned.
            return_hidden_int (bool, optional): If ``True``, in addition to
                ``hidden``, an additional variable ``hidden_int`` is returned
                containing the internal hidden states of recurrent layers (i.e.,
                the cell states :math:`c_t` for LSTMs and the actual hidden
                state :math:`h_t` for Elman layers) are returned.

        Returns:
            (torch.Tensor or tuple): Where the tuple is containing:

            - **output** (torch.Tensor): The output of the network.
            - **hidden** (list): If ``return_hidden`` is ``True``, then the
              hidden activities of each layer are returned, which have the shape
              ``(seq_length, batch_size, n_hidden)``.
            - **hidden_int**: If ``return_hidden_int`` is ``True``, then in
              addition to ``hidden`` a tensor ``hidden_int`` is returned
              containing internal hidden states of recurrent layers.
        """
        assert distilled_params is None

        if ((not self._use_context_mod and self._no_weights) or \
                (self._no_weights or self._context_mod_no_weights)) and \
                weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        #######################
        ### Extract weights ###
        #######################
        # Extract which weights should be used.
        int_weights, cm_weights = self.split_weights(weights)

        ### Split context-mod weights per context-mod layer.
        cm_inputs_weights, cm_fc_pre_layer_weights, cm_fc_layer_weights, \
            cm_rec_layer_weights, n_cm_rec, cmod_cond = self.split_cm_weights(
                cm_weights, condition, num_ts=x.shape[0])

        ### Extract internal weights.
        fc_pre_w_weights, fc_pre_b_weights, rec_weights, fc_w_weights, \
            fc_b_weights = self.split_internal_weights(int_weights)

        ###########################
        ### Forward Computation ###
        ###########################
        ret_hidden = None
        if return_hidden:
            ret_hidden = []

        h = x

        cm_offset = 0
        if self._use_context_mod and self._context_mod_inputs:
            cm_offset += 1
            # Apply context modulation in the inputs.
            h = self._context_mod_layers[0].forward(h,
                weights=cm_inputs_weights, ckpt_id=cmod_cond, bs_dim=1)

        ### Initial fully-connected layer activities.
        ret_hidden, h = self.compute_fc_outputs(h, fc_pre_w_weights, \
            fc_pre_b_weights, len(self._fc_layers_pre), \
            cm_fc_pre_layer_weights, cm_offset, cmod_cond, False, ret_hidden)

        ### Recurrent layer activities.
        ret_hidden_int = [] # the internal hidden activations
        for d in range(len(self._rnn_layers)):
            if self._use_context_mod:
                h, h_int = self.compute_hidden_states(h, d, rec_weights[d],
                    cm_rec_layer_weights[d], cmod_cond)
            else:
                h, h_int = self.compute_hidden_states(h, d, rec_weights[d],
                                                      None, None)
            if ret_hidden is not None:
                ret_hidden.append(h)
                ret_hidden_int.append(h_int)

        ### Fully-connected layer activities.
        cm_offset = self._cm_rnn_start_ind + n_cm_rec
        ret_hidden, h = self.compute_fc_outputs(h, fc_w_weights, fc_b_weights, \
            self._num_fc_cm_layers, cm_fc_layer_weights, cm_offset, cmod_cond,
            True, ret_hidden)

        # FIXME quite ugly
        if return_hidden:
            # The last element is the output activity.
            ret_hidden.pop()
            if return_hidden_int:
                return h, ret_hidden, ret_hidden_int
            else:
                return h, ret_hidden
        else:
            return h

    def compute_fc_outputs(self, h, fc_w_weights, fc_b_weights, num_fc_cm_layers,
                cm_fc_layer_weights, cm_offset, cmod_cond, is_post_fc, 
                ret_hidden):
        """Compute the forward pass through the fully-connected layers.

        This method also appends activations to ``ret_hidden``.

        Args:
            h (torch.Tensor): The input from the previous layer.
            fc_w_weights (list): The weights for the fc layers.
            fc_b_weights (list): The biases for the fc layers.
            num_fc_cm_layers (int): The number of context-modulation
                layers associated with this set of fully-connected layers.
            cm_fc_layer_weights (list): The context-modulation weights
                associated with the current layers.
            cm_offset (int): The index to access the correct context-mod
                layers.
            cmod_cond (bool): Some condition to perform context modulation.
            is_post_fc (bool); Whether those layers are applied as last
                layers of the network. In this case, there will be no
                activation applied to the last layer outputs.
            ret_hidden (list or None): The hidden recurrent activations.

        Return:
            (Tuple): Tuple containing:

            - **ret_hidden**: The hidden recurrent activations.
            - **h**: Transformed activation ``h``.
        """
        for d in range(len(fc_w_weights)):
            use_cm = self._use_context_mod and d < num_fc_cm_layers
            # Compute output.
            h = F.linear(h, fc_w_weights[d], bias=fc_b_weights[d])

            # Context-dependent modulation (pre-activation).
            if use_cm and not self._context_mod_post_activation:
                h = self._context_mod_layers[cm_offset+d].forward(h,
                    weights=cm_fc_layer_weights[d], ckpt_id=cmod_cond, 
                    bs_dim=1)

            # Non-linearity
            # Note, non-linearity is not applied to outputs of the network.
            if self._a_fun is not None and \
                    (not is_post_fc or d < len(fc_w_weights)-1):
                h = self._a_fun(h)

            # Context-dependent modulation (post-activation).
            if use_cm and self._context_mod_post_activation:
                h = self._context_mod_layers[cm_offset+d].forward(h,
                    weights=cm_fc_layer_weights[d], ckpt_id=cmod_cond, 
                    bs_dim=1)

            if ret_hidden is not None:
                ret_hidden.append(h)

        return ret_hidden, h

    def basic_rnn_step(self, d, t, x_t, h_t, int_weights, cm_weights, ckpt_id,
                       is_last_step):
        """Perform vanilla rnn pass from inputs to hidden units.

        Apply context modulation if necessary (i.e. if ``cm_weights`` is
        not ``None``).

        This function implements a step of an
        `Elman RNN <https://en.wikipedia.org/wiki/\
Recurrent_neural_network#Elman_networks_and_Jordan_networks>`__.

        Note:
            We made the following design choice regarding context-modulation.
            In contrast to the LSTM, the Elman network layer consists of "two
            steps", updating the hidden state and computing an output based
            on this hidden state. To be fair, context-mod should influence both
            these "layers". Therefore, we apply context-mod twice, but using the
            same weights. This of course assumes that the hidden state and
            output vector have the same dimensionality.

        Args:
            d (int): Index of the layer.
            t (int): Current timestep.
            x_t: Tensor of size ``[batch_size, n_hidden_prev]`` with inputs.
            h_t (tuple): Tuple of length 2, containing two tensors of size
                ``[batch_size, n_hidden]`` with previous hidden states ``h`` and
                and previous outputs ``y``.

                Note:
                    The previous outputs ``y`` are ignored by this method, since
                    they are not required in an Elman RNN step.
            int_weights: See docstring of method :meth:`compute_hidden_states`.
            cm_weights (list): The weights of the context-mod layer, if context-
                mod should be applied.
            ckpt_id: See docstring of method :meth:`compute_hidden_states`.
            is_last_step (bool): Whether the current time step is the last one.

        Returns:
            (tuple): Tuple containing:

            - **h_t** (torch.Tensor): The tensor ``h_t`` of size
              ``[batch_size, n_hidden]`` with the new hidden state.
            - **y_t** (torch.Tensor): The tensor ``y_t`` of size
              ``[batch_size, n_hidden]`` with the new cell state.
        """
        h_t = h_t[0]

        use_cm = self._use_context_mod and d < self._num_rec_cm_layers
        # Determine the index of the hidden context mod layer.
        # Number of cm-layers per recurrent layer.
        n_cm_per_rec = self._context_mod_num_ts if \
            self._context_mod_num_ts != -1 and \
                self._context_mod_separate_layers_per_ts else 1
        cm_idx = self._cm_rnn_start_ind + d * n_cm_per_rec
        if self._context_mod_num_ts != -1 and \
                self._context_mod_separate_layers_per_ts:
            cm_idx += t
 
        if self.has_bias:
            assert len(int_weights) == 6
            weight_ih = int_weights[0]
            bias_ih = int_weights[1]
            weight_hh = int_weights[2]
            bias_hh = int_weights[3]
        else:
            assert len(int_weights) == 3
            weight_ih = int_weights[0]
            bias_ih = None
            weight_hh = int_weights[1]
            bias_hh = None

        ###########################
        ### Update hidden state ###
        ###########################
        h_t = x_t @ weight_ih.t() + h_t @ weight_hh.t()
        if self.has_bias:
            h_t += bias_ih + bias_hh

        # Context-dependent modulation (pre-activation).
        if use_cm and not self._context_mod_post_activation:
            # Only apply context mod if you are in the last time step, or if
            # you want to apply it in every single time step (i.e. if
            # self._context_mod_last_step is False).
            if not self._context_mod_last_step or is_last_step:
                h_t = self._context_mod_layers[cm_idx].forward(h_t,
                    weights=cm_weights, ckpt_id=ckpt_id)

        # Compute activation.
        if self._a_fun is not None:
            h_t = self._a_fun(h_t)

        # Context-dependent modulation (post-activation).
        if use_cm and self._context_mod_post_activation:
            if not self._context_mod_last_step or is_last_step:
                h_t = self._context_mod_layers[cm_idx].forward(h_t,
                    weights=cm_weights, ckpt_id=ckpt_id)

        ######################
        ### Compute output ###
        ######################
        y_t = self.compute_basic_rnn_output(h_t, int_weights, use_cm, 
            cm_weights, cm_idx, ckpt_id, is_last_step)

        return h_t, y_t

    # def compute_basic_rnn_output(self, h_t, int_weights, use_cm, cm_weights,
    #                              cm_idx, ckpt_id, is_last_step):
    #     """Compute the output of a vanilla RNN given the hidden state.

    #     Args:
    #         (...): See docstring of method :meth:`basic_rnn_step`.
    #         use_cm (boolean): Whether context modulation is being used.
    #         cm_idx (int): Index of the context-mod layer.

    #     Returns:
    #         (torch.tensor): The output.
    #     """
    #     if self.has_bias:
    #         weight_ho = int_weights[4]
    #         bias_ho = int_weights[5]
    #     else:
    #         weight_ho = int_weights[2]
    #         bias_ho = None

    #     y_t = h_t @ weight_ho.t()
    #     if self.has_bias:
    #         y_t += bias_ho

    #     # Context-dependent modulation (pre-activation).
    #     if use_cm and not self._context_mod_post_activation:
    #         if not self._context_mod_last_step or is_last_step:
    #             y_t = self._context_mod_layers[cm_idx].forward(y_t,
    #                 weights=cm_weights, ckpt_id=ckpt_id)

    #     # Compute activation.
    #     if self._a_fun is not None:
    #         y_t = self._a_fun(y_t)

    #     # Context-dependent modulation (post-activation).
    #     if use_cm and self._context_mod_post_activation:
    #         if not self._context_mod_last_step or is_last_step:
    #             y_t = self._context_mod_layers[cm_idx].forward(y_t,
    #                 weights=cm_weights, ckpt_id=ckpt_id)
    #     return y_t
    def _num_context_mod_shapes(self):
        """The number of entries in :attr:`param_shapes` associated with
        context-modulation.

        Returns:
            (int): Returns ``0`` if :attr:`context_mod_layers` is ``None``.
        """
        if self.context_mod_layers is None:
            return 0

        ret = 0
        for cm_layer in self.context_mod_layers:
            ret += len(cm_layer.param_shapes)

        return ret
    def lstm_rnn_step(self, d, t, x_t, h_t, int_weights, cm_weights, ckpt_id,
                      is_last_step):
        """ Perform an LSTM pass from inputs to hidden units.

        Apply masks to the temporal sequence for computing the loss.
        Obtained from:

            https://mlexplained.com/2019/02/15/building-an-lstm-from-scratch-\
in-pytorch-lstms-in-depth-part-1/

        and:

            https://d2l.ai/chapter_recurrent-neural-networks/lstm.html

        Args:
            d (int): Index of the layer.
            t (int): Current timestep.
            x_t: Tensor of size ``[batch_size, n_inputs]`` with inputs.
            h_t (tuple): Tuple of length 2, containing two tensors of size
                ``[batch_size, n_hidden]`` with previous hidden states ``h`` and
                ``c``.
            int_weights: See docstring of method :meth:`basic_rnn_step`.
            cm_weights: See docstring of method :meth:`basic_rnn_step`.
            ckpt_id: See docstring of method :meth:`basic_rnn_step`.
            is_last_step (bool): See docstring of method :meth:`basic_rnn_step`.

        Returns:
            (tuple): Tuple containing:

            - **h_t** (torch.Tensor): The tensor ``h_t`` of size
              ``[batch_size, n_hidden]`` with the new hidden state.
            - **c_t** (torch.Tensor): The tensor ``c_t`` of size
              ``[batch_size, n_hidden]`` with the new cell state.
        """
        use_cm = self._use_context_mod and d < self._num_rec_cm_layers
        # Determine the index of the hidden context mod layer.
        # Number of cm-layers per recurrent layer.
        n_cm_per_rec = self._context_mod_num_ts if \
            self._context_mod_num_ts != -1 and \
                self._context_mod_separate_layers_per_ts else 1
        cm_idx = self._cm_rnn_start_ind + d * n_cm_per_rec
        if self._context_mod_num_ts != -1 and \
                self._context_mod_separate_layers_per_ts:
            cm_idx += t

        c_t = h_t[1]
        h_t = h_t[0]
        HS = self._rnn_layers[d]

        if self._has_bias:
            assert len(int_weights) == 4
            weight_ih = int_weights[0]
            bias_ih = int_weights[1]
            weight_hh = int_weights[2]
            bias_hh = int_weights[3]
        else:
            assert len(int_weights) == 2
            weight_ih = int_weights[0]
            bias_ih = None
            weight_hh = int_weights[1]
            bias_hh = None

        # Compute total pre-activation input.
        gates = x_t @ weight_ih.t() + h_t @ weight_hh.t()
        if self.has_bias:
            gates += bias_ih + bias_hh

        i_t = gates[:, :HS]
        f_t = gates[:, HS:HS*2]
        g_t = gates[:, HS*2:HS*3]
        o_t = gates[:, HS*3:]

        # Compute activation.
        i_t = torch.sigmoid(i_t) # input
        f_t = torch.sigmoid(f_t) # forget
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t) # output

        # Compute c states.
        c_t = f_t * c_t + i_t * g_t

        # Note, we don't want to context-modulate the internal state c_t.
        # Otherwise, it might explode over timesteps since it wouldn't be
        # limited to [-1, 1] anymore. Instead, we only modulate the current
        # state (which is used to compute the current output h_t.
        c_t_mod = c_t

        # Context-dependent modulation (pre-activation).
        if use_cm and not self._context_mod_post_activation:
            if not self._context_mod_last_step or is_last_step:
                c_t_mod = self._context_mod_layers[cm_idx].forward(c_t_mod,
                    weights=cm_weights, ckpt_id=ckpt_id)

        # Compute h states.
        if self._a_fun is not None:
            h_t = o_t * self._a_fun(c_t_mod)
        else:
            h_t = o_t * c_t_mod

        # Context-dependent modulation (post-activation).
        # FIXME Christian: Shouldn't we apply the output gate `o_t` after
        # applying post-activation context-mod?
        if use_cm and self._context_mod_post_activation:
            if not self._context_mod_last_step or is_last_step:
                h_t = self._context_mod_layers[cm_idx].forward(h_t,
                    weights=cm_weights, ckpt_id=ckpt_id)

        return h_t, c_t

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This network does not have any distillation targets.

        Returns:
            ``None``
        """
        return None

    def get_output_weight_mask(self, out_inds=None, device=None):
        """Create a mask for selecting weights connected solely to certain
        output units.

        This method will return a list of the same length as
        :attr:`param_shapes`. Entries in this list are either ``None`` or
        masks for the corresponding parameter tensors. For all parameter
        tensors that are not directly connected to output units, the
        corresponding entry will be ``None``. If ``out_inds is None``, then all
        output weights are selected by a masking value ``1``. Otherwise, only
        the weights connected to the output units in ``out_inds`` are selected,
        the rest is masked out.

        Note:
            This method only works for networks with a fully-connected output
            layer (see :attr:`has_fc_out`), that have the attribute
            :attr:`mask_fc_out` set. Otherwise, the method has to be overwritten
            by an implementing class.

        Args:
            out_inds (list, optional): List of integers. Each entry denotes an
                output unit.
            device: Pytorch device. If given, the created masks will be moved
                onto this device.

        Returns:
            (list): List of masks with the same length as :attr:`param_shapes`.
            Entries whose corresponding parameter tensors are not connected to
            the network outputs are ``None``.
        """
        if not (self.has_fc_out and self.mask_fc_out):
            raise NotImplementedError('Method not applicable for this ' +
                                      'network type.')

        ret = [None] * len(self.param_shapes)

        obias_ind = len(self.param_shapes)-1 if self.has_bias else None
        oweights_ind = len(self.param_shapes)-2 if self.has_bias \
            else len(self.param_shapes)-1

        # Bias weights for outputs.
        if obias_ind is not None:
            if out_inds is None:
                mask = torch.ones(*self.param_shapes[obias_ind],
                                  dtype=torch.bool)
            else:
                mask = torch.zeros(*self.param_shapes[obias_ind],
                                   dtype=torch.bool)
                mask[out_inds] = 1
            if device is not None:
                mask = mask.to(device)
            ret[obias_ind] = mask

        # Weights from weight matrix of output layer.
        if out_inds is None:
            mask = torch.ones(*self.param_shapes[oweights_ind],
                              dtype=torch.bool)
        else:
            mask = torch.zeros(*self.param_shapes[oweights_ind],
                               dtype=torch.bool)
            mask[out_inds, :] = 1
        if device is not None:
            mask = mask.to(device)
        ret[oweights_ind] = mask

        return ret

    def get_cm_weights(self):
        """Get internal maintained weights that are associated with context-
        modulation.

        Returns:
            (list): List of weights from
            :attr:`mnets.mnet_interface.MainNetInterface.internal_params` that
            are belonging to context-mod layers.
        """
        n_cm = self._num_context_mod_shapes()

        if n_cm == 0 or self._context_mod_no_weights:
            raise ValueError('Network maintains no context-modulation weights.')

        return self.internal_params[:n_cm]

    def get_non_cm_weights(self):
        """Get internal weights that are not associated with context-modulation.

        Returns:
            (list): List of weights from
            :attr:`mnets.mnet_interface.MainNetInterface.internal_params` that
            are not belonging to context-mod layers.
        """
        n_cm = 0 if self._context_mod_no_weights else \
            self._num_context_mod_shapes()

        return self.internal_params[n_cm:]

    def get_cm_inds(self):
        """Get the indices of
        :attr:`mnets.mnet_interface.MainNetInterface.param_shapes` that are
        associated with context-modulation.

        Returns:
            (list): List of integers representing indices of
            :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.
        """
        n_cm = self._num_context_mod_shapes()

        ret = []

        for i, meta in enumerate(self.param_shapes_meta):
            if meta['name'] == 'cm_shift' or meta['name'] == 'cm_scale':
                ret.append(i)
        assert n_cm == len(ret)

        return ret

    def init_hh_weights_orthogonal(self):
        """Initialize hidden-to-hidden weights orthogonally.

        This method will overwrite the hidden-to-hidden weights of recurrent
        layers.
        """
        for meta in self.param_shapes_meta:
            if meta['name'] == 'weight' and 'info' in meta.keys() and \
                    meta['info'] == 'hh' and meta['index'] != -1:
                print('Initializing hidden-to-hidden weights of recurrent ' +
                      'layer %d orthogonally.' % meta['layer'])
                W = self.internal_params[meta['index']]
                # LSTM weight matrices are stored such that the hidden-to-hidden 
                # matrices for the 4 gates are concatenated.
                if self.use_lstm:
                    out_dim, _ = W.shape
                    assert out_dim % 4 == 0
                    fs = out_dim // 4

                    W1 = W[:fs, :]
                    W2 = W[fs:2*fs, :]
                    W3 = W[2*fs:3*fs, :]
                    W4 = W[3*fs:, :]

                    torch.nn.init.orthogonal_(W1.data)
                    torch.nn.init.orthogonal_(W2.data)
                    torch.nn.init.orthogonal_(W3.data)
                    torch.nn.init.orthogonal_(W4.data)

                    # Sanity check to show that the init on partial matrices
                    # propagates back to the original tensor.
                    assert W[0,0] == W1[0,0]
                else:
                    torch.nn.init.orthogonal_(W.data)

    def _internal_weight_shapes(self):
        """Compute the tensor shapes of all internal weights (i.e., those not
        associated with context-modulation).

        Returns:
            (list): A list of list of integers, denoting the shapes of the
            individual parameter tensors.
        """
        coeff = 4 if self._use_lstm else 1
        shapes = []

        # Initial fully-connected layers.
        prev_dim = self._n_in
        for n_fc in self._fc_layers_pre:
            shapes.append([n_fc, prev_dim])
            if self._use_bias:
                shapes.append([n_fc])

            prev_dim = n_fc

        # Recurrent layers.
        for n_rec in self._rnn_layers:
            # Input-to-hidden
            shapes.append([n_rec*coeff, prev_dim])
            if self._use_bias:
                shapes.append([n_rec*coeff])

            # Hidden-to-hidden
            shapes.append([n_rec*coeff, n_rec])
            if self._use_bias:
                shapes.append([n_rec*coeff])

            if not self._use_lstm:
                # Hidden-to-output
                shapes.append([n_rec, n_rec])
                if self._use_bias:
                    shapes.append([n_rec])

            prev_dim = n_rec

        # Fully-connected layers.
        for n_fc in self._fc_layers:
            shapes.append([n_fc, prev_dim])
            if self._use_bias:
                shapes.append([n_fc])

            prev_dim = n_fc

        return shapes

if __name__ == '__main__':
    pass

    # # OUT_FEATURES updated to NUMCLASS
    # def Incremental_learning(self, numclass,device):
    #     #self.numclass = numclass
    #     if self.decoder is None:
    #         in_feature = self.token_num * self.token_num
    #         self.decoder = nn.Linear(in_feature, numclass, bias=True)
    #         self.decoder.to(device)
    #         self._init_weights()
    #     else:
    #         weight = self.decoder.weight.data
    #         bias = self.decoder.bias.data
    #         in_feature = self.decoder.in_features
    #         out_feature = self.decoder.out_features
    #         del self.decoder 
    #         self.decoder = nn.Linear(in_feature, numclass, bias=True)
    #         self.decoder.weight.data[:out_feature] = weight
    #         self.decoder.bias.data[:out_feature] = bias
    #         self.decoder.to(device)
    #     self.features_dim = self.decoder.in_features
    #     print("features_dim : ", self.decoder.in_features)

        
    