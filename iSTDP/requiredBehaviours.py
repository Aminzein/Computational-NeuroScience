import torch
import pandas as pd
import plotly.express as px
import copy
from pymonntorch import Behavior, SynapseGroup, Network, NeuronGroup, Recorder, EventRecorder

SRC_CONFIG = {
    "v_reset" : -65.0,
    "v_rest": -65.0,
    "tau" : 10.,
    "R" : 2.,
    "threshold" : -55.,
}
INH_CONFIG = {
    "v_reset" : -65.0,
    "v_rest": -65.0,
    "tau" : 10.,
    "R" : 2.,
    "threshold" : -55.,
}
DST_CONFIG = {
    "v_reset" : -65.0,
    "v_rest": -65.0,
    "tau" : 10.,
    "R" : 2.,
    "threshold" : -55.,
}
EXCITATORY_NEURON_SIZE = 100
INHIBITORY_NEURON_SIZE = EXCITATORY_NEURON_SIZE // 4
ITER = 4000
"""
Implementation of LIF neuron model.
"""

class LIF(Behavior):
    """
    The neural dynamics of LIF is defined by:
    tau*dv/dt = v_rest - v + R*I,
    if v >= threshold then v = v_reset.
    Args:
        tau (float): time constant of voltage decay.
        v_rest (float): voltage at rest.
        v_reset (float): value of voltage reset.
        threshold (float): the voltage threshold.
        R (float): the resistance of the membrane potential.
    """


    def initialize(self, neurons):
        """
        Set neuron parameters.
        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag("LIF")
        self.set_parameters_as_variables(neurons)
        neurons.v = neurons.vector(mode="ones") * neurons.v_rest
        neurons.spikes = neurons.vector(mode="zeros")

    def _dv_dt(self, neurons):
        """
        Single step voltage dynamics of simple LIF neurons.
        Args:
            neurons (NeuronGroup): the neural population.
        """
        delta_v = neurons.v_rest - neurons.v
        input_current = neurons.R * neurons.I
        return delta_v + input_current

    def _fire(self, neurons):
        """
        Single step of LIF dynamics.
        Args:
            neurons (NeuronGroup): the neural population.
        """
        neurons.spikes = neurons.v >= neurons.threshold
        neurons.v[neurons.spikes] = neurons.v_reset

    def forward(self, neurons):
        """
        Firing behavior of LIF neurons.
        Args:
            neurons (NeuronGroup): the neural population.
        """
        self._fire(neurons)
        neurons.v += self._dv_dt(neurons) / neurons.tau


"""
Implementation of Synapses.
"""
class Synapse(Behavior):

    
    def _set_synapse_weights(self, neurons):
        """
        Set synapse weights.
        Args:
            neurons (NeuronGroup): the neural population.
        """
        for s in neurons.afferent_synapses['All']:
            s.W = s.matrix(mode='uniform', density=neurons.D)
                      

    def initialize(self, neurons):
        """
        Set synapse parameters.
        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag('Synapse')
        self.set_parameters_as_variables(neurons)
        self._set_synapse_weights(neurons)
        
    
    
    
    def _get_presynaptic_inputs(self, synapse):
        """
        Calculate presynaptic inputs of population.
        Args:
            synapse (SynapseGroup): the connections between src and dst neurons.
        """
        spikes = synapse.src.spikes.float()
        return torch.matmul(synapse.W, spikes)
    

    def forward(self, neurons):
        """
        Implementation of both Excitatory and Inhibitory synaptic connections.
        Args:
            neurons (NeuronGroup): the post synaptic neural population.
        """
        for s in neurons.afferent_synapses['GLUTAMATE']: 
            neurons.I += self._get_presynaptic_inputs(s)
            

        for s in neurons.afferent_synapses['GABA']:
            neurons.I -= self._get_presynaptic_inputs(s)


class Input(Behavior):
    @staticmethod
    def _reset_inputs(neurons):
        neurons.I = neurons.vector(mode="zeros")

    def initialize(self, neurons):
        self.add_tag('Input')
        self.input = self.parameter('I', None)
        self._reset_inputs(neurons)


    def forward(self, neurons):
        self._reset_inputs(neurons)
        neurons.I += self.input[neurons.iteration-1]
        

            

"""
Implementation of synaptic trace.
"""

class Trace(Behavior):
    
    def initialize(self, synapse):
        """
        Set trace parameters.
        Args:
            synapse (SynapseGroup): the connections between src and dst neurons .
        """
        self.add_tag("Trace")
        self.set_parameters_as_variables(synapse)
        synapse.src.trace = synapse.src.vector(mode="zeros")
        synapse.dst.trace = synapse.dst.vector(mode="zeros")

    def _get_trace_change(self, s, n):
        """
        trace variables of both pre- and post-synaptic neurons are modified through time with:
        dx/dt = -x/tau + neurons.spikes,
        Args:
            n (NeuronGroup): NeuronGroup that is involved in s
            s (SynapseGroup): the connection between src and dst neurons.
        """
        d_trace = -1 * n.trace/s.tau + n.spikes 
        return d_trace
        
    def _update_spike_trace(self, synapse):
        """
        Single step of spike trace dynamics.  
        Args:
            synapse (SynapseGroup): the connection between src and dst neurons.
        """
        synapse.src.trace += self._get_trace_change(synapse, synapse.src)
        synapse.dst.trace += self._get_trace_change(synapse, synapse.dst)
    
        
    def forward(self, synapse):
        self._update_spike_trace(synapse)
        


class STDP(Behavior):
    """
    Spike-Timing Dependent Plasticity (STDP) rule for simple connections.
    Note: The implementation uses local variables (spike trace).
    Args:
        a_plus (float): Coefficient for the positive weight change. The default is None.
        a_minus (float): Coefficient for the negative weight change. The default is None.
    """

    def initialize(self, synapse):
        self.add_tag("STDP")
        self.enable_soft_bound = self.parameter('enable_soft_bound', False)
        self.w_max = self.parameter("w_max", None)
        self.w_min = self.parameter("w_min", None)
        self.a_plus = self.parameter("a_plus", None)
        self.a_minus = self.parameter("a_minus", None)

    def compute_coefs(self, synapse):
        coef_plus = (self.w_max - synapse.W) * self.a_plus if self.enable_soft_bound else self.a_plus
        coef_minus = (synapse.W - self.w_min) * self.a_minus if self.enable_soft_bound else self.a_minus
        return coef_plus, coef_minus

    def compute_dw(self, s):
        coef_plus, coef_minus = self.compute_coefs(s)
        dw_plus = torch.outer(s.dst.trace, s.src.spikes) * coef_plus
        dw_minus = torch.outer(s.dst.spikes, s.src.trace) * coef_minus
        return dw_plus - dw_minus


    def forward(self, synapse):
        synapse.W += self.compute_dw(synapse)


class anti_STDP(STDP):
    def forward(self, synapse):
        self.compute_coefs(synapse)
        synapse.W += (-1) * self.compute_dw(synapse)

"""
Implementation of inhibitory Spike-Time Dependent Plasticity (iSTDP).
"""


class iSTDP(Behavior):
    """
    the synaptic weights are updated for every pre- and post-synaptic event such that:
        W_ij = W_ij + lr*(x_i - alpha) for pre-synaptic spikes,
        W_ij = W_ij + lr * x_j  for post-synaptic spikes
    where lr is the learning rate, x_i and x_j are post- and pre-synaptic trace, alpha is the depression factor determined by:
        alpha = 2 * freq * tau 
    the freq parameter acts as a target firing rate. The learning rule implements a form of homeostatic plasticity 
    that stabilizes the postsynaptic firing rate.
    Args:
        lr (float): Learning rate.
        freq (float): Constant that determines post-synaptic firing rate.
    """

    def calculate_alpha(self, synapse):
        return 2 * self.freq * synapse.tau / 1000
    
    def initialize(self, synapse):
        self.add_tag("iSTDP")
        self.lr = self.parameter('lr', None)
        self.freq = self.parameter('freq', None)
        self.alpha = self.calculate_alpha(synapse)

    def get_weight_changes(self, s):
        pre_spike_changes = self.lr * (torch.outer(s.dst.trace, s.src.spikes) - self.alpha) 
        post_spike_changes = self.lr * torch.outer(s.dst.spikes, s.src.trace)
        return pre_spike_changes + post_spike_changes


    def forward(self, synapse):
        synapse.W += self.get_weight_changes(synapse)




class WeightClip(Behavior):

    def initialize(self, synapse):
        self.w_min = self.parameter('w_min', 0)
        self.w_max = self.parameter('w_max', 1)
        assert 0 <= self.w_min < self.w_max, "Invalid weight range!"

    def forward(self, synapse):
        synapse.W = torch.clip(synapse.W, self.w_min, self.w_max)





class InputGenerator:


    def __init__(self, mu, std, threshold):
        self.mean = mu
        self.std = std
        self.threshold = threshold
        self.signals = None

    def get_random_input(self, population_size, duration, seed):
        torch.manual_seed(seed)
        enable_input = torch.rand(size=(duration, population_size)) > self.threshold
        random_input = torch.normal(self.mean, self.std, size=(duration, population_size))
        return random_input * enable_input
    
    def get_zero_input(self, population_size, duration):
        return torch.zeros(size=(duration, population_size))
    
    def set_signals(self, signal_duration, population_size, seed):
        torch.manual_seed(seed)
        enable_input = torch.rand(size=(signal_duration, population_size)) > self.threshold
        random_input = torch.normal(self.mean, self.std, size=(signal_duration, population_size))
        self.signals = [enable_input*random_input, ~enable_input*random_input]
    

    def get_signal_over_iter(self, signal, signal_repeat, rest_duration, population_size):
        rest_signal = torch.zeros((rest_duration, population_size))
        signal_with_rest = torch.cat((signal, rest_signal))
        return signal_with_rest.repeat(signal_repeat, 1)
    
    def get_random_signals(self, iter, signal_duration, signal_repeat, rest_duration, population_size, seed):
        iter_duration = (signal_duration + rest_duration) * signal_repeat + rest_duration
        total_duration = iter_duration * iter
        self.signal_orders = []
        self.set_signals(signal_duration, population_size, seed)
        random_signal_input = torch.zeros((total_duration, population_size))
        for i in range(iter):
            start_time = i * (iter_duration)
            end_time = start_time + iter_duration
            signal_no = torch.rand(1).item() > 0.5
            iter_signal = self.get_signal_over_iter(self.signals[signal_no],signal_repeat,rest_duration,population_size)
            iter_rest = torch.zeros((rest_duration, population_size))
            random_signal_input[start_time:end_time, :] = torch.cat((iter_signal,iter_rest))
            self.signal_orders.append(signal_no)
        return random_signal_input


        


class Simulator:
    
    def __init__(self, net, excitatory_pops:list, inhibitory_pops:list, connections:dict, 
                                trace_params:dict, clip_params:dict):

        self.excitatory_pops = excitatory_pops
        self.inhibitory_pops = inhibitory_pops
        self.connections = connections
        self.trace_params, self.clip_params = trace_params, clip_params
        self.net = net

    def add_coonections(self, src_populations:list, dst_populations:list, connection_maps:list, 
                        connection_tag:str):
        for connection in connection_maps:
            src_pop = src_populations[connection['src']]
            dst_pop = dst_populations[connection['dst']]
            learning_rule, learning_params = connection['learning_rule'], connection['learning_params']
            SynapseGroup(net=self.net, src=src_pop, dst=dst_pop, tag=connection_tag, behavior={
                1: Trace(**self.trace_params),
                2: learning_rule(**learning_params) ,
                3: WeightClip(**self.clip_params)  
            })

    def set_coonections(self):
        self.add_coonections(self.inhibitory_pops, self.inhibitory_pops, self.connections['same']['inh'], 'GABA')
        self.add_coonections(self.excitatory_pops, self.excitatory_pops, self.connections['same']['exc'], 'GLUTAMATE')
        self.add_coonections(self.excitatory_pops, self.inhibitory_pops, self.connections['different']['exc_inh'], 'GLUTAMATE')
        self.add_coonections(self.inhibitory_pops, self.excitatory_pops, self.connections['different']['inh_exc'], 'GABA')


    def simulate(self, iter):
        self.set_coonections()
        self.net.initialize()
        self.net.simulate_iterations(iter)
        return self.net

