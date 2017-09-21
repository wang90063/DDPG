import tensorflow as tf 
import numpy as np
import math


# Hyper Parameters
from constants import LAYER_ENCODER_SIZE
from constants import LEARNING_RATE
from constants import TAU
from constants import BATCH_SIZE
from constants import max_time_step

class ActorNetwork:

    """Actor network generation

    This actor network is composed of 'user_num' independent LSTMs unrolled through the time steps.

    Parameters

    ----------

    state_dim : int
              The dimension for the state input for LSTM cell
    action_dim : int
              The dimension for the action output for LSTM cell
    user_num : int
               The total number of UE
    max_time_step : int
               The max unrolled time period

    """

    def __init__(self,sess,state_dim,action_dim,user_num):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.user_num = user_num
        self.max_time_step = max_time_step
        self.batch_size = BATCH_SIZE
        self.fc_layer_size = LAYER_ENCODER_SIZE

        # create actor network

        self.state_input,\
        self.action_output, \
        self.net, \
        self.lstm_state, \
        self.lstm_outputs, \
        self.initial_lstm_state, \
        self.step_size = self.create_network()

        # create target actor network
        self.target_state_input,\
        self.target_action_output, \
        self.target_net, \
        self.target_lstm_state,\
        self.target_lstm_outputs, \
        self.target_initial_lstm_state,\
        self.target_step_size = self.create_target_network()

        # define training rules
        self.create_training_method()

        self.sess.run(tf.initialize_all_variables())

        #self.load_network()



    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float",[None,self.user_num,self.action_dim])
        self.q_gradient_input_list=tf.unpack(self.q_gradient_input,axis=2)
        self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))


    def create_network(self):

        weight = []

        # The "state_input" is the state for all the UEs along all the time steps "[time_step, batch_size, user_num, state_dim]"
        state_input = tf.placeholder("float",[None,self.user_num,self.state_dim])

        action_input = tf.placeholder("float", [None,self.user_num,self.state_dim])

        step_size = tf.placeholder("float", [1])

        initial_lstm_state = tf.placeholder(tf.float32, [2, self.user_num,self.fc_layer_size])

        initial_lstm_state_list = tf.unpack(initial_lstm_state, axis=0)

        initial_lstm_state_input = tf.nn.rnn_cell.LSTMStateTuple(initial_lstm_state_list[0], initial_lstm_state_list[1])

        with tf.variable_scope("ac_net_UE_") as scope_pi:
            # For each UE/agent, we use '_getitem_' in 'tf.tensor' to derive the input through time, i.e. state_input[:,:,i,:]
            input_s = tf.reshape(state_input, [-1, self.state_dim])
            input_a = tf.reshape(action_input, [-1, self.action_dim])

            W1_s = tf.get_variable("W1_s", [self.state_dim, self.fc_layer_size],
                                   initializer=tf.random_uniform([self.state_dim, self.fc_layer_size],
                                                                 -1 / math.sqrt(self.state_dim),
                                                                 1 / math.sqrt(self.state_dim)))
            W1_a = tf.get_variable("W1_a", [self.action_dim, self.fc_layer_size],
                                   initializer=tf.random_uniform([self.action_dim, self.fc_layer_size],
                                                                 -1 / math.sqrt(self.action_dim),
                                                                 1 / math.sqrt(self.action_dim)))

            b1 = tf.get_variable("b1", [self.fc_layer_size],
                                 initializer=tf.random_uniform([self.fc_layer_size], -1 / math.sqrt(self.state_dim),
                                                               1 / math.sqrt(self.state_dim)))

            W2 = tf.get_variable("W2", [self.fc_layer_size, self.action_dim],
                                 initializer=tf.random_uniform([self.fc_layer_size, self.action_dim],
                                                               -1 / math.sqrt(self.fc_layer_size),
                                                               1 / math.sqrt(self.fc_layer_size)))
            b2 = tf.get_variable("b2", [self.action_dim],
                                 initializer=tf.random_uniform([self.action_dim], -1 / math.sqrt(self.fc_layer_size),
                                                               1 / math.sqrt(self.fc_layer_size)))

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.fc_layer_size, state_is_tuple=True)

            h_fc = tf.nn.relu(tf.matmul(input_s, W1_s) + tf.matmul(input_a, W1_a) + b1)

            h_fc1 = tf.reshape(h_fc, [-1, self.user_num, self.fc_layer_size])

            initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(initial_lstm_state)

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell,
                                                         h_fc1,
                                                         initial_state=initial_lstm_state_input,
                                                         sequence_length=step_size,
                                                         time_major=True,
                                                         scope=scope_pi)

            lstm_outputs_re = tf.reshape(lstm_outputs, [-1, self.fc_layer_size])

            # The dimension of 'action_output_UE' is 'time_step * batch_size'

            action_output = tf.reshape(tf.tanh(tf.matmul(lstm_outputs_re, W2) + b2),
                                          [-1, self.user_num, self.action_dim])


            scope_pi.reuse_variables()
            W_lstm = tf.get_variable("BasicLSTMCell/Linear/Matrix")
            b_lstm = tf.get_variable("BasicLSTMCell/Linear/Bias")
            weight_UE = [W1_s, W1_a, b1, W_lstm, b_lstm, W2, b2, ]

            weight.append(weight_UE)

        # the dimension of 'action_output' is [user_num, time_step, batch_size, action_dim]

        return state_input,action_output, weight, lstm_state, lstm_outputs, initial_lstm_state, step_size


    def create_target_network(self):
        weight = []

        # The "state_input" is the state for all the UEs along all the time steps "[time_step, batch_size, user_num, state_dim]"
        state_input = tf.placeholder("float", [None, self.user_num, self.state_dim])

        action_input = tf.placeholder("float", [None, self.user_num, self.state_dim])

        step_size = tf.placeholder("float", [1])

        initial_lstm_state = tf.placeholder(tf.float32, [2, self.user_num, self.fc_layer_size])

        initial_lstm_state_list = tf.unpack(initial_lstm_state, axis=0)

        initial_lstm_state_input =tf.nn.rnn_cell.LSTMStateTuple(initial_lstm_state_list[0], initial_lstm_state_list[1])

        with tf.variable_scope("target_ac_net_UE_") as scope_pi:
            # For each UE/agent, we use '_getitem_' in 'tf.tensor' to derive the input through time, i.e. state_input[:,:,i,:]
            input_s = tf.reshape(state_input, [-1, self.state_dim])
            input_a = tf.reshape(action_input, [-1, self.action_dim])

            W1_s = tf.get_variable("W1_s", [self.state_dim, self.fc_layer_size],
                                   initializer=tf.random_uniform([self.state_dim, self.fc_layer_size],
                                                                 -1 / math.sqrt(self.state_dim),
                                                                 1 / math.sqrt(self.state_dim)))
            W1_a = tf.get_variable("W1_a", [self.action_dim, self.fc_layer_size],
                                   initializer=tf.random_uniform([self.action_dim, self.fc_layer_size],
                                                                 -1 / math.sqrt(self.action_dim),
                                                                 1 / math.sqrt(self.action_dim)))

            b1 = tf.get_variable("b1", [self.fc_layer_size],
                                 initializer=tf.random_uniform([self.fc_layer_size], -1 / math.sqrt(self.state_dim),
                                                               1 / math.sqrt(self.state_dim)))

            W2 = tf.get_variable("W2", [self.fc_layer_size, self.action_dim],
                                 initializer=tf.random_uniform([self.fc_layer_size, self.action_dim],
                                                               -1 / math.sqrt(self.fc_layer_size),
                                                               1 / math.sqrt(self.fc_layer_size)))
            b2 = tf.get_variable("b2", [self.action_dim],
                                 initializer=tf.random_uniform([self.action_dim], -1 / math.sqrt(self.fc_layer_size),
                                                               1 / math.sqrt(self.fc_layer_size)))

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.fc_layer_size, state_is_tuple=True)

            h_fc = tf.nn.relu(tf.matmul(input_s, W1_s) + tf.matmul(input_a, W1_a) + b1)

            h_fc1 = tf.reshape(h_fc, [-1, self.user_num, self.fc_layer_size])

            initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(initial_lstm_state)

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell,
                                                         h_fc1,
                                                         initial_state=initial_lstm_state_input,
                                                         sequence_length=step_size,
                                                         time_major=True,
                                                         scope=scope_pi)

            lstm_outputs_re = tf.reshape(lstm_outputs, [-1, self.fc_layer_size])

            # The dimension of 'action_output_UE' is 'time_step * batch_size'

            action_output = tf.reshape(tf.tanh(tf.matmul(lstm_outputs_re, W2) + b2),
                                       [-1, self.user_num, self.action_dim])

            scope_pi.reuse_variables()
            W_lstm = tf.get_variable("BasicLSTMCell/Linear/Matrix")
            b_lstm = tf.get_variable("BasicLSTMCell/Linear/Bias")
            weight_UE = [W1_s, W1_a, b1, W_lstm, b_lstm, W2, b2, ]

            weight.append(weight_UE)

        return state_input,action_output, weight, lstm_state, lstm_outputs, initial_lstm_state, step_size

    def reset_state(self):
        self.lstm_state = tf.nn.rnn_cell.LSTMStateTuple(np.zeros([ self.user_num, self.fc_layer_size]),
                                                            np.zeros([ self.user_num, self.fc_layer_size]))

    def train(self,q_gradient_batch,state_batch,time_step):
        self.sess.run(self.optimizer,feed_dict={
            self.q_gradient_input:q_gradient_batch,
            self.state_input:state_batch,
            self.initial_lstm_state: self.lstm_state,
            self.step_size: time_step
            })


    def train_target(self):
         self.sess.run ([
                self.target_net[0].assign(TAU*self.net[0]+(1-TAU)*self.target_net[0]),
                self.target_net[1].assign(TAU*self.net[1]+(1-TAU)*self.target_net[1]),
				self.target_net[2].assign(TAU*self.net[2]+(1-TAU)*self.target_net[2]),
				self.target_net[3].assign(TAU*self.net[3]+(1-TAU)*self.target_net[3]),
				self.target_net[4].assign(TAU*self.net[4]+(1-TAU)*self.target_net[4]),
				self.target_net[5].assign(TAU*self.net[5]+(1-TAU)*self.target_net[5]),
                self.target_net[6].assign(TAU*self.net[6]+(1-TAU)*self.target_net[6])
         ])


    def actions(self,state_batch, lstm_state,time_step):
        return self.sess.run(self.action_output, self.lstm_state, self.lstm_outputs,feed_dict={
            self.state_input:state_batch,
            self.initial_lstm_state: lstm_state,
            self.step_size: time_step
            })

    # def action(self,state):
    #     return self.sess.run(self.action_output,feed_dict={
    #         self.state_input:[state] self.max_time_step
    #         })[0]


    def target_actions(self,state_batch, target_lstm_state,time_step):
        return self.sess.run(self.target_action_output, self.target_lstm_state, self.target_lstm_outputs    ,feed_dict={
            self.target_state_input:state_batch,
            self.target_initial_lstm_state: target_lstm_state,
            self.target_step_size: time_step
            })

'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
    def save_network(self,time_step):
        print 'save actor-network...',time_step
        self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''


