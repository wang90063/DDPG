import tensorflow as tf 
import numpy as np
import math


# Hyper Parameters
LAYER_ENCODER_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64

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

    def __init__(self,sess,state_dim,action_dim,user_num,max_time_step):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.user_num = user_num
        self.max_time_step = max_time_step
        self.batch_size = BATCH_SIZE

        # create actor network

        self.state_input,self.action_output,self.net = self.create_network()

        # create target actor network
        self.target_state_input,self.target_action_output,self.target_update,self.target_net = self.create_target_network(state_dim,action_dim,self.net)

        # define training rules
        self.create_training_method()

        self.sess.run(tf.initialize_all_variables())

        self.update_target()
        #self.load_network()



    def create_training_method(self,user_index):
        self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output[user_index],self.net[user_index],-self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net[user_index]))


    def create_network(self):
        fc_layer_size = LAYER_ENCODER_SIZE
        action_output = []
        weight = []

        # The "state_input" is the state for all the UEs along all the time steps "[time_step, batch_size, user_num, state_dim]"
        state_input = tf.placeholder("float",[None,self.batch_size,self.user_num,self.state_dim])

        action_input = tf.placeholder("float", [None,self.batch_size,self.user_num,self.action_dim])
        action_input_dim = action_input.get_shape().as_list()
        zero_padding = tf.zeros([1, action_input_dim[1],action_input_dim[2],action_input_dim[3]])
        action_input_zero_padded = tf.concat(0,[zero_padding, action_input])[0:action_input_dim[0]-2,:,:,:]

        step_size = tf.placeholder('float', [1])

        initial_lstm_state0 = tf.placeholder(tf.float32, [self.batch_size, self.user_num, fc_layer_size])
        initial_lstm_state1 = tf.placeholder(tf.float32, [self.batch_size, self.user_num, fc_layer_size])

        initial_lstm_state0_list = tf.unpack(initial_lstm_state0, axis=1)
        initial_lstm_state1_list = tf.unpack(initial_lstm_state1, axis=1)

        for i in range(0,self.user_num):
            with tf.variable_scope("ac_net_UE_"  + str(i)) as scope_pi:
                # For each UE/agent, we use '_getitem_' in 'tf.tensor' to derive the input through time, i.e. state_input[:,:,i,:]
                input_s = tf.reshape(state_input[:,:,i,:], [-1, self.state_dim])
                input_a = tf.reshape(action_input_zero_padded[:,:,i,:], [-1,self.action_dim])



                W1_s = tf.get_variable("W1_s", [self.state_dim, fc_layer_size],
                                     initializer=tf.random_uniform([self.state_dim, fc_layer_size], -1 / math.sqrt(self.state_dim),
                                                                   1 / math.sqrt(self.state_dim)))
                W1_a = tf.get_variable("W1_a", [self.action_dim, fc_layer_size],
                                     initializer=tf.random_uniform([self.action_dim, fc_layer_size], -1 / math.sqrt(self.action_dim),
                                                                   1 / math.sqrt(self.action_dim)))

                b1 = tf.get_variable("b1", [fc_layer_size],
                                     initializer=tf.random_uniform([fc_layer_size], -1 / math.sqrt(self.state_dim),
                                                                   1 / math.sqrt(self.state_dim)))

                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(fc_layer_size, state_is_tuple=True)


                h_fc = tf.nn.relu(tf.matmul(input_s,W1_s)+tf.matmul(input_a, W1_a)+b1)

                h_fc1 = tf.reshape(h_fc, [-1,self.batch_size,self.state_dim])

                initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(initial_lstm_state0_list[i],
                                                                   initial_lstm_state1_list[i])


                lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell,
                                                                  h_fc1,
                                                                  initial_state=initial_lstm_state,
                                                                  sequence_length=step_size,
                                                                  time_major=True,
                                                                  scope=scope_pi)






                action_output_agent,weight_agent = self._actor_agent(state_input)
                action_output.append(action_output_agent)
                weight.append(weight_agent)

        return state_input,action_output,weight

    def _actor_agent(self,state_input):
        """
        Encoder layer:
           Before feed the state into the LSTM cell, we use a fully connected layer to encode the state
        """





        layer2_size = LAYER2_SIZE


        W1 = tf.get_variable("W1",[state_dim,layer1_size],initializer=tf.random_uniform([state_dim,layer1_size],-1/math.sqrt(state_dim),1/math.sqrt(state_dim)))
        b1 = tf.get_variable("b1",[layer1_size],initializer=tf.random_uniform([layer1_size],-1/math.sqrt(state_dim),1/math.sqrt(state_dim)))
        W2 = tf.get_variable("W2",[layer1_size,layer2_size],initializer=tf.random_uniform([layer1_size,layer2_size],-1/math.sqrt(layer1_size),1/math.sqrt(layer1_size)))
        b2 = tf.get_variable("b2",[layer2_size],initializer=tf.random_uniform([layer2_size],-1/math.sqrt(layer2_size),1/math.sqrt(layer2_size)))
        W3 = tf.get_variable("W3",[layer2_size,action_dim],initializer=tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3))
        b3 = tf.get_variable("b3",[action_dim],initializer=tf.random_uniform([action_dim],-3e-3,3e-3))

        layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
        action_output_agent = tf.tanh(tf.matmul(layer2,W3) + b3)

        return action_output_agent, [W1,b1,W2,b2,W3,b3]


    def create_target_network(self,state_dim,action_dim,net):
        user_num = USER_NUM
        state_input = tf.placeholder("float",[None,state_dim])
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_net = []
        target_update = []
        action_output = []

        for i in range(1,user_num+1):
            target_update_agent = ema.apply(net[i])
            target_update.append(target_update_agent)
            target_net_agent = [ema.average(x) for x in net[i]]
            target_net.append(target_net_agent)
            layer1 = tf.nn.relu(tf.matmul(state_input, target_net_agent[0]) + target_net_agent[1])
            layer2 = tf.nn.relu(tf.matmul(layer1, target_net_agent[2]) + target_net_agent[3])
            action_output_agent = tf.tanh(tf.matmul(layer2, target_net_agent[4]) + target_net_agent[5])
            action_output.append(action_output_agent)

        return state_input,action_output,target_update,target_net

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,q_gradient_batch,state_batch):
        self.sess.run(self.optimizer,feed_dict={
            self.q_gradient_input:q_gradient_batch,
            self.state_input:state_batch
            })

    def actions(self,state_batch):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:state_batch
            })

    def action(self,state):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:[state]
            })[0]


    def target_actions(self,state_batch):
        return self.sess.run(self.target_action_output,feed_dict={
            self.target_state_input:state_batch
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


