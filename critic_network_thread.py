
import tensorflow as tf 
import numpy as np
import math


ENCODER_LAYER1_SIZE = 400
ENCODER_LAYER2_SIZE = 300
COMMUN_LAYER3_SIZE = 200

LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01

USER_NUM = 100

class CriticNetworkthread:
    """docstring for CriticNetwork"""
    def __init__(self,sess,state_dim_agent,action_dim_agent,thread_index):
        self.time_step = 0
        self.sess = sess
        # create q network
        self.state_input,\
        self.action_input, \
        self.q_value_output,\
        self.net = self.create_q_network(state_dim_agent,action_dim_agent)

        # create target q network (the same structure with q network)
        self.target_state_input,\
        self.target_action_input, \
        self.target_q_value_output,\
        self.target_update = self.create_target_q_network(state_dim_agent,action_dim_agent,self.net)

        self.create_training_method(thread_index)

        # initialization
        self.sess.run(tf.initialize_all_variables())

        self.update_target()

    def create_training_method(self,user_index):
        # Define training optimizer
        self.y_input = tf.placeholder("float",[None,1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net[user_index]])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output[user_index])) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output[user_index],self.action_input[user_index])

    def create_q_network(self,state_dim_agent,action_dim_agent):
        user_num = USER_NUM

        state_input = tf.placeholder("float",[None,state_dim_agent])
        action_input = tf.placeholder("float",[None,action_dim_agent])

        # use two lists to represent the output and weights for all agents
        q_value_output = []
        weight = []

        for i in range(1,user_num+1):
           with tf.variable_scope("cr_net_agent_"  + str(i)):
                q_value_output_agent,weight_agent = self._critic_agent(state_dim_agent,action_dim_agent,state_input,action_input)
                q_value_output.append(q_value_output_agent)
                weight.append(weight_agent)


        return state_input,action_input,q_value_output,weight


    def _critic_agent(self,state_dim_agent,action_dim_agent,state_input,action_input):
        # the layer size could be changed
        encoder_layer1_size = ENCODER_LAYER1_SIZE
        encoder_layer2_size = ENCODER_LAYER2_SIZE
        user_num = USER_NUM
        # encoder layer parameters
        W1 = tf.get_variable("W1",[state_dim_agent,encoder_layer1_size],initializer=tf.random_uniform([state_dim_agent,encoder_layer1_size],-1/math.sqrt(state_dim_agent),1/math.sqrt

        (state_dim_agent)))
        b1 = tf.get_variable("b1",[encoder_layer1_size],initializer=tf.random_uniform([encoder_layer1_size],-1/math.sqrt(state_dim_agent),1/math.sqrt(state_dim_agent)))
        W2 = tf.get_variable("W2",[encoder_layer1_size,encoder_layer2_size],initializer=tf.random_uniform([encoder_layer1_size,encoder_layer2_size],-1/math.sqrt(encoder_layer1_size + action_dim_agent*user_num),1/math.sqrt(encoder_layer1_size + action_dim_agent*user_num)))
        W2_action = tf.get_variable("W2_action",[action_dim_agent*user_num, encoder_layer2_size],initializer=tf.random_uniform([action_dim_agent*user_num, encoder_layer2_size],-1/math.sqrt(encoder_layer1_size + action_dim_agent*user_num),1/math.sqrt(encoder_layer1_size + action_dim_agent*user_num)))
        b2 = tf.get_variable("b2",[encoder_layer2_size],initializer=tf.random_uniform([encoder_layer2_size],-1/math.sqrt(encoder_layer1_size + action_dim_agent*user_num),1/math.sqrt(encoder_layer1_size + action_dim_agent*user_num)))


        # output layer parameters
        W3 = tf.get_variable("W3",[encoder_layer2_size, 1],initializer=tf.random_uniform([encoder_layer2_size, 1],-3e-3, 3e-3))
        b3 = tf.get_variable("b3",[ 1],initializer=tf.random_uniform([1],-3e-3, 3e-3))

        # encoder layer
        encoder_layer1_output = tf.nn.relu(tf.matmul(state_input, W1) + b1)
        encoder_layer2_output = tf.nn.relu(tf.matmul(encoder_layer1_output, W2) + tf.matmul(action_input, W2_action) + b2)

        # output layer
        q_value_output_agent = tf.identity(tf.matmul(encoder_layer2_output, W3) + b3)

        return  q_value_output_agent, [W1,b1,W2,W2_action,b2,W3,b3]


    def create_target_q_network(self,state_dim_agent,action_dim_agent,net):
        user_num = USER_NUM
        state_input = tf.placeholder("float",[None,state_dim_agent])
        action_input = tf.placeholder("float",[None,action_dim_agent])
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_net = []
        target_update = []
        q_value_output = []
        for i in range(1,user_num+1):
            target_update_agent = ema.apply(net[i])
            target_update.append(target_update_agent)
            target_net_agent = [ema.average(x) for x in net[i]]
            target_net.append(target_net_agent)
            encoder_layer1_output = tf.nn.relu(tf.matmul(state_input, target_net_agent[0]) + target_net_agent[1])
            encoder_layer2_output = tf.nn.relu(tf.matmul(encoder_layer1_output, target_net_agent[2]) + tf.matmul(action_input, target_net_agent[3]) + target_net_agent[4])
            q_value_output_agent = tf.identity(tf.matmul(encoder_layer2_output,target_net_agent[5]) + target_net_agent[6])
            q_value_output.append(q_value_output_agent)

            return state_input,action_input,q_value_output,target_update

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,y_batch,state_batch,action_batch):
        self.time_step += 1
        self.sess.run(self.optimizer,feed_dict={
            self.y_input:y_batch,
            self.state_input:state_batch,
            self.action_input:action_batch,
            })

    def gradients(self,state_batch,action_batch):
        return self.sess.run(self.action_gradients,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch,
            })[0]

    def target_q(self,state_batch,action_batch):
        return self.sess.run(self.target_q_value_output,feed_dict={
            self.target_state_input:state_batch,
            self.target_action_input:action_batch,
            })

    def q_value(self,state_batch,action_batch):
        return self.sess.run(self.q_value_output,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch})


'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

    def save_network(self,time_step):
        print 'save critic-network...',time_step
        self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step = time_step)
'''
