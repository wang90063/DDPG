import tensorflow as tf 
import numpy as np
import math


# Hyper Parameters
LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64
USER_NUM = 100

class ActorNetworkthread:
    """docstring for ActorNetwork"""
    def __init__(self,sess,state_dim_agent,action_dim_agent,thread_index):

        self.sess = sess
        self.state_dim_agent = state_dim_agent
        self.action_dim_agent = action_dim_agent

        # create actor network

        self.state_input,self.action_output,self.net = self.create_network(state_dim_agent,action_dim_agent)

        # create target actor network
        self.target_state_input,self.target_action_output,self.target_update,self.target_net = self.create_target_network(state_dim_agent,action_dim_agent,self.net)

        # define training rules
        self.create_training_method(thread_index)

        self.sess.run(tf.initialize_all_variables())

        self.update_target()
        #self.load_network()

    def create_training_method(self,user_index):
        self.q_gradient_input = tf.placeholder("float",[None,self.action_dim_agent])
        self.parameters_gradients = tf.gradients(self.action_output[user_index],self.net[user_index],-self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net[user_index]))

    def create_network(self,state_dim_agent,action_dim_agent):
        user_num = USER_NUM
        # use two lists to represent the output and weights for each agent
        action_output = []
        weight = []
        state_input = tf.placeholder("float",[None,state_dim_agent])

        for i in range(1,user_num+1):
            with tf.variable_scope("ac_net_agent_"  + str(i)):
                action_output_agent,weight_agent = self._actor_agent(state_dim_agent,action_dim_agent,state_input)
                action_output.append(action_output_agent)
                weight.append(weight_agent)

        return state_input,action_output,weight

    def _actor_agent(self,state_dim_agent,action_dim_agent,state_input):
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        W1 = tf.get_variable("W1",[state_dim_agent,layer1_size],initializer=tf.random_uniform([state_dim_agent,layer1_size],-1/math.sqrt(state_dim_agent),1/math.sqrt(state_dim_agent)))
        b1 = tf.get_variable("b1",[layer1_size],initializer=tf.random_uniform([layer1_size],-1/math.sqrt(state_dim_agent),1/math.sqrt(state_dim_agent)))
        W2 = tf.get_variable("W2",[layer1_size,layer2_size],initializer=tf.random_uniform([layer1_size,layer2_size],-1/math.sqrt(layer1_size),1/math.sqrt(layer1_size)))
        b2 = tf.get_variable("b2",[layer2_size],initializer=tf.random_uniform([layer2_size],-1/math.sqrt(layer2_size),1/math.sqrt(layer2_size)))
        W3 = tf.get_variable("W3",[layer2_size,action_dim_agent],initializer=tf.random_uniform([layer2_size,action_dim_agent],-3e-3,3e-3))
        b3 = tf.get_variable("b3",[action_dim_agent],initializer=tf.random_uniform([action_dim_agent],-3e-3,3e-3))

        layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
        action_output_agent = tf.tanh(tf.matmul(layer2,W3) + b3)

        return action_output_agent, [W1,b1,W2,b2,W3,b3]


    def create_target_network(self,state_dim_agent,action_dim_agent,net):
        user_num = USER_NUM
        state_input = tf.placeholder("float",[None,state_dim_agent])
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


