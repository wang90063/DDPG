
import sys
import random
import numpy as np


from collections import Counter
from math import exp
from constants import num_user
from constants import num_rb


#The parameters for systemModel
# 400 pixels represent the largest distance of the area, i.e. 100m
WINDOW_WIDTH = 400  # size of window's width in pixels
WINDOW_HEIGHT = 400  # size of windows' height in pixels
cell_radius = 25 # "m" this is the radius of cell
resolution  = cell_radius * 4./WINDOW_WIDTH # meter/pixel, the longest distace in the simulation system is "cell_radius * 4"

outlayer_userrange_x_low = -30 / resolution
outlayer_userrange_x_high = 30 / resolution

outlayer_userrange_y_low = -30 / resolution
outlayer_userrange_y_high = 30 / resolution

outer_radius = 50/ resolution
inner_radius = 22.5 / resolution
Num_CELL = 6

noise = 1.8 * 10**(-12.4)

p_max = 10**(2.1)
# generate cell list, every cell in the lists consists of 5 params:
# (1)loc_x(left) (2)loc_y(top) (3)cell_id
# :return: cell_list: (1)loc_x(left) (2)loc_y(top) (3)cell_id

# cell location
cell_id = np.arange(Num_CELL)

# the locations of cells are fixed and the coordinates are given
# cell_x = [200, 200, 370, 370, 200, 30, 30]
# cell_y = [200, 0, 100, 300, 400, 300, 100]

cell_x = [ 0.0, 170.0, 170.0, 0.0, -170, -170]
cell_y = [ 200.0, 100.0, -100.0, -200.0, -100, 100]

cells = np.vstack((cell_x, cell_y,cell_id)).T


# "action" is the "cell_id" selected by the agent
class SystemModel:
  def __init__(self):
      self.init_users()
      self.handover_indicator = 0  #np.zeros(LOCAL_T_MAX)
      self.reward_handover = 0
      self.handover_consumption =2.3#0.43#2.3
      self.terminal = False
      self.num_user = num_user
      self.num_rb = num_rb
      self.cells = cells
      self.num_cell = Num_CELL
      self.p_max = p_max


  def state_update(self, last_action, action):
      """
      the func can generate the reward and state, also update the users locations
      :param: users:the locations of the users
              action: the "cell_id" selected by users
              last_action
      """
      s_t = self._get_state(last_action)
      r= self._get_reward(last_action,action)
      self._move_user()
      s_t1 = self._get_state(action)
      self.reward = r
      self.s_t = s_t
      self.s_t1 = s_t1


  def update(self):
      self.s_t = self.s_t1


  def _move_user(self):
      """
      low mobility users are considered, i.e. the user only move one pixel every frame. different mobility trajectories will be tested to present the robustness of the neural network
      """
      self.terminal = False
      mobility_speed = 1

      for i in range(self.num_user):
          move_x = random.randint(-mobility_speed, mobility_speed)
          user_x_tmp = self.users[i][0] + move_x
          move_y = random.randint(-mobility_speed, mobility_speed)
          user_y_tmp = self.users[i][1] + move_y

          if np.abs(user_x_tmp) > np.sqrt(3) / 2.0 * outer_radius or (np.abs(user_x_tmp) + np.abs(user_y_tmp) / np.sqrt(
                  3)) > outer_radius:  # and (np.abs(user_x_tmp) > np.sqrt(3)/2.0*inner_radius or (np.abs(user_x_tmp)+np.abs(user_y_tmp)/np.sqrt(3)) > inner_radius):
              self.terminal = True
              user_x = user_x_tmp
              user_y = user_y_tmp
          else:
              user_x = user_x_tmp
              user_y = user_y_tmp

          self.users[i] = np.hstack((user_x, user_y))

  def _get_reward(self, last_action, action):
      """
      :param users: the location of user before moving
      :param action: the taken action to obtain "users"
      :return: reward : the weighted sum of rate and reward for handover, i.e. "handover error occurs" -- 0, "handover successes" -- 1
      """
      self.handover_indicator = np.zeros(self.num_user)
      reward = np.zeros(self.num_user)
      for i in  range(self.num_user):

          if action[i] == last_action[i]:
              self.handover_indicator[i] = 0
          else:
              self.handover_indicator[i] = 1
          reward[i] = self.rates[i][action[i][1]] - self.handover_indicator[i] * self.handover_consumption
      return  reward


  def _get_state(self, action):
      # The dimension of the input "action" is [user_num, action_dim]
      # The vector 'action[i]' consists of three elements: '[resource_pattern, SBS_index,transmit_power]'
      # The 'action' is the action after refining, so 'resource_pattern, SBS_index' are intergers
     feature_reource_bitmap = np.zeros([self.num_user, self.num_rb])
     feature_serving_bs_index = np.zeros([self.num_user,self.num_cell])
     resource_length = np.zeros(self.num_user)
     feature_transmit_power = np.zeros(self.num_user)
     for i in range(self.num_user):
         feature_reource_bitmap[i], resource_length[i] = self.convert_pattern_to_indexvec_ue(action[i][0])
         feature_serving_bs_index[i] [action[i][1]] = 1
         feature_transmit_power[i] = action[i][2]
     rates = self.get_rate_all_cell(self.users, self.cells, feature_reource_bitmap, resource_length, feature_transmit_power)
     s_t = np.hstack((feature_reource_bitmap,feature_serving_bs_index,feature_transmit_power,rates))
     return s_t

  def init_users(self):
    """
    initialize user. every user consists of 4 params:
    (1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
    user mobility type is divided into 3 categories: low, medium and high. Low mobility users takes 70% of all,
    while medium and high takes 20% and 10%.
    :return: user: (1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
    """
    users_list = []
    resource_id_vector_list = []
    resource_length_list = []
    resource_pattern = np.random.randint(0, self.num_rb*(self.num_rb+1)/2+1, size = self.num_user)
    transmit_power = np.random.rand(self.num_user)*p_max
    for i in range(self.num_user):
        while True:
            user_x_tmp = np.random.randint(outlayer_userrange_x_low, outlayer_userrange_x_high + 1, size=1,
                                           dtype='int')
            user_y_tmp = np.random.randint(outlayer_userrange_y_low, outlayer_userrange_y_high + 1, size=1,
                                           dtype='int')
            if np.abs(user_x_tmp) < np.sqrt(3) / 2.0 * outer_radius and (
                np.abs(user_x_tmp) + np.abs(user_y_tmp) / np.sqrt(3)) < outer_radius:  # and (np.abs(user_x_tmp) > np.sqrt(3)/2.0*inner_radius or (np.abs(user_x_tmp)+np.abs(user_y_tmp)/np.sqrt(3)) > inner_radius):
                user_x = user_x_tmp
                user_y = user_y_tmp
                break
        users_list.append(np.hstack((user_x, user_y)))
        resource_id_vector, resource_length = self.convert_pattern_to_indexvec_ue(resource_pattern[i])
        resource_id_vector_list.append(resource_id_vector)
        resource_length_list.append(resource_length)

    resource_id_matrix = np.asarray(resource_id_vector_list)
    resource_length_vec = np.asarray(resource_length_list)


    # the locations of all the users initially
    self.users = np.asarray(users_list)

    self.rates = self.get_rate_all_cell(self.users, self.cells,resource_id_matrix,resource_length_vec,transmit_power)

    actions_before_initial_state = np.concatenate((resource_pattern, np.argmax(self.rates,axis=1) ,transmit_power),axis=1)


    # Initial states for all the users
    self.s_t = self._get_state(action=actions_before_initial_state)



  def convert_pattern_to_indexvec_ue (self,resource_pattern):
      # input 'resource_pattern' is a integer
      resource_id_vector = np.zeros(self.num_rb)
      for num in range(self.num_rb):
          if resource_pattern >= num*(21-num)/2 and resource_pattern < self.num_rb-1+num*(19-num)/2:
              resource_length = num+1
              resource_id_vector[resource_pattern-num*(21-num)/2:resource_pattern-num*(21-num)/2+num]=1
              break

      return resource_id_vector,resource_length



  def get_rate_all_cell(self, users, cells, resource_id, resource_length, transmit_power):
      """
      get the rates of the user in all the cells if this user connects to the cell. return the array "rate" to represent the rate in the cells
      """
      channels_square = np.random.rayleigh(1,[self.num_user, self.num_cell,self.num_rb])  # the fast fading from the user to all the cells
      distance = np.zeros([self.num_user, self.num_cell])
      fadings = np.zeros([self.num_user,self.num_cell,self.num_rb])
      rates = np.zeros([self.num_user, self.num_cell])
      for i in range(self.num_user):
         for j in cell_id:
          # print(num)
          # print (cells[num][0] - users[0]) ** 2
          # print (cells[num][1] - users[1]) ** 2
          # print np.sqrt((cells[num][0] - users[0]) ** 2 + (cells[num][1] - users[1]) ** 2) * resolution / 20.0
             distance[i,j] = np.sqrt((cells[j][0] - users[i][0]) ** 2 + (cells[j][1] - users[i][1]) ** 2) * resolution  # calculate the distance between user and each base station
             fadings = channels_square[i,j]*(distance[i,j]**-4)

      for i in range(self.num_user):
          for j in range(resource_length[i]):
              interference_rb_all_cell = 0
              for m in range(np.nonzero(resource_id[:,np.nonzero(resource_id[i])[0][j]])[0].shape[0]):
                  if m != i:
                      interference_rb_all_cell += fadings[m,:,np.nonzero(resource_id[i])[0][j]] * transmit_power[m]/resource_length[m]
              snr_rb_all_cell = fadings[i,:,np.nonzero(resource_id[i])[0][j]] * transmit_power[i]/resource_length[i]/(interference_rb_all_cell+noise)
              rates[i] += np.log2(1+snr_rb_all_cell)

      return rates


