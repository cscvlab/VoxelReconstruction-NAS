########################################################
#                   NAS PARAMETERS                     #
########################################################
CONTROLLER_SAMPLING_EPOCHS =5
SAMPLES_PER_CONTROLLER_EPOCH = 6
CONTROLLER_TRAINING_EPOCHS = 10
ARCHITECTURE_TRAINING_EPOCHS = 40
BEST_ARCHITECTURE_TRAINING_EPOCHS=250
CONTROLLER_LOSS_ALPHA = 0.9
THREAD_ACC=0.001

########################################################
#               CONTROLLER PARAMETERS                  #
########################################################
CONTROLLER_LSTM_DIM = 100
CONTROLLER_OPTIMIZER = 'Adam'
CONTROLLER_LEARNING_RATE = 0.001
CONTROLLER_DECAY = 0.1
CONTROLLER_MOMENTUM = 0.0
CONTROLLER_USE_PREDICTOR = True

########################################################
#                   MLP PARAMETERS                     #
########################################################
MAX_ARCHITECTURE_LENGTH = 7
MLP_OPTIMIZER = 'Adam'
MLP_LEARNING_RATE = 0.001
MLP_DECAY = 0
MLP_MOMENTUM = 0.0
MLP_DROPOUT = 0.2
MLP_LOSS_FUNCTION = 'binary_crossentropy'
MLP_ONE_SHOT = True
MLP_BATCH_SIZE=16384

########################################################
#                   DATA PARAMETERS                    #
########################################################
TARGET_CLASSES = 2

########################################################
#                  OUTPUT PARAMETERS                   #
########################################################
TOP_N = 1

########################################################
#                         PATH                         #
########################################################
CHECK_POINT_PATH= 'CHECKPOINT/check_point.h5'
########################################################
#                     POINTS_TYPE                      #
########################################################
INSIDE_RIGHT = 1
OUTSIDE_RIGHT = 2
INSIDE_WRONG = 3
OUTSIDE_WRONG = 4
INSIDE_RIGHT_COLOR=str(0)+" "+str(0)+" "+str(225)
INSIDE_WRONG_COLOR=str(255)+" "+str(0)+" "+str(0)
OUTSIDE_WRONG_COLOR=str(0) + " " + str(255) + " " + str(0)
