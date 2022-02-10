import pickle
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from CONSTANTS import *
from controller import Controller
from mlp_generator import MLPGenerator

from utils import *


class MLPNAS(Controller):

    def __init__(self,train_x,train_y, validation_x, validation_y,weights=None):
        self.train_x = train_x
        self.train_y = train_y
        self.validation_x=validation_x
        self.validation_y=validation_y
        self.weights=weights
        self.target_classes = TARGET_CLASSES
        self.controller_sampling_epochs = CONTROLLER_SAMPLING_EPOCHS
        self.samples_per_controller_epoch = SAMPLES_PER_CONTROLLER_EPOCH
        self.controller_train_epochs = CONTROLLER_TRAINING_EPOCHS
        self.architecture_train_epochs = ARCHITECTURE_TRAINING_EPOCHS
        self.best_architecture_train_epochs =BEST_ARCHITECTURE_TRAINING_EPOCHS
        self.controller_loss_alpha = CONTROLLER_LOSS_ALPHA
        self.data = []
        self.nas_data_log = 'LOGS/nas_data.pkl'
        self.checkpoint_path=CHECK_POINT_PATH
        clean_log()

        super().__init__()

        self.model_generator = MLPGenerator()

        self.controller_batch_size = len(self.data)
        self.controller_input_shape = (1, MAX_ARCHITECTURE_LENGTH - 1)
        if self.use_predictor:
            self.controller_model = self.hybrid_control_model(self.controller_input_shape, self.controller_batch_size)
        else:
            self.controller_model = self.control_model(self.controller_input_shape, self.controller_batch_size)

    def create_architecture(self, sequence):
        if self.target_classes == 2:
            self.model_generator.loss_func = 'binary_crossentropy'
        model = self.model_generator.create_model(sequence, np.shape(self.train_x[0]))
        model = self.model_generator.compile_model(model)
        return model

    def train_architecture(self,model,epochs,callbacks=None):
        # x, y = unison_shuffled_copies(self.x, self.y,self.weights)
        # path=os.path.join(SAMPLE_POINTS_PATH,POINTS_FILE_NAME)
        # save_points(x,y,path)
        train_x=self.train_x
        train_y=self.train_y
        validation_x=self.validation_x
        validation_y=self.validation_y
        history = self.model_generator.train_model(model, train_x, train_y,validation_x,validation_y,self.weights,nb_epochs=epochs,callbacks=callbacks)
        return history

    def append_model_metrics(self, sequence, history,size=None,pred_accuracy=None):
        if len(history.history['val_accuracy']) == 1:
            if pred_accuracy:
                self.data.append([sequence,
                                  history.history['val_accuracy'][0],
                                  size,
                                  pred_accuracy])
            else:
                self.data.append([sequence,
                                  history.history['val_accuracy'][0],
                                  size]
                                  )
            print('validation accuracy: ', history.history['val_accuracy'][0])
        else:
            val_acc = np.ma.average(history.history['val_accuracy'],
                                    weights=np.arange(1, len(history.history['val_accuracy']) + 1),
                                    axis=-1)
            if pred_accuracy:
                self.data.append([sequence,
                                  val_acc,
                                  size,
                                  pred_accuracy])
            else:
                self.data.append([sequence,
                                  size,
                                  val_acc])
            print('validation accuracy: ', val_acc)

    def prepare_controller_data(self, sequences):
        controller_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        xc = controller_sequences[:, :-1].reshape(len(controller_sequences), 1, self.max_len - 1)
        yc = to_categorical(controller_sequences[:,-1], self.controller_classes)
        val_acc_target = [item[1] for item in self.data]
        return xc, yc, val_acc_target


    def custom_loss(self, target, output):
        baseline = 0.98
        nisize=7553
        maxsize=21121
        reward = np.array([((item[1] - baseline))+((nisize-item[2])/maxsize) for item in self.data[-self.samples_per_controller_epoch:]]).reshape(
            self.samples_per_controller_epoch, 1)
        loss = - K.log(output) * reward[:, None]
        return loss

    def train_controller(self, model, x, y, pred_accuracy=None):
        if self.use_predictor:
            self.train_hybrid_model(model,
                                    x,
                                    y,
                                    pred_accuracy,
                                    self.custom_loss,
                                    len(self.data),
                                    self.controller_train_epochs)
        else:
            self.train_control_model(model,
                                     x,
                                     y,
                                     self.custom_loss,
                                     len(self.data),
                                     self.controller_train_epochs)

    def search(self):
        for controller_epoch in range(self.controller_sampling_epochs):
            print('------------------------------------------------------------------')
            print('                       CONTROLLER EPOCH: {}'.format(controller_epoch))
            print('------------------------------------------------------------------')
            sequences = self.sample_architecture_sequences(self.controller_model, self.samples_per_controller_epoch)
            if self.use_predictor:
                pred_accuracies = self.get_predicted_accuracies_hybrid_model(self.controller_model, sequences)
                print("pred_accuracies:",pred_accuracies)
            for i, sequence in enumerate(sequences):
                print('Architecture: ', self.decode_sequence(sequence))
                model = self.create_architecture(sequence)
                history = self.train_architecture(model,self.architecture_train_epochs)
                size=architectrue_size(self.decode_sequence(sequence))
                print(history)

                if self.use_predictor:
                    self.append_model_metrics(sequence, history,size=size,pred_accuracy=pred_accuracies[i])
                else:
                    self.append_model_metrics(sequence, history,size=size)
                print('------------------------------------------------------')
            xc, yc, val_acc_target = self.prepare_controller_data(sequences)
            self.train_controller(self.controller_model,
                                  xc,
                                  yc,
                                  val_acc_target[-self.samples_per_controller_epoch:])


        best_sequence = get_best_sequence(self.data)
        log_best_architectrue(best_sequence)
        bestmodel = self.create_architecture(best_sequence)

        metric="val_accuracy"
        checkpoint=ModelCheckpoint(self.checkpoint_path, monitor=metric,verbose=1, save_best_only=True, mode='max')
        callbacks = [checkpoint]
        history = self.train_architecture(bestmodel,self.best_architecture_train_epochs,callbacks)


        bestmodel.load_weights(self.checkpoint_path)
        bestmodel.save('LOGS/mlp_model.h5')
        with open(self.nas_data_log, 'wb') as f:
            pickle.dump(self.data, f)
        return self.data
