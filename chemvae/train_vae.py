"""

This version of autoencoder is able to save weights and load weights for the
encoder and decoder portions of the network

"""

# from gpu_utils import pick_gpu_lowest_memory
# gpu_free_number = str(pick_gpu_lowest_memory())
#
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_free_number)

import argparse
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
import yaml
import time
import os
from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
import hyperparameters
import mol_utils as mu
import mol_callbacks as mol_cb
from keras.callbacks import CSVLogger
from models import encoder_model, load_encoder
from models import decoder_model, load_decoder
from models import property_predictor_model, load_property_predictor
from models import variational_layers
from functools import partial
from keras.layers import Lambda



from keras.utils import to_categorical
import numpy as np
DICT = {'5': 29, '=': 22, 'N': 31, 'l': 16, 'H': 18, ']': 3, '@': 21, '6': 1, 'O': 17, 'c': 19, '2': 27, '8': 25, '3': 4, '7': 0, 'I': 15, 'C': 26, 'F': 28, '-': 7, 'P': 24, '/': 9, ')': 13, ' ': 34, '#': 14, 'r': 30, '\\': 33, '1': 20, 'n': 23, '+': 32, '[': 12, 'o': 2, 's': 5, '4': 11, 'S': 8, '(': 6, 'B': 10}
str = "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1"
def one_hot(str, LEN_MAX = 120):
    str = list(str)
    if len(str) < LEN_MAX:
        for i in range(LEN_MAX - len(str)):
            str.append(" ")
    hot  = []
    for char in list(str):
      hot.append(DICT[char])
    return to_categorical(hot)

import pandas as pd
def load_data:
	link1 = '250k_rndm_zinc_drugs_clean_3.csv'
	df1 = pd.read_csv(link1, delimiter=',', names = ['smiles','1','2','3'])
	smiles = list(df1.smiles)[1:]
	X = []
	for smile in smiles:
	  try:
	    X.append(one_hot(smile[:-1]))  
	  except:
	      print ("ahihi do ngoc")
	X = np.array(X)
	print(X.shape)

	id = int (X.shape[0] / 20)
	idx = int (id * 0.8)
	X_train = X[:idx,:,:]
	X_val = X[idx:id,:,:]
	X_test = X[id:id+100,:,:]
	print(X_train.shape)
	print(X_test.shape)
        return X_train, X_val


def load_models(params):

    def identity(x):
        return K.identity(x)

    # def K_params with kl_loss_var
    kl_loss_var = K.variable(params['kl_loss_weight'])

    if params['reload_model'] == True:
        encoder = load_encoder(params)
        decoder = load_decoder(params)
    else:
        encoder = encoder_model(params)
        decoder = decoder_model(params)

    x_in = encoder.inputs[0]

    z_mean, enc_output = encoder(x_in)
    z_samp, z_mean_log_var_output = variational_layers(z_mean, enc_output, kl_loss_var, params)

    # Decoder
    if params['do_tgru']:
        x_out = decoder([z_samp, x_in])
    else:
        x_out = decoder(z_samp)

    x_out = Lambda(identity, name='x_pred')(x_out)
    model_outputs = [x_out, z_mean_log_var_output]

    AE_only_model = Model(x_in, model_outputs)

    if params['do_prop_pred']:
        if params['reload_model'] == True:
            property_predictor = load_property_predictor(params)
        else:
            property_predictor = property_predictor_model(params)

        if (('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ) and
                ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 )):

            reg_prop_pred, logit_prop_pred   = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name='reg_prop_pred')(reg_prop_pred)
            logit_prop_pred = Lambda(identity, name='logit_prop_pred')(logit_prop_pred)
            model_outputs.extend([reg_prop_pred,  logit_prop_pred])

        # regression only scenario
        elif ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ):
            reg_prop_pred = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name='reg_prop_pred')(reg_prop_pred)
            model_outputs.append(reg_prop_pred)

        # logit only scenario
        elif ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 ):
            logit_prop_pred = property_predictor(z_mean)
            logit_prop_pred = Lambda(identity, name='logit_prop_pred')(logit_prop_pred)
            model_outputs.append(logit_prop_pred)

        else:
            raise ValueError('no logit tasks or regression tasks specified for property prediction')

        # making the models:
        AE_PP_model = Model(x_in, model_outputs)
        return AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var

    else:
        return AE_only_model, encoder, decoder, kl_loss_var


def kl_loss(truth_dummy, x_mean_log_var_output):
    x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
    print('x_mean shape in kl_loss: ', x_mean.get_shape())
    kl_loss = - 0.5 * \
        K.mean(1 + x_log_var - K.square(x_mean) -
              K.exp(x_log_var), axis=-1)
    return kl_loss


def main_no_prop(params):
    start_time = time.time()

    X_train, X_test = load_data
    print("---------------------------")
    print(X_train)
    print(X_test.shape)
    print("---------------------------")
    AE_only_model, encoder, decoder, kl_loss_var = load_models(params)

    # compile models
    if params['optim'] == 'adam':
        optim = Adam(lr=params['lr'], beta_1=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optim = RMSprop(lr=params['lr'], rho=params['momentum'])
    elif params['optim'] == 'sgd':
        optim = SGD(lr=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplemented("Please define valid optimizer")

    model_losses = {'x_pred': params['loss'],
                        'z_mean_log_var': kl_loss}

    # vae metrics, callbacks
    vae_sig_schedule = partial(mol_cb.sigmoid_schedule, slope=params['anneal_sigmod_slope'],
                               start=params['vae_annealer_start'])
    vae_anneal_callback = mol_cb.WeightAnnealer_epoch(
            vae_sig_schedule, kl_loss_var, params['kl_loss_weight'], 'vae' )

    csv_clb = CSVLogger(params["history_file"], append=False)
    callbacks = [ vae_anneal_callback, csv_clb]


    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    xent_loss_weight = K.variable(params['xent_loss_weight'])
    print("---------------------------")
    print(X_train)
    model_train_targets = {'x_pred':X_train,
                'z_mean_log_var':np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))}
    model_test_targets = {'x_pred':X_test,
        'z_mean_log_var':np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))}

    AE_only_model.compile(loss=model_losses,
        loss_weights=[xent_loss_weight,
          kl_loss_var],
        optimizer=optim,
        metrics={'x_pred': ['categorical_accuracy',vae_anneal_metric]}
        )

    keras_verbose = params['verbose_print']
    print("=======================")
    print(X_train)
    print(X_test)
    print("=======================")
    AE_only_model.fit(X_train, model_train_targets,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    initial_epoch=params['prev_epochs'],
                    callbacks=callbacks,
                    verbose=keras_verbose,
                    validation_data=[ X_test, model_test_targets]
                    )

    encoder.save(params['encoder_weights_file'])
    decoder.save(params['decoder_weights_file'])
    print('time of run : ', time.time() - start_time)
    print('**FINISHED**')
    print(encoder.summary())
    print("---------------------------")
    print(decoder.summary())
    print("--------------------------")
    print(AE_only_model.summary())
    return 

def main_property_run(params):
    start_time = time.time()

    # load data
    X_train, X_test, Y_train, Y_test = vectorize_data(params)

    # load full models:
    AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var = load_models(params)

    # compile models
    if params['optim'] == 'adam':
        optim = Adam(lr=params['lr'], beta_1=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optim = RMSprop(lr=params['lr'], rho=params['momentum'])
    elif params['optim'] == 'sgd':
        optim = SGD(lr=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplemented("Please define valid optimizer")

    model_train_targets = {'x_pred':X_train,
                'z_mean_log_var':np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))}
    model_test_targets = {'x_pred':X_test,
        'z_mean_log_var':np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))}
    model_losses = {'x_pred': params['loss'],
                        'z_mean_log_var': kl_loss}

    xent_loss_weight = K.variable(params['xent_loss_weight'])
    ae_loss_weight = 1. - params['prop_pred_loss_weight']
    model_loss_weights = {
                    'x_pred': ae_loss_weight*xent_loss_weight,
                    'z_mean_log_var':   ae_loss_weight*kl_loss_var}

    prop_pred_loss_weight = params['prop_pred_loss_weight']


    if ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ):
        model_train_targets['reg_prop_pred'] = Y_train[0]
        model_test_targets['reg_prop_pred'] = Y_test[0]
        model_losses['reg_prop_pred'] = params['reg_prop_pred_loss']
        model_loss_weights['reg_prop_pred'] = prop_pred_loss_weight
    if ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 ):
        if ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ):
            model_train_targets['logit_prop_pred'] = Y_train[1]
            model_test_targets['logit_prop_pred'] = Y_test[1]
        else:
            model_train_targets['logit_prop_pred'] = Y_train[0]
            model_test_targets['logit_prop_pred'] = Y_test[0]
        model_losses['logit_prop_pred'] = params['logit_prop_pred_loss']
        model_loss_weights['logit_prop_pred'] = prop_pred_loss_weight


    # vae metrics, callbacks
    vae_sig_schedule = partial(mol_cb.sigmoid_schedule, slope=params['anneal_sigmod_slope'],
                               start=params['vae_annealer_start'])
    vae_anneal_callback = mol_cb.WeightAnnealer_epoch(
            vae_sig_schedule, kl_loss_var, params['kl_loss_weight'], 'vae' )

    csv_clb = CSVLogger(params["history_file"], append=False)

    callbacks = [ vae_anneal_callback, csv_clb]
    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    # control verbose output
    keras_verbose = params['verbose_print']

    if 'checkpoint_path' in params.keys():
        callbacks.append(mol_cb.EncoderDecoderCheckpoint(encoder, decoder,
                params=params, prop_pred_model = property_predictor,save_best_only=False))

    AE_PP_model.compile(loss=model_losses,
               loss_weights=model_loss_weights,
               optimizer=optim,
               metrics={'x_pred': ['categorical_accuracy',
                    vae_anneal_metric]})


    AE_PP_model.fit(X_train, model_train_targets,
                         batch_size=params['batch_size'],
                         epochs=params['epochs'],
                         initial_epoch=params['prev_epochs'],
                         callbacks=callbacks,
                         verbose=keras_verbose,
         validation_data=[X_test, model_test_targets]
     )

    encoder.save(params['encoder_weights_file'])
    decoder.save(params['decoder_weights_file'])
    property_predictor.save(params['prop_pred_weights_file'])

    print('time of run : ', time.time() - start_time)
    print('**FINISHED**')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_file',
                        help='experiment file', default='exp.json')
    parser.add_argument('-d', '--directory',
                        help='exp directory', default='/home/ntd/Downloads/chemical_vae-master/models/zinc')
    args = vars(parser.parse_args())
    if args['directory'] is not None:
        args['exp_file'] = os.path.join(args['directory'], args['exp_file'])

    params = hyperparameters.load_params(args['exp_file'])
    print("All params:", params)

    if params['do_prop_pred'] :
        main_property_run(params)
    else:
        main_no_prop(params)
