"""

This version of autoencoder is able to save weights and load weights for the
encoder and decoder portions of the network

"""

# from gpu_utils import pick_gpu_lowest_memory
# gpu_free_number = str(pick_gpu_lowest_memory())
#
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_free_number)
from sklearn.preprocessing import OneHotEncoder
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



def vectorize_data(params):
    # @out : Y_train /Y_test : each is list of datasets.
    #        i.e. if reg_tasks only : Y_train_reg = Y_train[0]
    #             if logit_tasks only : Y_train_logit = Y_train[0]
    #             if both reg and logit_tasks : Y_train_reg = Y_train[0], Y_train_reg = 1
    #             if no prop tasks : Y_train = []

    MAX_LEN = params['MAX_LEN']

    CHARS = yaml.safe_load(open('../models/zinc/'+ params['char_file']))
    params['NCHARS'] = len(CHARS)
    NCHARS = len(CHARS)
    CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))
    print(CHAR_INDICES)
    print("==========------------------------------------=========================================================")
    #INDICES_CHAR = dict((i, c) for i, c in enumerate(CHARS))

    ## Load data for properties
    
    smiles = mu.load_smiles_and_data_df('../models/zinc/'+params['data_file'], MAX_LEN)

    ahihi = sorted(list(CHAR_INDICES.keys()))
    print('Training set size is', len(smiles))
    print('first smiles: \"', smiles[0], '\"')
    print('total chars:', NCHARS)
    enc = OneHotEncoder(handle_unknown='ignore')
    sample = (np.array(ahihi).reshape(-1,1))
    enc.fit(sample)
    enc.categories_
    def one_hot(str, LEN_MAX = 120):
        str = list(str)
        if len(str) < LEN_MAX:
            for i in range(LEN_MAX - len(str)):
                str.append(" ")
        str = np.array(str).reshape(-1,1)
        return enc.transform(str).toarray()

    X = []
    for smile in smiles:
        X.append(one_hot(smile))
    X = np.array(X)
    print('Vectorization...')
    ind = int(X.shape[0] / 126) 
    idx = int(ind * 0.8)
    print(X[0][0])
    X_train = X[:idx*126,:,:]

    X_test = X[idx*126:ind*126,:,:]
    return X_train, X_test


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
    print(encoder.summary())
    print("---------------------------")
    print(decoder.summary())
    print("--------------------------")
    print(AE_only_model.summary())
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

    X_train, X_test = vectorize_data(params)
    print("---------------------------")
    print(X_train.shape)
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

    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_file',
                        help='experiment file', default='exp.json')
    parser.add_argument('-d', '--directory',
                        help='exp directory', default="../models/zinc")
    args = vars(parser.parse_args())
    if args['directory'] is not None:
        args['exp_file'] = os.path.join(args['directory'], args['exp_file'])

    params = hyperparameters.load_params(args['exp_file'])
    print("All params:--------------------------------\n", params)
    print("----------------------------------------")

    main_no_prop(params)
